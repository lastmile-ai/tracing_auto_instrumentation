import logging
from typing import Any, Dict, Callable, Collection, Optional, Type

from langchain_core.callbacks import BaseCallbackManager
from langchain_core.tracers.schemas import Run
from openinference.instrumentation.langchain._tracer import (
    OpenInferenceTracer,
    _as_utc_nano,
    _update_span,
)
from openinference.instrumentation.langchain.package import _instruments
from openinference.instrumentation.langchain.version import __version__
from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.sdk.trace import Span
from opentelemetry.instrumentation.instrumentor import (
    BaseInstrumentor,
)  # type: ignore
from wrapt import wrap_function_wrapper

# TODO: Fix typing
from lastmile_eval.rag.debugger.tracing.sdk import get_lastmile_tracer

from lastmile_eval.rag.debugger.common.utils import (
    DEFAULT_PROJECT_NAME,
    LASTMILE_SPAN_KIND_KEY_NAME,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class LangChainInstrumentor(BaseInstrumentor):
    """
    This is a callback handler for automatically instrumenting with
    Langchain. Here's how to use it:

    ```
    from lastmile_eval.rag.debugger.tracing.auto_instrumentation import LangChainInstrumentor
    LangChainInstrumentor().instrument()
    # Do regular LangChain calls as usual
    ```
    """

    def __init__(
        self,
        project_name: str = DEFAULT_PROJECT_NAME,
        lastmile_api_token: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._tracer = get_lastmile_tracer(
            tracer_name=project_name,
            lastmile_api_token=lastmile_api_token,
        )

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:

        wrap_function_wrapper(
            module="langchain_core.callbacks",
            name="BaseCallbackManager.__init__",
            wrapper=_BaseCallbackManagerInit(
                # TODO: Define our own LastMileTracerLangChainTracer to
                # inherit OpenInferenceTracer. Override _end_trace to save
                # span kind in our own field
                tracer=self._tracer,
                cls=_LastMileLangChainTracer,
            ),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        pass


class _LastMileLangChainTracer(OpenInferenceTracer):
    def _end_trace(self, run: Run) -> None:
        # OpenInferenceTracer ends the span and we need to do that manually
        # ourselves to avoid exporting the trace data and resetting it before
        # we can extract the span info to add a RAG event
        super(OpenInferenceTracer, self)._end_trace(run)

        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return
        with self._lock:
            event_data = self._runs.pop(run.id, None)
        if event_data:
            span = event_data.span
            try:
                _update_span(span, run)
            except Exception:
                logger.exception("Failed to update span with run data.")

            if isinstance(span, Span):
                span_kind = str(
                    "agent" if "agent" in run.name.lower() else (run.run_type)
                )
                if not _should_skip(span_kind):
                    serializable_payload: Dict[str, Any] = {}
                    for key, value in span.attributes.items():
                        serializable_payload[key] = value
                    self._tracer.add_rag_event_for_span(
                        event_name=span_kind,
                        span=span,
                        event_data=serializable_payload,
                    )
                span.set_attribute(LASTMILE_SPAN_KIND_KEY_NAME, span_kind)

            # We can't use real time because the handler may be
            # called in a background thread.
            end_time_utc_nano = (
                _as_utc_nano(run.end_time) if run.end_time else None
            )
            span.end(end_time=end_time_utc_nano)


class _BaseCallbackManagerInit:
    __slots__ = ("_tracer", "_cls")

    def __init__(
        self, tracer: trace_api.Tracer, cls: Type[_LastMileLangChainTracer]
    ):
        self._tracer = tracer
        self._cls = cls

    def __call__(
        self,
        wrapped: Callable[..., None],
        instance: BaseCallbackManager,
        args: Any,
        kwargs: Any,
    ) -> None:
        wrapped(*args, **kwargs)
        for handler in instance.inheritable_handlers:
            # Handlers may be copied when new managers are created, so we
            # don't want to keep adding. E.g. see the following location.
            # https://github.com/langchain-ai/langchain/blob/5c2538b9f7fb64afed2a918b621d9d8681c7ae32/libs/core/langchain_core/callbacks/manager.py#L1876  # noqa: E501
            if isinstance(handler, self._cls):
                break
        else:
            instance.add_handler(self._cls(tracer=self._tracer), True)


def _should_skip(event_type: str) -> bool:
    """
    The Langchain event types are:
        1. "tool"
        2. "retriever"
        3. "chat_model"
        4. "llm"
        5. "chain"
        6. "parser"
        7. "prompt"
    Source: https://github.com/search?q=repo%3Alangchain-ai%2Flangchain+run_type%3D+language%3APython&type=code
    """
    return event_type in {"chain"}
