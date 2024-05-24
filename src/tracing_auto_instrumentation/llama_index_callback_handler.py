import logging
from time import time_ns
from typing import Any, Dict, Optional

from llama_index.core.callbacks import CBEventType, EventPayload
from openinference.instrumentation.llama_index._callback import (
    OpenInferenceTraceCallbackHandler,
    payload_to_semantic_attributes,
    _is_streaming_response,  # type: ignore
    _flatten,  # type: ignore
    _ResponseGen,  # type: ignore
    _EventData,  # type: ignore
    # Opinionated params we explicit want to save, see source for full list
    DOCUMENT_SCORE,
    EMBEDDING_MODEL_NAME,
    INPUT_VALUE,
    LLM_INVOCATION_PARAMETERS,
    LLM_MODEL_NAME,
    LLM_PROMPT_TEMPLATE,
    MESSAGE_FUNCTION_CALL_NAME,
    OUTPUT_VALUE,
    RERANKER_MODEL_NAME,
    RERANKER_OUTPUT_DOCUMENTS,
    RERANKER_QUERY,
    RERANKER_TOP_K,
    RETRIEVAL_DOCUMENTS,
    TOOL_CALL_FUNCTION_NAME,
    TOOL_NAME,
    # # Explicit chose not to do these two because context can be huge and we
    # # can extract this from both the prompt template template and variables
    # LLM_INPUT_MESSAGES,
    # LLM_OUTPUT_MESSAGES,
)
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry import trace as trace_api
from opentelemetry import context as context_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY  # type: ignore
from lastmile_eval.rag.debugger.tracing import get_lastmile_tracer
from lastmile_eval.rag.debugger.common.utils import (
    DEFAULT_PROJECT_NAME,
    LASTMILE_SPAN_KIND_KEY_NAME,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

PARAM_SET_SUBSTRING_MATCHES = (
    DOCUMENT_SCORE,
    EMBEDDING_MODEL_NAME,
    INPUT_VALUE,
    LLM_INVOCATION_PARAMETERS,
    LLM_MODEL_NAME,
    LLM_PROMPT_TEMPLATE,
    MESSAGE_FUNCTION_CALL_NAME,
    OUTPUT_VALUE,
    RERANKER_MODEL_NAME,
    RERANKER_OUTPUT_DOCUMENTS,
    RERANKER_QUERY,
    RERANKER_TOP_K,
    RETRIEVAL_DOCUMENTS,
    TOOL_CALL_FUNCTION_NAME,
    TOOL_NAME,
)


class LlamaIndexCallbackHandler(OpenInferenceTraceCallbackHandler):
    """
    This is a callback handler for automatically instrumenting with
    LLamaIndex. Here's how to use it:

    ```
    from lastmile_eval.rag.debugger.tracing import LlamaIndexCallbackHandler
    llama_index.core.global_handler = LlamaIndexCallbackHandler()
    # Do regular LlamaIndex calls as usual
    ```
    """

    def __init__(
        self,
        project_name: str = DEFAULT_PROJECT_NAME,
        lastmile_api_token: Optional[str] = None,
    ):
        tracer = get_lastmile_tracer(
            tracer_name=project_name,
            lastmile_api_token=lastmile_api_token,
        )
        super().__init__(tracer)

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return

        with self._lock:
            if event_type is CBEventType.TEMPLATING:
                if (
                    parent_id := self._templating_parent_id.pop(event_id, None)
                ) and payload:
                    if parent_id in self._templating_payloads:
                        self._templating_payloads[parent_id].append(payload)
                    else:
                        self._templating_payloads[parent_id] = [payload]
                return
            if not (event_data := self._event_data.pop(event_id, None)):
                return

        event_data.end_time = time_ns()
        is_dispatched = False

        if payload is not None:
            event_data.payloads.append(payload.copy())
            if isinstance(
                (exception := payload.get(EventPayload.EXCEPTION)), Exception
            ):
                event_data.exceptions.append(exception)
            try:
                event_data.attributes.update(
                    payload_to_semantic_attributes(
                        event_type, payload, is_event_end=True
                    ),
                )
            except Exception:
                logger.exception(
                    f"Failed to convert payload to semantic attributes. "
                    f"event_type={event_type}, payload={payload}",
                )
            if (
                _is_streaming_response(
                    response := payload.get(EventPayload.RESPONSE)
                )
                and response.response_gen is not None
            ):
                response.response_gen = _ResponseGen(
                    response.response_gen, event_data
                )
                is_dispatched = True

        if not is_dispatched:
            _finish_tracing(
                event_data,
                tracer=self._tracer,
                event_type=event_type,
            )
        return


def _finish_tracing(
    event_data: _EventData,
    tracer,
    event_type: CBEventType,
) -> None:
    if not (span := event_data.span):
        return
    attributes = event_data.attributes
    if event_data.exceptions:
        status_descriptions: list[str] = []
        for exception in event_data.exceptions:
            span.record_exception(exception)
            # Follow the format in OTEL SDK for description, see:
            # https://github.com/open-telemetry/opentelemetry-python/blob/2b9dcfc5d853d1c10176937a6bcaade54cda1a31/opentelemetry-api/src/opentelemetry/trace/__init__.py#L588  # noqa E501
            status_descriptions.append(
                f"{type(exception).__name__}: {exception}"
            )
        status = trace_api.Status(
            status_code=trace_api.StatusCode.ERROR,
            description="\n".join(status_descriptions),
        )
    else:
        status = trace_api.Status(status_code=trace_api.StatusCode.OK)
    span.set_status(status=status)
    try:
        span.set_attributes(dict(_flatten(attributes)))

        # Remove "serialized" from the spans
        # We need to do this because this info stores the API keys and we
        # want to remove. Other pertinent data is stored in the span attributes
        # like model name and invocation params already
        if (
            isinstance(span, ReadableSpan)
            and span._attributes is not None
            and "serialized" in span._attributes
        ):
            del span._attributes["serialized"]

        if not _should_skip(event_type):
            serializable_payload: Dict[str, Any] = {}
            for key, value in span._attributes.items():
                # Only save the opinionated data to event data and param set
                if _save_to_param_set(key):
                    serializable_payload[key] = value
            tracer.add_rag_event_for_span(
                event_name=str(event_data.event_type),
                span=span,
                event_data=serializable_payload,
            )
            tracer.register_params(
                params=serializable_payload,
                should_also_save_in_span=False,
                span=span,
            )

        # Save span kind into span attribute, but don't add it to trace-level
        # params (since that should be for trace-level data) or rag span event
        # (since it's already used for the event name there)
        span.set_attribute(LASTMILE_SPAN_KIND_KEY_NAME, event_data.event_type)

    except Exception:
        logger.exception(
            f"Failed to set attributes on span. event_type={event_data.event_type}, "
            f"attributes={attributes}",
        )
    span.end(end_time=event_data.end_time)


def _should_skip(event_type: CBEventType) -> bool:
    # TODO: Add some actual crieria for skipping to log span events
    return event_type in {
        CBEventType.CHUNKING,
        CBEventType.NODE_PARSING,
        CBEventType.TREE,
        CBEventType.SYNTHESIZE,
    }


def _save_to_param_set(key: str):
    for substring in PARAM_SET_SUBSTRING_MATCHES:
        if substring in key:
            return True
    return False
