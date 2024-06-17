import logging
import inspect

from abc import abstractmethod
from typing import Any, Protocol

from ibm_watsonx_ai.foundation_models import Model

from lastmile_eval.rag.debugger.api import LastMileTracer
from lastmile_eval.rag.debugger.tracing.decorators import (
    _try_log_input,
    _try_log_output,
)


from tracing_auto_instrumentation.utils import (
    NamedWrapper,
)

logger = logging.getLogger(__name__)


# TODO: type these correctly
class IBMWatsonXGenerateMethod(Protocol):
    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        pass


class IBMWatsonXGenerateTextMethod(Protocol):
    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        pass


class IBMWatsonXGenerateTextStreamMethod(Protocol):
    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        pass


class GenerateWrapper:
    def __init__(
        self, generate: IBMWatsonXGenerateMethod, tracer: LastMileTracer
    ):
        self.generate_fn = generate
        self.tracer = tracer

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        prompt = kwargs["prompt"] if "prompt" in kwargs else args[0]
        f_sig = inspect.signature(self.generate_fn)
        with self.tracer.start_as_current_span("text-generate-span") as span:
            _try_log_input(span, f_sig, args, kwargs)
            response = self.generate_fn(  # probably dict, the Watson generate_text method returns Any
                *args, **kwargs
            )
            _try_log_output(span, response)

            llm_result: dict[str, Any] = response["results"][0]
            llm_output: str = llm_result["generated_text"]
            self.tracer.add_query_event(
                query=prompt,
                llm_output=llm_output,
                span=span,
                # already saved into span attributes with `_try_log_input` and `_try_log_output`
                should_also_save_in_span=False,
            )

            trace_params: dict[str, Any] = {
                "model_id": response["model_id"],
                "model_version": response["model_version"],
                "llm_output": llm_result["generated_text"],
                # "generated_tokens": llm_result["generated_tokens"],
                "generated_token_count": llm_result["generated_token_count"],
                "input_token_count": llm_result["input_token_count"],
            }
            self.tracer.register_params(trace_params)

            self.tracer.add_rag_event_for_trace(
                event_name="generate",
                input=prompt,
                output=llm_output,
                event_data=trace_params,
            )
            return response


class GenerateTextWrapper:
    def __init__(
        self,
        generate_text: IBMWatsonXGenerateTextMethod,
        tracer: LastMileTracer,
    ):
        self.generate_text_fn = generate_text
        self.tracer = tracer

    def generate_text(self, *args: Any, **kwargs: Any) -> Any:
        prompt = kwargs["prompt"] if "prompt" in kwargs else args[0]
        f_sig = inspect.signature(self.generate_text_fn)
        with self.tracer.start_as_current_span("text-generate-span") as span:
            _try_log_input(span, f_sig, args, kwargs)
            response = self.generate_text_fn(  # probably str, the Watson generate_text method returns Any
                *args, **kwargs
            )
            _try_log_output(span, response)

            llm_output: str = str(response).replace("\r", "").replace("\n", "")
            self.tracer.add_query_event(
                query=prompt,
                llm_output=llm_output,
                span=span,
                # already saved into span attributes with `_try_log_input` and `_try_log_output`
                should_also_save_in_span=False,
            )

            trace_params: dict[str, Any] = {
                "model_id": response["model_id"],
                "model_version": response["model_version"],
            }
            self.tracer.register_params(trace_params)

            self.tracer.add_rag_event_for_trace(
                event_name="generate_text",
                input=prompt,
                output=llm_output,
                event_data=trace_params,
            )
            return response


class IBMWatsonXModelWrapper(NamedWrapper[Model]):
    def __init__(self, ibm_watsonx_model: Model, tracer: LastMileTracer):
        super().__init__(ibm_watsonx_model)
        self.ibm_watsonx_model = ibm_watsonx_model
        self.tracer = tracer

        self.generate_fn = GenerateWrapper(
            ibm_watsonx_model.generate, tracer  # type: ignore
        ).generate

        self.generate_text_fn = GenerateTextWrapper(
            ibm_watsonx_model.generate_text, tracer  # type: ignore
        ).generate_text

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        return self.generate_fn(*args, **kwargs)

    def generate_text(self, *args: Any, **kwargs: Any) -> Any:
        return self.generate_text_fn(*args, **kwargs)


def wrap_watson(
    ibm_watsonx_model: Model, tracer: LastMileTracer
) -> IBMWatsonXModelWrapper:
    """
    Wrapper method around Watson's Model class which adds LastMile tracing to
    the methods `generate`, `generate_text`, and `generate_text_stream`.

    To use it, wrap it around an existing Model and tracer object like so:

    ```python
    from ibm_watsonx_ai.foundation_models import Model
    from ibm_watsonx_ai.metanames import (
        GenTextParamsMetaNames as GenParams,
    )
    from ibm_watsonx_ai.foundation_models.utils.enums import (
        ModelTypes,
    )
    from lastmile_eval.rag.debugger.tracing.auto_instrumentation import (
        wrap_watson,
    )
    from lastmile_eval.rag.debugger.tracing.sdk import get_lastmile_tracer

    tracer = get_lastmile_tracer(<tracer-name>, <lastmile-api-token>)
    model = Model(
        model_id=ModelTypes.GRANITE_13B_CHAT_V2,
        params=generate_params,
        credentials=dict(
            api_key=os.getenv("WATSONX_API_KEY"),
            url="https://us-south.ml.cloud.ibm.com",
        ),
        space_id=os.getenv("WATSONX_SPACE_ID"),
        verify=None,
        validate=True,
    )
    wrapped_model = wrap_watson(tracer, model)
    ```

    """
    return IBMWatsonXModelWrapper(ibm_watsonx_model, tracer)
