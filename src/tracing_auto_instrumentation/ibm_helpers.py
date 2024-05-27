import logging
import inspect
import json

from abc import abstractmethod
from typing import Any, Protocol

from lastmile_eval.rag.debugger.api import LastMileTracer
from lastmile_eval.rag.debugger.tracing.decorators import (
    _get_serializable_input,
    _try_log_output,
)

from lastmile_eval.rag.debugger.common.query_trace_types import (
    LLMOutputReceived,
    PromptResolved,
)

from ibm_watsonx_ai.foundation_models import Model

from .wrap_utils import (
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


# logger.info("Generate text:")

# logger.info(model.generate_text("the quick brown fox"))


# logger.info("Generate text stream")

# for t in model.generate_text_stream("the quick brown fox"):
#     logger.info(t, end="")

# logger.info("Details:")
# logger.info(model.get_details())


class GenerateWrapper:
    def __init__(
        self, generate: IBMWatsonXGenerateMethod, tracer: LastMileTracer
    ):
        self.generate_fn = generate
        self.tracer = tracer

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        f_sig = inspect.signature(self.generate_fn)
        with self.tracer.start_as_current_span("text-generate-span") as span:
            input_serializable = _get_serializable_input(f_sig, args, kwargs)
            span.set_attribute("input", json.dumps(input_serializable))
            response = self.generate_fn(  # will proably be dict
                *args, **kwargs
            )
            _try_log_output(span, response)

            self.tracer.add_rag_event_for_span(
                "generate",
                span=span,
                input=input_serializable,
                output=response,
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
        f_sig = inspect.signature(self.generate_text_fn)
        with self.tracer.start_as_current_span("text-generate-span") as span:
            input_serializable = _get_serializable_input(f_sig, args, kwargs)
            span.set_attribute("input", json.dumps(input_serializable))
            response = self.generate_text_fn(  # will proably be str
                *args, **kwargs
            )
            _try_log_output(span, response)

            self.tracer.add_rag_event_for_span(
                "generate",
                span=span,
                input=input_serializable,
                output=response,
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
        res = self.generate_fn(*args, **kwargs)

        prompt: str = kwargs["prompt"] if "prompt" in kwargs else args[0]

        logger.info(f"invoking tracer to mark query event: {prompt=}")
        tracer_res = self.tracer.mark_rag_query_trace_event(
            PromptResolved(fully_resolved_prompt=prompt),
        )
        logger.info(f"did call `mark_rag_query_trace_event`: {tracer_res=}")

        llm_result: dict = res["results"][0]
        llm_output: dict = llm_result["generated_text"]

        logger.info(f"invoking tracer to mark query event: {llm_output=}")
        tracer_res = self.tracer.mark_rag_query_trace_event(
            LLMOutputReceived(llm_output=llm_output)
        )
        logger.info(f"did call `mark_rag_query_trace_event`: {tracer_res=}")

        trace_params: dict = {
            "model_id": res["model_id"],
            "model_version": res["model_version"],
            "llm_output": llm_result["generated_text"],
            "generated_tokens": llm_result["generated_text"],
            "generated_token_count": llm_result["generated_token_count"],
            "input_token_count": llm_result["input_token_count"],
        }

        logger.info(f"about to call `register_params`: {trace_params=}")
        self.tracer.register_params(trace_params)
        logger.info("did call `register_params`")

        return res

    def generate_text(self, *args: Any, **kwargs: Any) -> Any:
        res = self.generate_text_fn(*args, **kwargs)

        prompt: str = kwargs["prompt"] if "prompt" in kwargs else args[0]

        logger.info(f"invoking tracer to mark query event: {prompt=}")
        tracer_res = self.tracer.mark_rag_query_trace_event(
            PromptResolved(fully_resolved_prompt=prompt),
        )
        logger.info(f"did call `mark_rag_query_trace_event`: {tracer_res=}")

        llm_output: str = res.replace("\r", "").replace("\n", "")

        logger.info(f"invoking tracer to mark query event: {llm_output=}")
        tracer_res = self.tracer.mark_rag_query_trace_event(
            LLMOutputReceived(llm_output=llm_output)
        )

        logger.info(f"did call `mark_rag_query_trace_event`: {tracer_res=}")
        return res
