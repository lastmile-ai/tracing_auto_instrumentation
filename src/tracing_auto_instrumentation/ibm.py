from abc import abstractmethod
import os
from typing import Any, Protocol
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import (
    ModelTypes,
)

from lastmile_eval.rag.debugger.tracing.lastmile_tracer import LastMileTracer

from lastmile_eval.rag.debugger.common.query_trace_types import LLMOutputReceived, PromptResolved
from lastmile_eval.rag.debugger.tracing.wrap_utils import (
    NamedWrapper,
    json_serialize_anything,
)

# NamedWrapper = "1"
# json_serialize_anything = "2"


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


# To display example params enter
GenParams().get_example_values()

generate_params = {GenParams.MAX_NEW_TOKENS: 25}

model = Model(
    model_id=ModelTypes.GRANITE_13B_CHAT_V2,
    params=generate_params,
    credentials=dict(
        api_key=os.getenv("WATSONX_API_KEY"),
        url="https://us-south.ml.cloud.ibm.com",
    ),
    verify=None,
    validate=True,
)

# print("Generate text:")

# print(model.generate_text("the quick brown fox"))

print("Generate")
print(model.generate("the quick brown fox"))

# print("Generate text stream")

# for t in model.generate_text_stream("the quick brown fox"):
#     print(t, end="")

# print("Details:")
# print(model.get_details())


class GenerateWrapper:
    def __init__(
        self, generate: IBMWatsonXGenerateMethod, tracer: LastMileTracer
    ):
        self.generate_fn = generate
        self.tracer = tracer

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        with self.tracer.start_as_current_span("text-generate-span") as span:
            self.tracer.mark_rag_query_trace_event(
                PromptResolved(
                    fully_resolved_prompt=json_serialize_anything(
                        {"args": args, "kwargs": kwargs}
                    )
                )
            )
            raw_response = self.generate_fn(*args, **kwargs)
            self.tracer.mark_rag_query_trace_event(
                event=LLMOutputReceived(llm_output="xyz")
            )
            return raw_response


class IBMWatsonXModelWrapper(NamedWrapper[Model]):
    def __init__(self, ibm_watsonx_model: Model, tracer: LastMileTracer):
        self.ibm_watsonx_model = ibm_watsonx_model
        self.tracer = tracer

        self.generate = GenerateWrapper(ibm_watsonx_model.generate, tracer)  # type: ignore
        # self.generate_text = GenerateTextWrapper(
        #     ibm_watsonx_model.generate_text, tracer
        # )
        # self.generate_text_stream = GenerateTextStreamWrapper(
        #     ibm_watsonx_model.generate_text_stream, tracer
        # )


def wrap(ibm_watsonx_model: Model, tracer: LastMileTracer):
    return IBMWatsonXModelWrapper(ibm_watsonx_model, tracer)
