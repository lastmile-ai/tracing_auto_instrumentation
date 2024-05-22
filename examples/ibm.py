import argparse
import dotenv
import logging
import os
import sys

from enum import Enum

from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

from lastmile_eval.rag.debugger.api import LastMileTracer
from lastmile_eval.rag.debugger.tracing.sdk import get_lastmile_tracer

from tracing_auto_instrumentation.ibm import wrap_watson

logger = logging.getLogger(__name__)


class Mode(Enum):
    GENERATE = "GENERATE"
    GENERATE_TEXT = "GENERATE_TEXT"


def init_watson_model() -> Model:
    # To display example params enter
    GenParams().get_example_values()
    generate_params = {GenParams.MAX_NEW_TOKENS: 25}

    watson_model = Model(
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

    return watson_model


def run_generate(prompt: str, trace_name: str) -> None:
    tracer: LastMileTracer = get_lastmile_tracer(
        trace_name, os.getenv("LASTMILE_API_TOKEN")
    )

    watson_model: Model = init_watson_model()

    tracer.log("start lastmile wrap...")
    wrapper = wrap_watson(watson_model, tracer)
    tracer.log("lastmile wrap complete.")

    with tracer.start_as_current_span(trace_name) as span:
        tracer.log("start watsonx generate...")
        response = wrapper.generate(prompt)
        tracer.log(f"watsonx generate: {response=}")


def run_generate_text(prompt: str, trace_name: str) -> None:
    tracer: LastMileTracer = get_lastmile_tracer(
        trace_name, os.getenv("LASTMILE_API_TOKEN")
    )

    watson_model: Model = init_watson_model()

    tracer.log("start lastmile wrap...")
    wrapper = wrap_watson(watson_model, tracer)
    tracer.log("lastmile wrap complete.")

    with tracer.start_as_current_span(trace_name) as span:
        tracer.log("start watsonx generate_text...")
        response = wrapper.generate_text(prompt)
        tracer.log(f"watsonx generate_text: {response=}")


def main():
    logger.info("IBM WatsonX Generation Script Starting...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--mode", type=Mode, choices=list(Mode), required=True)
    parser.add_argument(
        "--trace-name", type=str, default="elementary-my-dear-watson"
    )
    args = parser.parse_args()

    # n.b. required for multiple api keys, make sure `.env` is in your local path
    dotenv.load_dotenv()

    mode: Mode = args.mode
    if Mode.GENERATE == mode:
        logger.info(f"running with mode: {mode}")
        run_generate(args.prompt, args.trace_name)
    elif Mode.GENERATE_TEXT == mode:
        logger.info(f"running with mode: {mode}")
        run_generate_text(args.prompt, args.trace_name)
    else:
        logger.error(f"unsupported mode: {mode}")
        return -1

    logger.info("IBM WatsonX Generation Script Complete.")
    return 0


if "__main__" == __name__:
    sys.exit(main())
