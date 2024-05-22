import os
import dotenv
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from lastmile_eval.rag.debugger.tracing.sdk import get_lastmile_tracer
from tracing_auto_instrumentation.ibm import wrap_watson

print("Generate")

dotenv.load_dotenv()

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
    space_id=os.getenv("WATSONX_SPACE_ID"),
    verify=None,
    validate=True,
)

tracer = get_lastmile_tracer(
    "elementary-my-dear-watson", os.getenv("LASTMILE_API_TOKEN")
)
wrapper = wrap_watson(model, tracer)
response = wrapper.generate("This is the generate call")
print(response)

response = wrapper.generate_text("this is the generate text call")
print(response)
