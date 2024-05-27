from typing import TYPE_CHECKING
from lastmile_eval.rag.debugger.api import LastMileTracer

from tracing_auto_instrumentation.wrap_utils import verify_package_installed



if TYPE_CHECKING:
    # Optional Dependencies.
    from ibm_watsonx_ai.foundation_models import Model
    from tracing_auto_instrumentation.ibm_helpers import IBMWatsonXModelWrapper


def wrap_watson(
    ibm_watsonx_model: "Model", tracer: LastMileTracer
) -> "IBMWatsonXModelWrapper":
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
    # Check deps
    deps = ["ibm-watsonx-ai"]
    for dep in deps:
        verify_package_installed(deps, "ibm")

    # Perform imports after verifying package installation
    from tracing_auto_instrumentation.ibm_helpers import IBMWatsonXModelWrapper

    return IBMWatsonXModelWrapper(ibm_watsonx_model, tracer)
