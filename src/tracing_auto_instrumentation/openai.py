from typing import TYPE_CHECKING


from lastmile_eval.rag.debugger.tracing.lastmile_tracer import LastMileTracer

from tracing_auto_instrumentation.wrap_utils import verify_package_installed

if TYPE_CHECKING:
    import openai as openai_module
    from tracing_auto_instrumentation.openai_helpers import OpenAIV0Wrapper
    from tracing_auto_instrumentation.openai_helpers import OpenAIV1Wrapper

def wrap(
    openai: openai_module.OpenAI, tracer: LastMileTracer
) -> OpenAIV0Wrapper | OpenAIV1Wrapper:
    """
    Wrap the openai module (pre v1) or OpenAI instance (post v1) to add tracing.

    :param openai: The openai module or OpenAI object
    """
    # Check deps
    deps = ["openai"]
    for dep in deps:
        verify_package_installed(dep, instrumentor_name="openai")

    # Perform Imports after verifying package installation
    from tracing_auto_instrumentation.openai_helpers import OpenAIV0Wrapper
    from tracing_auto_instrumentation.openai_helpers import OpenAIV1Wrapper

    if hasattr(openai, "chat") and hasattr(openai.chat, "completions"):
        return OpenAIV1Wrapper(openai, tracer)
    else:
        return OpenAIV0Wrapper(openai, tracer)


wrap_openai = wrap