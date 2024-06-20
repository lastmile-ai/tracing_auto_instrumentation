from .openai_wrapper import OpenAIWrapper
from .async_openai_wrapper import AsyncOpenAIWrapper
from lastmile_eval.rag.debugger.api import LastMileTracer

import openai


def wrap_openai(
    client: openai.OpenAI | openai.AsyncOpenAI,
    tracer: LastMileTracer,
) -> OpenAIWrapper | AsyncOpenAIWrapper:
    """
    Wrap an OpenAI Client to add LastMileTracer so that
    any calls to it will contain tracing data.

    Currently only v1 API is supported, which was released November 6, 2023:
        https://stackoverflow.com/questions/77435356/openai-api-new-version-v1-of-the-openai-python-package-appears-to-contain-bre
    We also only support `/v1/chat/completions` api and not `/v1/completions`

    :param client: OpenAI client created using openai.OpenAI()

    Example usage:
    ```python
    import openai
    from tracing_auto_instrumentation.openai import wrap_openai
    from lastmile_eval.rag.debugger.tracing.sdk import get_lastmile_tracer

    tracer = get_lastmile_tracer(
        tracer_name="my-tracer-name",
        lastmile_api_token="my-lastmile-api-token",
    )
    client = wrap_openai(openai.OpenAI(), tracer)
    # Use client as you would normally use the OpenAI client
    ```
    """
    if isinstance(client, openai.OpenAI):
        return OpenAIWrapper(client, tracer)
    return AsyncOpenAIWrapper(client, tracer)
