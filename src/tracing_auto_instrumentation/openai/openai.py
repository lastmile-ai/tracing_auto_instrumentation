import time
from typing import Any, Mapping
import openai as openai_module

from lastmile_eval.rag.debugger.tracing.lastmile_tracer import LastMileTracer
from lastmile_eval.rag.debugger.common.query_trace_types import (
    LLMOutputReceived,
    PromptResolved,
)
from tracing_auto_instrumentation.wrap_utils import (
    NamedWrapper,
    json_serialize_anything,
)


def flatten_json(obj: Mapping[str, Any]):
    return {k: json_serialize_anything(v) for k, v in obj.items()}


def merge_dicts(d1, d2):
    return {**d1, **d2}


def postprocess_streaming_results(all_results):
    role = None
    content = None
    tool_calls = None
    finish_reason = None
    for result in all_results:
        delta = result["choices"][0]["delta"]
        if role is None and delta.get("role") is not None:
            role = delta.get("role")

        if delta.get("finish_reason") is not None:
            finish_reason = delta.get("finish_reason")

        if delta.get("content") is not None:
            content = (content or "") + delta.get("content")
        if delta.get("tool_calls") is not None:
            if tool_calls is None:
                tool_calls = [
                    {
                        "id": delta["tool_calls"][0]["id"],
                        "type": delta["tool_calls"][0]["type"],
                        "function": delta["tool_calls"][0]["function"],
                    }
                ]
            else:
                tool_calls[0]["function"]["arguments"] += delta["tool_calls"][
                    0
                ]["function"]["arguments"]

    return [
        {
            "index": 0,
            "message": {
                "role": role,
                "content": content,
                "tool_calls": tool_calls,
            },
            "logprobs": None,
            "finish_reason": finish_reason,
        }
    ]


class ChatCompletionWrapper:
    def __init__(self, create_fn, acreate_fn, tracer: LastMileTracer):
        self.create_fn = create_fn
        self.acreate_fn = acreate_fn
        self.tracer: LastMileTracer = tracer

    def create(self, *args, **kwargs):
        params = self._parse_params(kwargs)
        params_flat = flatten_json(params)

        with self.tracer.start_as_current_span("chat-completion-span") as span:
            self.tracer.mark_rag_query_trace_event(
                PromptResolved(
                    fully_resolved_prompt=json_serialize_anything(params)
                )
            )
            stream = kwargs.get("stream", False)

            start = time.time()
            raw_response = self.create_fn(*args, **kwargs)
            if stream:

                def gen():
                    first = True
                    all_results = []
                    for item in raw_response:
                        if first:
                            span.set_attribute(
                                "time_to_first_token", time.time() - start
                            )
                            first = False
                        all_results.append(
                            item if isinstance(item, dict) else item.dict()
                        )
                        yield item

                    stream_output = postprocess_streaming_results(all_results)
                    span.set_attribute(flatten_json(stream_output))

                return gen()
            else:
                log_response = (
                    raw_response
                    if isinstance(raw_response, dict)
                    else raw_response.dict()
                )
                span.set_attributes(
                    {
                        "time_to_first_token": time.time() - start,
                        "tokens": log_response["usage"]["total_tokens"],
                        "prompt_tokens": log_response["usage"][
                            "prompt_tokens"
                        ],
                        "completion_tokens": log_response["usage"][
                            "completion_tokens"
                        ],
                        "choices": json_serialize_anything(
                            log_response["choices"]
                        ),
                        **params_flat,
                    }
                )
                try:
                    output = log_response["choices"][0]["message"]["content"]
                    self.tracer.mark_rag_query_trace_event(
                        LLMOutputReceived(llm_output=output)
                    )
                except Exception as e:
                    # TODO log this
                    pass
                return raw_response

    async def acreate(self, *args, **kwargs):
        params = self._parse_params(kwargs)
        stream = kwargs.get("stream", False)

        with self.tracer.start_as_current_span("chat-completion") as span:
            self.tracer.mark_rag_query_trace_event(
                PromptResolved(
                    fully_resolved_prompt=json_serialize_anything(params)
                )
            )

            span.set_attributes(flatten_json(params))
            start = time.time()
            raw_response = await self.acreate_fn(*args, **kwargs)
            if stream:

                async def gen():
                    first = True
                    all_results = []
                    async for item in raw_response:
                        if first:
                            span.set_attributes(
                                {
                                    "time_to_first_token": time.time() - start,
                                }
                            )
                            first = False
                        all_results.append(
                            item if isinstance(item, dict) else item.dict()
                        )
                        yield item

                    stream_output = postprocess_streaming_results(all_results)
                    span.set_attributes(flatten_json(stream_output))

                return gen()
            else:
                log_response = (
                    raw_response
                    if isinstance(raw_response, dict)
                    else raw_response.dict()
                )
                span.set_attributes(
                    {
                        "tokens": log_response["usage"]["total_tokens"],
                        "prompt_tokens": log_response["usage"][
                            "prompt_tokens"
                        ],
                        "completion_tokens": log_response["usage"][
                            "completion_tokens"
                        ],
                        "choices": json_serialize_anything(
                            log_response["choices"]
                        ),
                    }
                )
                try:
                    output = log_response["choices"][0]["message"]["content"]
                    self.tracer.mark_rag_query_trace_event(
                        LLMOutputReceived(llm_output=output)
                    )
                except Exception as e:
                    # TODO log this
                    pass
                return raw_response

    @classmethod
    def _parse_params(cls, params):
        # First, destructively remove span_info
        ret = params.pop("span_info", {})

        # Then, copy the rest of the params
        params = {**params}
        messages = params.pop("messages", None)
        return merge_dicts(
            ret,
            {
                "input": messages,
                "metadata": params,
            },
        )


class EmbeddingWrapper:
    def __init__(self, create_fn, acreate_fn, tracer):
        self.create_fn = create_fn
        self.acreate_fn = acreate_fn
        self.tracer = tracer

    def create(self, *args, **kwargs):
        params = self._parse_params(kwargs)
        params_flat = flatten_json(params)

        with self.tracer.start_as_current_span("embedding") as span:
            raw_response = self.create_fn(*args, **kwargs)
            log_response = (
                raw_response
                if isinstance(raw_response, dict)
                else raw_response.dict()
            )
            span.set_attributes(
                {
                    "tokens": log_response["usage"]["total_tokens"],
                    "prompt_tokens": log_response["usage"]["prompt_tokens"],
                    "embedding_length": len(
                        log_response["data"][0]["embedding"]
                    ),
                    **flatten_json(params),
                },
            )
            return raw_response

    async def acreate(self, *args, **kwargs):
        params = self._parse_params(kwargs)

        with self.tracer.start_as_current_span("embedding") as span:
            raw_response = await self.acreate_fn(*args, **kwargs)
            log_response = (
                raw_response
                if isinstance(raw_response, dict)
                else raw_response.dict()
            )
            span.set_attributes(
                {
                    "tokens": log_response["usage"]["total_tokens"],
                    "prompt_tokens": log_response["usage"]["prompt_tokens"],
                    "embedding_length": len(
                        log_response["data"][0]["embedding"]
                    ),
                    **flatten_json(params),
                },
            )
            return raw_response

    @classmethod
    def _parse_params(cls, params):
        # First, destructively remove span_info
        ret = params.pop("span_info", {})

        params = {**params}
        input = params.pop("input", None)

        return merge_dicts(
            ret,
            {
                "input": input,
                "metadata": params,
            },
        )


class ChatCompletionV0Wrapper(NamedWrapper):
    def __init__(self, chat, tracer):
        self.__chat = chat
        self.tracer = tracer
        super().__init__(chat)

    def create(self, *args, **kwargs):
        return ChatCompletionWrapper(
            self.__chat.create, self.__chat.acreate, self.tracer
        ).create(*args, **kwargs)

    async def acreate(self, *args, **kwargs):
        return await ChatCompletionWrapper(
            self.__chat.create, self.__chat.acreate, self.tracer
        ).acreate(*args, **kwargs)


class EmbeddingV0Wrapper(NamedWrapper):
    def __init__(self, embedding, tracer):
        self.__embedding = embedding
        self.tracer = tracer
        super().__init__(embedding)

    def create(self, *args, **kwargs):
        return EmbeddingWrapper(
            self.__embedding.create, self.__embedding.acreate, self.tracer
        ).create(*args, **kwargs)

    async def acreate(self, *args, **kwargs):
        return await ChatCompletionWrapper(
            self.__embedding.create, self.__embedding.acreate, self.tracer
        ).acreate(*args, **kwargs)


# This wraps 0.*.* versions of the openai module, eg https://github.com/openai/openai-python/tree/v0.28.1
class OpenAIV0Wrapper(NamedWrapper):
    def __init__(self, openai, tracer):
        super().__init__(openai)
        self.tracer = tracer
        self.ChatCompletion = ChatCompletionV0Wrapper(
            openai.ChatCompletion, tracer
        )
        self.Embedding = EmbeddingV0Wrapper(openai.Embedding, tracer)


class CompletionsV1Wrapper(NamedWrapper):
    def __init__(self, completions, tracer):
        self.__completions = completions
        self.tracer = tracer
        super().__init__(completions)

    def create(self, *args, **kwargs):
        return ChatCompletionWrapper(
            self.__completions.create, None, self.tracer
        ).create(*args, **kwargs)


class EmbeddingV1Wrapper(NamedWrapper):
    def __init__(self, embedding, tracer):
        self.__embedding = embedding
        self.tracer = tracer
        super().__init__(embedding)

    def create(self, *args, **kwargs):
        return EmbeddingWrapper(
            self.__embedding.create, None, self.tracer
        ).create(*args, **kwargs)


class AsyncCompletionsV1Wrapper(NamedWrapper):
    def __init__(self, completions, tracer):
        self.__completions = completions
        self.tracer = tracer
        super().__init__(completions)

    async def create(self, *args, **kwargs):
        return await ChatCompletionWrapper(
            None, self.__completions.create, self.tracer
        ).acreate(*args, **kwargs)


class AsyncEmbeddingV1Wrapper(NamedWrapper):
    def __init__(self, embedding, tracer):
        self.__embedding = embedding
        self.tracer = tracer
        super().__init__(embedding)

    async def create(self, *args, **kwargs):
        return await EmbeddingWrapper(
            None, self.__embedding.create, self.tracer
        ).acreate(*args, **kwargs)


class ChatV1Wrapper(NamedWrapper):
    def __init__(self, chat, tracer):
        super().__init__(chat)
        self.tracer = tracer

        import openai

        if (
            type(chat.completions)
            == openai.resources.chat.completions.AsyncCompletions
        ):
            self.completions = AsyncCompletionsV1Wrapper(
                chat.completions, self.tracer
            )
        else:
            self.completions = CompletionsV1Wrapper(
                chat.completions, self.tracer
            )


# This wraps 1.*.* versions of the openai module, eg https://github.com/openai/openai-python/tree/v1.1.0
class OpenAIV1Wrapper(NamedWrapper):
    def __init__(self, openai, tracer):
        super().__init__(openai)
        self.tracer = tracer

        self.chat = ChatV1Wrapper(openai.chat, self.tracer)

        if (
            type(openai.embeddings)
            == openai_module.resources.embeddings.AsyncEmbeddings
        ):
            self.embeddings = AsyncEmbeddingV1Wrapper(
                openai.embeddings, self.tracer
            )
        else:
            self.embeddings = EmbeddingV1Wrapper(
                openai.embeddings, self.tracer
            )


def wrap(
    openai: openai_module.OpenAI, tracer: LastMileTracer
) -> OpenAIV0Wrapper | OpenAIV1Wrapper:
    """
    Wrap the openai module (pre v1) or OpenAI instance (post v1) to add tracing.

    :param openai: The openai module or OpenAI object
    """
    if hasattr(openai, "chat") and hasattr(openai.chat, "completions"):
        return OpenAIV1Wrapper(openai, tracer)
    else:
        return OpenAIV0Wrapper(openai, tracer)


wrap_openai = wrap