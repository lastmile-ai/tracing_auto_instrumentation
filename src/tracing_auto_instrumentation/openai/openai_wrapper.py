import json
import time
from typing import Any, Mapping, Optional

import openai as openai_module
from lastmile_eval.rag.debugger.api import LastMileTracer
from openai.resources.chat import AsyncChat, AsyncCompletions, Chat
from openai.resources.embeddings import AsyncEmbeddings
from openai.types.chat import ChatCompletionChunk

from ..utils import (
    NamedWrapper,
    json_serialize_anything,
)

# pylint: disable=missing-function-docstring


def flatten_json(obj: Mapping[str, Any]):
    return {k: json_serialize_anything(v) for k, v in obj.items()}


def merge_dicts(d1, d2):
    return {**d1, **d2}


def postprocess_streaming_results(all_results: list[Any]) -> Mapping[str, Any]:
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

    return {
        "index": 0,  # TODO: Can be multiple if n > 1
        "message": {
            "role": role,
            "content": content,
            "tool_calls": tool_calls,
        },
        "logprobs": None,
        "finish_reason": finish_reason,
    }


class ChatCompletionWrapper:
    def __init__(
        self,
        create_fn,
        acreate_fn,
        tracer: LastMileTracer,
    ):
        self.create_fn = create_fn
        self.acreate_fn = acreate_fn
        self.tracer: LastMileTracer = tracer

    def create(self, *args, **kwargs):
        params = self._parse_params(kwargs)
        params_flat = flatten_json(params)
        stream = kwargs.get("stream", False)

        rag_event_input = json_serialize_anything(params)
        with self.tracer.start_as_current_span("chat-completion-span") as span:
            start = time.time()
            raw_response = self.create_fn(*args, **kwargs)
            if stream:

                def gen():
                    first = True
                    accumulated_text = None
                    for item in raw_response:
                        if first:
                            span.set_attribute(
                                "time_to_first_token", time.time() - start
                            )
                            first = False
                        if isinstance(item, ChatCompletionChunk):
                            # Ignore multiple responses for now,
                            # will support in future PR, by looking at the index in choice dict
                            # We need to also support tool call handling (as well as tool
                            # call handling streaming, which we never did properly):
                            # https://community.openai.com/t/has-anyone-managed-to-get-a-tool-call-working-when-stream-true/498867
                            choice = item.choices[
                                0
                            ]  # TODO: Can be multiple if n > 1
                            if (
                                choice
                                and choice.delta
                                and (choice.delta.content is not None)
                            ):
                                accumulated_text = (
                                    accumulated_text or ""
                                ) + choice.delta.content
                        yield item

                    if accumulated_text is not None:
                        # TODO: Save all the data inside of the span instead of
                        # just the output text content
                        span.set_attribute("output_content", accumulated_text)
                    _add_rag_event_with_output(
                        self.tracer,
                        "chat_completion_create_stream",
                        span,
                        input=rag_event_input,
                        output=accumulated_text,
                        event_data=json.loads(rag_event_input),
                    )

                yield from gen()

            # Non-streaming part
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
                    # TODO: Handle responses where n > 1
                    output = log_response["choices"][0]["message"]["content"]
                    _add_rag_event_with_output(
                        self.tracer,
                        "chat_completion_create",
                        span,
                        input=rag_event_input,
                        output=output,
                        event_data=json.loads(rag_event_input),
                    )
                except Exception as e:
                    # TODO log this
                    pass

                # Python is a clown language and does not allow you to both
                # yield and return in the same function (return just exits
                # the generator early), and there's no explicit way of
                # making this obviously because Python isn't statically typed
                #
                # We have to yield instead of returning a generator for
                # streaming because if we return a generator, the generator
                # does not actually compute or execute the values until later
                # and at which point the span has already closed. Besides
                # getting an an error saying that there's no defined trace id
                # to log the rag events, this is just not good to do because
                # what's the point of having a span to trace data if it ends
                # prematurely before we've actually computed any values?
                # Therefore we must yield to ensure the span remains open for
                # streaming events.
                #
                # For non-streaming, this means that we're still yielding the
                # "returned" response and we just process it at callsites
                # using return next(response)
                yield raw_response

    async def acreate(self, *args, **kwargs):
        params = self._parse_params(kwargs)
        stream = kwargs.get("stream", False)

        rag_event_input = json_serialize_anything(params)
        with self.tracer.start_as_current_span("chat-completion") as span:
            span.set_attributes(flatten_json(params))
            start = time.time()
            raw_response = await self.acreate_fn(*args, **kwargs)
            if stream:

                async def gen():
                    first = True
                    accumulated_text = None
                    async for item in raw_response:
                        if first:
                            span.set_attribute(
                                "time_to_first_token", time.time() - start
                            )
                            first = False
                        if isinstance(item, ChatCompletionChunk):
                            # Ignore multiple responses for now,
                            # will support in future PR, by looking at the index in choice dict
                            # We need to also support tool call handling (as well as tool
                            # call handling streaming, which we never did properly):
                            # https://community.openai.com/t/has-anyone-managed-to-get-a-tool-call-working-when-stream-true/498867
                            choice = item.choices[
                                0
                            ]  # TODO: Can be multiple if n > 1
                            if (
                                choice
                                and choice.delta
                                and (choice.delta.content is not None)
                            ):
                                accumulated_text = (
                                    accumulated_text or ""
                                ) + choice.delta.content
                        yield item

                    if accumulated_text is not None:
                        # TODO: Save all the data inside of the span instead of
                        # just the output text content
                        # stream_output = postprocess_streaming_results(all_results)
                        # span.set_attributes(flatten_json(stream_output))
                        span.set_attribute("output_content", accumulated_text)
                    _add_rag_event_with_output(
                        self.tracer,
                        "chat_completion_acreate_stream",
                        span,
                        input=rag_event_input,
                        output=accumulated_text,
                        event_data=json.loads(rag_event_input),
                    )

                async for chunk in gen():
                    yield chunk

            # Non-streaming part
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
                    _add_rag_event_with_output(
                        self.tracer,
                        "chat_completion_acreate",
                        span,
                        input=rag_event_input,
                        output=output,  # type: ignore
                        event_data=json.loads(rag_event_input),
                    )
                except Exception as e:
                    # TODO log this
                    pass

                # Python is a clown language and does not allow you to both
                # yield and return in the same function (return just exits
                # the generator early), and there's no explicit way of
                # making this obviously because Python isn't statically typed
                #
                # We have to yield instead of returning a generator for
                # streaming because if we return a generator, the generator
                # does not actually compute or execute the values until later
                # and at which point the span has already closed. Besides
                # getting an an error saying that there's no defined trace id
                # to log the rag events, this is just not good to do because
                # what's the point of having a span to trace data if it ends
                # prematurely before we've actually computed any values?
                # Therefore we must yield to ensure the span remains open for
                # streaming events.
                #
                # For non-streaming, this means that we're still yielding the
                # "returned" response and we just process it at callsites
                # using return next(response)
                yield raw_response

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
        response = ChatCompletionWrapper(
            self.__chat.create, self.__chat.acreate, self.tracer
        ).create(*args, **kwargs)

        stream = kwargs.get("stream", False)
        if not stream:
            non_streaming_response_value = next(response)
            return non_streaming_response_value
        return response

    async def acreate(self, *args, **kwargs):
        response = ChatCompletionWrapper(
            self.__chat.create, self.__chat.acreate, self.tracer
        ).acreate(*args, **kwargs)
        stream = kwargs.get("stream", False)

        if not stream:
            non_streaming_response_value = await anext(response)
            return non_streaming_response_value
        return response


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
        response = await ChatCompletionWrapper(
            self.__embedding.create, self.__embedding.acreate, self.tracer
        ).acreate(*args, **kwargs)
        stream = kwargs.get("stream", False)
        if not stream:
            non_streaming_response_value = await anext(response)
            return non_streaming_response_value
        return response


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
        response = ChatCompletionWrapper(
            self.__completions.create,
            None,
            self.tracer,
        ).create(*args, **kwargs)

        stream = kwargs.get("stream", False)
        if not stream:
            non_streaming_response_value = next(response)
            return non_streaming_response_value
        return response


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
        response = ChatCompletionWrapper(
            None, self.__completions.create, self.tracer
        ).acreate(*args, **kwargs)

        stream = kwargs.get("stream", False)
        if not stream:
            non_streaming_response_value = await anext(response)
            return non_streaming_response_value
        return response


class AsyncEmbeddingV1Wrapper(NamedWrapper):
    def __init__(self, embedding, tracer):
        self.__embedding = embedding
        self.tracer = tracer
        super().__init__(embedding)

    async def create(self, *args, **kwargs):
        return await EmbeddingWrapper(
            None, self.__embedding.create, self.tracer
        ).acreate(*args, **kwargs)


class ChatV1Wrapper(NamedWrapper[Chat | AsyncChat]):
    def __init__(self, chat: Chat | AsyncChat, tracer: LastMileTracer):
        super().__init__(chat)
        self.tracer = tracer

        import openai

        if isinstance(chat.completions, AsyncCompletions):
            self.completions = AsyncCompletionsV1Wrapper(
                chat.completions, self.tracer
            )
        else:
            self.completions = CompletionsV1Wrapper(
                chat.completions, self.tracer
            )


# This wraps 1.*.* versions of the openai module, eg https://github.com/openai/openai-python/tree/v1.1.0
class OpenAIV1Wrapper(
    NamedWrapper[openai_module.OpenAI | openai_module.AsyncOpenAI]
):
    def __init__(
        self,
        client: openai_module.OpenAI | openai_module.AsyncOpenAI,
        tracer: LastMileTracer,
    ):
        super().__init__(client)
        self.tracer: LastMileTracer = tracer

        self.chat = ChatV1Wrapper(client.chat, self.tracer)

        if isinstance(client.embeddings, AsyncEmbeddings):
            self.embeddings = AsyncEmbeddingV1Wrapper(
                client.embeddings, self.tracer
            )
        else:
            self.embeddings = EmbeddingV1Wrapper(
                client.embeddings, self.tracer
            )


def wrap(
    client_or_module: openai_module.OpenAI | openai_module.AsyncOpenAI,
    tracer: LastMileTracer,
) -> OpenAIV0Wrapper | OpenAIV1Wrapper:
    """
    Wrap the openai module (pre v1) or OpenAI client (post v1) to add tracing.

    :param client_or_module: The openai module or OpenAI client
    """
    if hasattr(client_or_module, "chat") and hasattr(
        client_or_module.chat, "completions"
    ):
        return OpenAIV1Wrapper(client_or_module, tracer)
    return OpenAIV0Wrapper(client_or_module, tracer)


wrap_openai = wrap


### Help methods
def _add_rag_event_with_output(
    tracer: LastMileTracer,
    event_name: str,
    span=None,  # type: ignore
    input: Optional[Any] = None,
    output: Optional[Any] = None,
    event_data: dict[Any, Any] | None = None,
) -> None:
    if output is not None:
        tracer.add_rag_event_for_span(
            event_name,
            span,  # type: ignore
            input=input,
            output=output,
        )
    else:
        tracer.add_rag_event_for_span(
            event_name,
            span,  # type: ignore
            event_data=event_data,
        )
