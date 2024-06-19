import json
import time
from typing import (
    Any,
    Callable,
    Coroutine,
    Generator,
    Mapping,
    Optional,
    ParamSpecArgs,
    ParamSpecKwargs,
    TypeVar,
    cast,
)

import openai
from lastmile_eval.rag.debugger.api import LastMileTracer
from openai.resources.chat import (
    AsyncChat,
    AsyncCompletions,
    Chat,
    Completions,
)
from openai.resources.embeddings import (
    AsyncEmbeddings,
    Embeddings,
)
from openai.types import CreateEmbeddingResponse
from openai.types.chat import ChatCompletionChunk, ChatCompletion
from openai import AsyncStream, Stream

from ..utils import (
    NamedWrapper,
    json_serialize_anything,
)

T_co = TypeVar("T_co", covariant=True)  # Success type
U = TypeVar("U")

# pylint: disable=missing-function-docstring


def flatten_json(obj: Mapping[str, Any]):
    return {k: json_serialize_anything(v) for k, v in obj.items()}


def merge_dicts(d1: dict[T_co, Any], d2: dict[T_co, Any]) -> dict[T_co, Any]:
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


class ChatCompletionWrapperImpl:
    def __init__(
        self,
        create_fn: Callable[  # TODO: Map this directly to OpenAI package
            ..., (ChatCompletion | Stream[ChatCompletionChunk])
        ],
        tracer: LastMileTracer,
    ):
        self.create_fn = create_fn
        self.tracer: LastMileTracer = tracer

    def create(self, *args: ParamSpecArgs, **kwargs: ParamSpecKwargs):
        params = _parse_params(kwargs)
        params_flat = flatten_json(params)

        rag_event_input = json_serialize_anything(params)
        with self.tracer.start_as_current_span(
            "chat-completion-create"
        ) as span:
            start = time.time()
            raw_response = self.create_fn(*args, **kwargs)
            if isinstance(raw_response, Stream):

                def gen():
                    first = True
                    accumulated_text = None
                    for item in raw_response:
                        if first:
                            span.set_attribute(
                                "time_to_first_token", time.time() - start
                            )
                            first = False
                        if isinstance(item, ChatCompletionChunk):  # type: ignore
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
                        # TODO: Support tool calls
                        # TODO: Use enum from lastmile-eval package
                        span_kind="query",
                    )

                yield from gen()

            # Non-streaming part
            else:
                log_response = (
                    raw_response
                    if isinstance(raw_response, dict)
                    else raw_response.model_dump()
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
                        # TODO: Support tool calls
                        # TODO: Use enum from lastmile-eval package
                        span_kind="query",
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


class AsyncChatCompletionWrapperImpl:
    def __init__(
        self,
        create_fn: Callable[  # TODO: Map this directly to OpenAI package
            ...,
            Coroutine[
                Any, Any, ChatCompletion | AsyncStream[ChatCompletionChunk]
            ],
        ],
        tracer: LastMileTracer,
    ):
        self.create_fn = create_fn
        self.tracer: LastMileTracer = tracer

    async def create(self, *args: ParamSpecArgs, **kwargs: ParamSpecKwargs):
        params = _parse_params(kwargs)

        rag_event_input = json_serialize_anything(params)
        with self.tracer.start_as_current_span(
            "async-chat-completion-create"
        ) as span:
            span.set_attributes(flatten_json(params))
            start = time.time()
            raw_response = await self.create_fn(*args, **kwargs)
            if isinstance(raw_response, AsyncStream):

                async def gen():
                    first = True
                    accumulated_text = None
                    async for item in raw_response:
                        if first:
                            span.set_attribute(
                                "time_to_first_token", time.time() - start
                            )
                            first = False
                        if isinstance(item, ChatCompletionChunk):  # type: ignore
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
                        # TODO: Support tool calls
                        # TODO: Use enum from lastmile-eval package
                        span_kind="query",
                    )

                async for chunk in gen():
                    yield chunk

            # Non-streaming part
            else:
                log_response = (
                    raw_response
                    if isinstance(raw_response, dict)
                    else raw_response.model_dump()
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
                        # TODO: Support tool calls
                        # TODO: Use enum from lastmile-eval package
                        span_kind="query",
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


def _parse_params(params: dict[str, ParamSpecKwargs]):
    # First, destructively remove span_info
    empty_dict: dict[str, Any] = {}
    ret = cast(dict[str, Any], params.pop("span_info", empty_dict))

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


class EmbeddingWrapperImpl:
    def __init__(
        self,
        create_fn: Callable[..., CreateEmbeddingResponse],
        tracer: LastMileTracer,
    ):
        self.create_fn = create_fn
        self.tracer: LastMileTracer = tracer

    def create(self, *args: ParamSpecArgs, **kwargs: ParamSpecKwargs):
        params = _parse_params(kwargs)
        # params_flat = flatten_json(params)

        with self.tracer.start_as_current_span("embedding") as span:
            raw_response = self.create_fn(*args, **kwargs)
            log_response = (
                raw_response
                if isinstance(raw_response, dict)
                else raw_response.model_dump()
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


class AsyncEmbeddingWrapperImpl:
    def __init__(
        self,
        create_fn: Callable[..., Coroutine[Any, Any, CreateEmbeddingResponse]],
        tracer: LastMileTracer,
    ):
        self.create_fn = create_fn
        self.tracer: LastMileTracer = tracer

    async def create(self, *args: ParamSpecArgs, **kwargs: ParamSpecKwargs):
        params = _parse_params(kwargs)

        with self.tracer.start_as_current_span("async-embedding") as span:
            raw_response = await self.create_fn(*args, **kwargs)
            log_response = (
                raw_response
                if isinstance(raw_response, dict)
                else raw_response.model_dump()
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


class CompletionsWrapper(NamedWrapper[Completions]):
    def __init__(
        self,
        completions: Completions,
        tracer: LastMileTracer,
    ):
        self.__completions = completions
        self.tracer = tracer
        super().__init__(completions)

    def create(
        self, *args: ParamSpecArgs, **kwargs: ParamSpecKwargs
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        # response: Generator[ChatCompletion, Any, Any] | Stream[ChatCompletionChunk] = cast(
        #     ChatCompletion | Stream[ChatCompletionChunk],
        #     ChatCompletionWrapperImpl(
        #         self.__completions.create,
        #         None,
        #         self.tracer,
        #     ).create(*args, **kwargs),
        # )
        response = ChatCompletionWrapperImpl(
            self.__completions.create,
            self.tracer,
        ).create(*args, **kwargs)

        stream = kwargs.get("stream", False)
        if not stream:
            non_streaming_response_value = next(response)
            return non_streaming_response_value
        return response


class EmbeddingWrapper(NamedWrapper[Embeddings]):
    def __init__(self, embedding: Embeddings, tracer: LastMileTracer):
        self.__embedding = embedding
        self.tracer = tracer
        super().__init__(embedding)

    def create(self, *args: ParamSpecArgs, **kwargs: ParamSpecKwargs):
        return EmbeddingWrapperImpl(
            self.__embedding.create, None, self.tracer
        ).create(*args, **kwargs)


class AsyncCompletionsWrapper(NamedWrapper[AsyncCompletions]):
    def __init__(self, completions: AsyncCompletions, tracer: LastMileTracer):
        self.__completions = completions
        self.tracer = tracer
        super().__init__(completions)

    async def create(
        self, *args: ParamSpecArgs, **kwargs: ParamSpecKwargs
    ) -> Coroutine[
        Any, Any, ChatCompletion | AsyncStream[ChatCompletionChunk]
    ]:
        response = AsyncChatCompletionWrapperImpl(
            self.__completions.create,  # --> Coroutine[Any, Any, ChatCompletion | AsyncStream[ChatCompletionChunk]]
            self.tracer,
        ).acreate(*args, **kwargs)

        stream = kwargs.get("stream", False)
        if not stream:
            non_streaming_response_value = await anext(response)
            return non_streaming_response_value
        return response


class AsyncEmbeddingWrapper(NamedWrapper[AsyncEmbeddings]):
    def __init__(self, embedding: AsyncEmbeddings, tracer: LastMileTracer):
        self.__embedding = embedding
        self.tracer = tracer
        super().__init__(embedding)

    async def create(
        self, *args: ParamSpecArgs, **kwargs: ParamSpecKwargs
    ) -> Coroutine[Any, Any, CreateEmbeddingResponse]:
        return await AsyncEmbeddingWrapperImpl(
            None, self.__embedding.create, self.tracer
        ).create(*args, **kwargs)


class ChatWrapper(NamedWrapper[Chat]):
    def __init__(self, chat: Chat, tracer: LastMileTracer):
        super().__init__(chat)
        self.completions = CompletionsWrapper(chat.completions, tracer)


class AsyncChatWrapper(NamedWrapper[AsyncChat]):
    def __init__(self, chat: AsyncChat, tracer: LastMileTracer):
        super().__init__(chat)
        self.completions = AsyncCompletionsWrapper(chat.completions, tracer)


class OpenAIWrapper(NamedWrapper[openai.OpenAI]):
    def __init__(
        self,
        client: openai.OpenAI,
        tracer: LastMileTracer,
    ):
        super().__init__(client)
        self.tracer: LastMileTracer = tracer
        self.chat = ChatWrapper(client.chat, self.tracer)
        self.embeddings = EmbeddingWrapper(client.embeddings, self.tracer)


class AsyncOpenAIWrapper(NamedWrapper[openai.AsyncOpenAI]):
    def __init__(
        self,
        client: openai.AsyncOpenAI,
        tracer: LastMileTracer,
    ):
        super().__init__(client)
        self.tracer: LastMileTracer = tracer
        self.chat = AsyncChatWrapper(client.chat, self.tracer)
        self.embeddings = AsyncEmbeddingWrapper(client.embeddings, self.tracer)


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


### Help methods
def _add_rag_event_with_output(
    tracer: LastMileTracer,
    event_name: str,
    span=None,  # type: ignore
    input: Optional[Any] = None,
    output: Optional[Any] = None,
    event_data: dict[Any, Any] | None = None,
    span_kind: Optional[str] = None,
) -> None:
    # TODO: Replace with rag-specific API instead of add_rag_event_for_span
    if output is not None:
        tracer.add_rag_event_for_span(
            event_name,
            span,  # type: ignore
            input=input,
            output=output,
            should_also_save_in_span=True,
            span_kind=span_kind,
        )
    else:
        tracer.add_rag_event_for_span(
            event_name,
            span,  # type: ignore
            event_data=event_data,
            should_also_save_in_span=True,
            span_kind=span_kind,
        )
