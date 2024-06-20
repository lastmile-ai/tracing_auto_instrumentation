# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
import json
import time
from typing import (
    Callable,
    ParamSpecArgs,
    ParamSpecKwargs,
)

import openai
from lastmile_eval.rag.debugger.api import LastMileTracer
from openai.resources.chat import (
    Chat,
    Completions,
)
from openai.resources.embeddings import (
    Embeddings,
)
from openai.types import CreateEmbeddingResponse
from openai.types.chat import ChatCompletionChunk, ChatCompletion
from openai import Stream

from ..utils import (
    Wrapper,
    json_serialize_anything,
)

from .shared import (
    add_rag_event_with_output,
    parse_params,
    flatten_json,
)


class OpenAIWrapper(Wrapper[openai.OpenAI]):
    def __init__(
        self,
        client: openai.OpenAI,
        tracer: LastMileTracer,
    ):
        super().__init__(client)
        self.tracer: LastMileTracer = tracer
        self.chat = ChatWrapper(client.chat, self.tracer)
        self.embeddings = EmbeddingWrapper(client.embeddings, self.tracer)


class EmbeddingWrapper(Wrapper[Embeddings]):
    def __init__(self, embedding: Embeddings, tracer: LastMileTracer):
        super().__init__(embedding)
        self._embedding = embedding
        self.tracer = tracer

    def create(
        self, *args: ParamSpecArgs, **kwargs: ParamSpecKwargs
    ) -> CreateEmbeddingResponse:
        return self._create_impl(*args, **kwargs)

    def _create_impl(
        self, *args: ParamSpecArgs, **kwargs: ParamSpecKwargs
    ) -> CreateEmbeddingResponse:
        params = parse_params(kwargs)
        # params_flat = flatten_json(params)

        with self.tracer.start_as_current_span("embedding-create") as span:
            raw_response = self._embedding.create(*args, **kwargs)
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


class ChatWrapper(Wrapper[Chat]):
    def __init__(self, chat: Chat, tracer: LastMileTracer):
        super().__init__(chat)
        self.completions = CompletionsWrapper(chat.completions, tracer)


class CompletionsWrapper(Wrapper[Completions]):
    def __init__(
        self,
        completions: Completions,
        tracer: LastMileTracer,
    ):
        super().__init__(completions)
        self.__completions = completions
        self.tracer = tracer

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
        params = parse_params(kwargs)
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
                    add_rag_event_with_output(
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
                    add_rag_event_with_output(
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


class EmbeddingWrapperImpl:
    def __init__(
        self,
        create_fn: Callable[..., CreateEmbeddingResponse],
        tracer: LastMileTracer,
    ):
        self.create_fn = create_fn
        self.tracer: LastMileTracer = tracer

    def create(self, *args: ParamSpecArgs, **kwargs: ParamSpecKwargs):
        params = parse_params(kwargs)
        # params_flat = flatten_json(params)

        with self.tracer.start_as_current_span("embedding-create") as span:
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
