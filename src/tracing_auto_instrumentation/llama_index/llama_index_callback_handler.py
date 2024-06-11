import logging
from collections import defaultdict
from time import time_ns
from typing import Any, Dict, Optional, Union

from lastmile_eval.rag.debugger.api import (
    LastMileTracer,
    RetrievedNode,
    TextEmbedding,
)
from lastmile_eval.rag.debugger.common.utils import LASTMILE_SPAN_KIND_KEY_NAME
from lastmile_eval.rag.debugger.tracing import get_lastmile_tracer
from llama_index.core.callbacks import CBEventType, EventPayload
from openinference.instrumentation.llama_index._callback import (
    OpenInferenceTraceCallbackHandler,
    payload_to_semantic_attributes,
    _is_streaming_response,  # type: ignore
    _flatten,  # type: ignore
    _ResponseGen,  # type: ignore
    _EventData,  # type: ignore
    # Opinionated params we explicit want to save, see source for full list
    DOCUMENT_SCORE,
    EMBEDDING_EMBEDDINGS,
    EMBEDDING_MODEL_NAME,
    INPUT_VALUE,
    LLM_INVOCATION_PARAMETERS,
    LLM_MODEL_NAME,
    LLM_PROMPT_TEMPLATE,
    LLM_PROMPT_TEMPLATE_VARIABLES,
    LLM_TOKEN_COUNT_COMPLETION,
    LLM_TOKEN_COUNT_PROMPT,
    LLM_TOKEN_COUNT_TOTAL,
    MESSAGE_FUNCTION_CALL_NAME,
    OUTPUT_VALUE,
    RERANKER_MODEL_NAME,
    RERANKER_OUTPUT_DOCUMENTS,
    RERANKER_QUERY,
    RERANKER_TOP_K,
    RETRIEVAL_DOCUMENTS,
    TOOL_CALL_FUNCTION_NAME,
    TOOL_NAME,
    # # Explicit chose not to do these two because context can be huge and we
    # # can extract this from both the prompt template template and variables
    # LLM_INPUT_MESSAGES,
    # LLM_OUTPUT_MESSAGES,
)
from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY  # type: ignore
from opentelemetry.sdk.trace import ReadableSpan

from ..utils import DEFAULT_TRACER_NAME_PREFIX

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

PARAM_SET_SUBSTRING_MATCHES = (
    DOCUMENT_SCORE,
    EMBEDDING_EMBEDDINGS,
    EMBEDDING_MODEL_NAME,
    INPUT_VALUE,
    LLM_INVOCATION_PARAMETERS,
    LLM_MODEL_NAME,
    LLM_PROMPT_TEMPLATE,
    LLM_PROMPT_TEMPLATE_VARIABLES,
    LLM_TOKEN_COUNT_COMPLETION,
    LLM_TOKEN_COUNT_PROMPT,
    LLM_TOKEN_COUNT_TOTAL,
    MESSAGE_FUNCTION_CALL_NAME,
    OUTPUT_VALUE,
    RERANKER_MODEL_NAME,
    RERANKER_OUTPUT_DOCUMENTS,
    RERANKER_QUERY,
    RERANKER_TOP_K,
    RETRIEVAL_DOCUMENTS,
    TOOL_CALL_FUNCTION_NAME,
    TOOL_NAME,
)


class LlamaIndexCallbackHandler(OpenInferenceTraceCallbackHandler):
    """
    This is a callback handler for automatically instrumenting with
    LLamaIndex. Here's how to use it:

    ```
    from lastmile_eval.rag.debugger.tracing import LlamaIndexCallbackHandler
    llama_index.core.global_handler = LlamaIndexCallbackHandler()
    # Do regular LlamaIndex calls as usual
    ```
    """

    def __init__(
        self,
        project_name: Optional[str] = None,
        lastmile_api_token: Optional[str] = None,
    ):
        tracer: LastMileTracer = get_lastmile_tracer(
            tracer_name=project_name
            or (DEFAULT_TRACER_NAME_PREFIX + " - LlamaIndex"),
            lastmile_api_token=lastmile_api_token,
        )
        super().__init__(tracer)

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return

        with self._lock:
            if event_type is CBEventType.TEMPLATING:
                if (
                    parent_id := self._templating_parent_id.pop(event_id, None)
                ) and payload:
                    if parent_id in self._templating_payloads:
                        self._templating_payloads[parent_id].append(payload)
                    else:
                        self._templating_payloads[parent_id] = [payload]
                return
            if not (event_data := self._event_data.pop(event_id, None)):
                return

        event_data.end_time = time_ns()
        is_dispatched = False

        if payload is not None:
            event_data.payloads.append(payload.copy())
            if isinstance(
                (exception := payload.get(EventPayload.EXCEPTION)), Exception
            ):
                event_data.exceptions.append(exception)
            try:
                event_data.attributes.update(
                    payload_to_semantic_attributes(
                        event_type, payload, is_event_end=True
                    ),
                )
            except Exception:
                logger.exception(
                    f"Failed to convert payload to semantic attributes. "
                    f"event_type={event_type}, payload={payload}",
                )
            if (
                _is_streaming_response(
                    response := payload.get(EventPayload.RESPONSE)
                )
                and response.response_gen is not None
            ):
                response.response_gen = _ResponseGen(
                    response.response_gen, event_data
                )
                is_dispatched = True

        if not is_dispatched:
            _finish_tracing(
                event_data,
                tracer=self._tracer,
                event_type=event_type,
            )
        return


def _finish_tracing(
    event_data: _EventData,
    tracer: LastMileTracer,
    event_type: CBEventType,
) -> None:
    if not (span := event_data.span):
        return
    attributes = event_data.attributes
    if event_data.exceptions:
        status_descriptions: list[str] = []
        for exception in event_data.exceptions:
            span.record_exception(exception)
            # Follow the format in OTEL SDK for description, see:
            # https://github.com/open-telemetry/opentelemetry-python/blob/2b9dcfc5d853d1c10176937a6bcaade54cda1a31/opentelemetry-api/src/opentelemetry/trace/__init__.py#L588  # noqa E501
            status_descriptions.append(
                f"{type(exception).__name__}: {exception}"
            )
        status = trace_api.Status(
            status_code=trace_api.StatusCode.ERROR,
            description="\n".join(status_descriptions),
        )
    else:
        status = trace_api.Status(status_code=trace_api.StatusCode.OK)
    span.set_status(status=status)
    try:
        span.set_attributes(dict(_flatten(attributes)))

        # Remove "serialized" from the spans
        # We need to do this because this info stores the API keys and we
        # want to remove. Other pertinent data is stored in the span attributes
        # like model name and invocation params already
        if (
            isinstance(span, ReadableSpan)
            and span._attributes is not None
            and "serialized" in span._attributes
        ):
            del span._attributes["serialized"]

        if not _should_skip(event_type):
            serializable_payload: Dict[str, Any] = {}
            for key, value in span._attributes.items():
                # Only save the opinionated data to event data and param set
                if _save_to_param_set(key):
                    serializable_payload[key] = value

            event_type = event_data.event_type
            if event_type == CBEventType.QUERY:
                tracer.add_query_event(
                    query=serializable_payload[INPUT_VALUE],
                    llm_output=serializable_payload[OUTPUT_VALUE],
                    span=span,
                    should_also_save_in_span=True,
                )
            elif event_type == CBEventType.RETRIEVE:
                # extract document data

                # doc_info contains as key the index of the document and value the
                # info of the document
                # Example: {0: {'id': 'doc1', 'score': 0.5, 'content': 'doc1 content'}}
                doc_info: defaultdict[int, dict[str, Union[str, float]]] = (
                    defaultdict(dict)
                )
                for key, value in serializable_payload.items():
                    if RETRIEVAL_DOCUMENTS in key:
                        # Example of key would be "retrieval.documents.1.document.score"
                        key_parts = key.split(".")
                        doc_index: int = -1
                        for part in key_parts:
                            if part.isnumeric():
                                doc_index = int(part)
                        if doc_index == -1:
                            continue

                        # info will be either "id", "score", or "content"
                        info_type = key.split(".")[-1]
                        doc_info[doc_index][info_type] = value

                # build list of retrieved nodes
                retrieved_nodes: list[RetrievedNode] = []
                # Sort the keys (document index) to add them in correct order
                # to the retrieved nodes array
                for info_dict in dict(sorted(doc_info.items())).values():
                    retrieved_nodes.append(
                        RetrievedNode(
                            id=str(info_dict["id"]),
                            score=float(info_dict["score"]),
                            text=str(info_dict["content"]),
                        )
                    )
                tracer.add_retrieval_event(
                    query=serializable_payload[INPUT_VALUE],
                    retrieved_nodes=retrieved_nodes,
                    span=span,
                    should_also_save_in_span=True,
                )
            elif event_type == CBEventType.EMBEDDING:
                # embed_info contains as key the index of the text chunk
                # (or query) and value the embedding info
                # Example: {0: {'text': 'something to embed', 'vector': [0.1, 0.2, 0.3 ... 0.2343]}}
                embed_info: defaultdict[
                    int, dict[str, Union[str, list[float]]]
                ] = defaultdict(dict)
                model_name: str = ""
                for key, value in serializable_payload.items():
                    if EMBEDDING_EMBEDDINGS in key:
                        # Example of key would be "embedding.embeddings.0.embedding.text"
                        key_parts = key.split(".")
                        text_index: int = -1
                        for part in key_parts:
                            if part.isnumeric():
                                text_index = int(part)
                        if text_index == -1:
                            continue

                        # info will be either "text", or "vector"
                        info_type = key.split(".")[-1]
                        embed_info[text_index][info_type] = value
                    if EMBEDDING_MODEL_NAME in key:
                        model_name = value

                # build list of embeddings
                embeddings: list[TextEmbedding] = []
                for key, info_dict in dict(sorted(embed_info.items())).items():
                    embeddings.append(
                        TextEmbedding(
                            id=f"embedding-{key}",
                            text=str(info_dict["text"]),
                            vector=list(info_dict["vector"]),  # type: ignore
                        )
                    )

                if len(embeddings) == 1:
                    tracer.add_embedding_event(
                        embedding=embeddings[0],
                        span=span,
                        should_also_save_in_span=True,
                        metadata={
                            f"{EMBEDDING_MODEL_NAME}": model_name,
                        },
                    )
                elif len(embeddings) > 1:
                    curr_span_index: str = span._name[0]
                    span.update_name(f"{curr_span_index} - multi_embedding")
                    tracer.add_multi_embedding_event(
                        embeddings=embeddings,
                        span=span,
                        should_also_save_in_span=True,
                        metadata={
                            "embedding.count": len(embed_info),
                            f"{EMBEDDING_MODEL_NAME}": model_name,
                        },
                    )
            elif event_type == CBEventType.LLM:
                template: str = serializable_payload[LLM_PROMPT_TEMPLATE]
                template_variables: dict[str, str] = serializable_payload[
                    LLM_PROMPT_TEMPLATE_VARIABLES
                ]
                resolved_prompt = template
                for key, value in serializable_payload[
                    LLM_PROMPT_TEMPLATE_VARIABLES
                ].items():
                    resolved_prompt = template.replace(f"{{{key}}}", value)
                tracer.add_query_event(
                    query=resolved_prompt,
                    # TODO: Scan for the system prompt in the input messages
                    # system_prompt=...
                    llm_output=serializable_payload[OUTPUT_VALUE],
                    span=span,
                    should_also_save_in_span=True,
                    metadata={
                        LLM_INVOCATION_PARAMETERS: serializable_payload[
                            LLM_INVOCATION_PARAMETERS
                        ],
                        LLM_PROMPT_TEMPLATE: template,
                        LLM_PROMPT_TEMPLATE_VARIABLES: template_variables,
                        LLM_TOKEN_COUNT_COMPLETION: serializable_payload[
                            LLM_TOKEN_COUNT_COMPLETION
                        ],
                        LLM_TOKEN_COUNT_PROMPT: serializable_payload[
                            LLM_TOKEN_COUNT_PROMPT
                        ],
                        LLM_TOKEN_COUNT_TOTAL: serializable_payload[
                            LLM_TOKEN_COUNT_TOTAL
                        ],
                    },
                )
            else:
                tracer.add_rag_event_for_span(
                    event_name=str(event_data.event_type),
                    span=span,
                    event_data=serializable_payload,
                    should_also_save_in_span=True,
                )
            tracer.register_params(
                params=serializable_payload,
                should_also_save_in_span=True,
                span=span,
            )

        # Save span kind into span attribute, but don't add it to trace-level
        # params (since that should be for trace-level data) or rag span event
        # (since it's already used for the event name there)
        span.set_attribute(LASTMILE_SPAN_KIND_KEY_NAME, event_data.event_type)

    except Exception:
        logger.exception(
            f"Failed to set attributes on span. event_type={event_data.event_type}, "
            f"attributes={attributes}",
        )
    span.end(end_time=event_data.end_time)


def _should_skip(event_type: CBEventType) -> bool:
    # TODO: Add some actual crieria for skipping to log span events
    return event_type in {
        CBEventType.CHUNKING,
        CBEventType.NODE_PARSING,
        CBEventType.TREE,
        CBEventType.SYNTHESIZE,
    }


def _save_to_param_set(key: str):
    for substring in PARAM_SET_SUBSTRING_MATCHES:
        if substring in key:
            return True
    return False


# TODO: Clean later, for now it's just helpful for me to keep track of all events
def _add_rag_event_to_tracer() -> None:
    """
    Skip these
    CHUNKING = "chunking"
    NODE_PARSING = "node_parsing"
    SYNTHESIZE = "synthesize"
    TREE = "tree"

    Done
    QUERY = "query"
    RETRIEVE = "retrieve"
    EMBEDDING = "embedding"
    LLM = "llm" # part of query

    TODO
    SUB_QUESTION = "sub_question"
    TEMPLATING = "templating"
    FUNCTION_CALL = "function_call"
    RERANKING = "reranking"
    EXCEPTION = "exception"
    AGENT_STEP = "agent_step"
    """
