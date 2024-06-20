# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
from typing import Any, Mapping, Optional, ParamSpecKwargs

from lastmile_eval.rag.debugger.api import LastMileTracer
from opentelemetry.trace import Span

from ..utils import json_serialize_anything, T_co


def parse_params(params: dict[str, ParamSpecKwargs]):
    # First, destructively remove span_info
    empty_dict: dict[str, Any] = {}
    ret = params.pop("span_info", empty_dict)

    # Then, copy the rest of the params
    params = {**params}
    messages = params.pop("messages", None)
    return _merge_dicts(
        ret,
        {
            "input": messages,
            "metadata": params,
        },
    )


def _merge_dicts(d1: dict[T_co, Any], d2: dict[T_co, Any]) -> dict[T_co, Any]:
    return {**d1, **d2}


def flatten_json(obj: Mapping[str, Any]):
    return {k: json_serialize_anything(v) for k, v in obj.items()}


def add_rag_event_with_output(
    tracer: LastMileTracer,
    event_name: str,
    span: Optional[Span] = None,  # type: ignore
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
