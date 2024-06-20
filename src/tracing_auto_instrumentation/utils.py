# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
import json
from typing import Any, Generic, TypeVar


T_co = TypeVar("T_co", covariant=True)
DEFAULT_TRACER_NAME_PREFIX = "LastMileTracer"


class NamedWrapper(Generic[T_co]):
    def __init__(self, wrapped: T_co):
        self.__wrapped = wrapped

    def __getattr__(self, name: str):
        return getattr(self.__wrapped, name)


def json_serialize_anything(obj: Any) -> str:
    try:
        return json.dumps(
            obj, sort_keys=True, indent=2, default=lambda o: o.__dict__
        )
    except Exception as e:
        return json.dumps(
            {
                "object_as_string": str(obj),
                "serialization_error": str(e),
            }
        )
