import json
from typing import Any, Generic, TypeVar


T_INV = TypeVar("T_INV")


class NamedWrapper(Generic[T_INV]):
    def __init__(self, wrapped: T_INV):
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
