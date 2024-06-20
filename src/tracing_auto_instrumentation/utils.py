# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
import json
from typing import Any, Generic, TypeVar


T_co = TypeVar("T_co", covariant=True)
DEFAULT_TRACER_NAME_PREFIX = "LastMileTracer"


class Wrapper(Generic[T_co]):
    """
    Parent class used by a proxy class to wrap an instance of type T_co,
    allowing that proxy to access the object's attributes directly with
    the same interface as if the proxy class was the object itself.

    Example:
    ```python
    obj = MyObject()
    obj.get_id() # returns 1

    class MyProxy(Wrapper[MyObject]):
        def __init__(self, my_object: MyObject):
            super().__init__(my_object)

    proxy = MyProxy(obj)
    proxy.get_id() # calls obj.get_id() and returns 1
    ```

    If we didn't have this, we would need to have to manually define each
    method for the proxy directly:
    ```python
    class MyProxy():
        def __init__(self, my_object: MyObject):
            self.my_object = my_object

        def get_id(self):
            return self.my_object.get_id()

    proxy = MyProxy(obj)
    proxy.get_id() # calls proxy.my_object.get_id() and returns 1
    ```
    """

    def __init__(self, obj: T_co):
        self._obj = obj

    def __getattr__(self, name: str):
        return getattr(self._obj, name)

    # TODO (rossdan): Add __setattr__ and __delattr__ to allow
    # setting and deleting attributes on the


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
