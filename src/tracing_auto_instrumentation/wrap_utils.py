import json
from typing import Any, Generic, TypeVar
import importlib.util

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


def verify_package_installed(package_name, instrumentor_name: str):
    """
    Checks if the specified package is installed.

    Args:
        package_name (str): The name of the package to check.
        instrumentor_name (str): The name of the instrumentor associated with the package.
 
    Raises:
        ModuleNotFoundError: If the specified package is not installed.
    """
    package_spec = importlib.util.find_spec(package_name)
    if package_spec is None:
        raise ModuleNotFoundError(
            f"The '{package_name}' package is not installed, which is required for the '{instrumentor_name}' instrumentor.\n"
            f"To install the package, please run the following command:\n"
            f"    pip install tracing-auto-instrumentation[{instrumentor_name}]\n"
            f"Make sure you have the necessary permissions to install packages."
        )
