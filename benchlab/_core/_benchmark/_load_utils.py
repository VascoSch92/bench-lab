import importlib
from typing import Any

from benchlab._core._types import InstanceType

__all__ = ["get_instances_from_json"]


def get_instances_from_json(json_instances: list[dict[str, Any]]) -> list[InstanceType]:
    instances = []

    class_module: str | None = None
    class_name: str | None = None
    instance_cls: InstanceType | None = None
    for instance in json_instances:
        instance_class_module = instance.pop("class_module")
        instance_class_name = instance.pop("class_name")

        if instance_cls is None:
            # we need to import the class cls just once as
            # we suppose all the classes have the same instance
            class_module = instance_class_module
            class_name = instance_class_name
            module = importlib.import_module(class_module)
            instance_cls = getattr(module, class_name, None)

            if instance_cls is None:
                # todo: better error msg
                raise ValueError

        if instance_class_module != class_module or instance_class_name != class_name:
            raise ValueError("All the instance must be of the same class.")

        loaded_instance = instance_cls(**instance)
        instances.append(loaded_instance)
    return instances
