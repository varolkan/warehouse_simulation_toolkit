"""ROS parameters for agv_core's configuration classes.

Every field of a config dataclass becomes a parameter `<prefix>.<field>` with the
dataclass's default, so each threshold can be set in a YAML file or on the command line
without a line of code, and `ros2 param list` shows them all.
"""

import dataclasses
from typing import Type, TypeVar

T = TypeVar('T')


def declare_config(node, prefix: str, config_class: Type[T]) -> T:
    """Declare one parameter per field of `config_class` and return the config they make."""
    defaults = config_class()
    values = {}
    for field in dataclasses.fields(config_class):
        default = getattr(defaults, field.name)
        name = f'{prefix}.{field.name}'
        value = node.declare_parameter(name, default).value
        if isinstance(default, float) and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        if type(value) is not type(default):
            raise ValueError(f'parameter {name} must be a {type(default).__name__}, '
                             f'not {value!r}')
        values[field.name] = value
    return config_class(**values)


def stamp_seconds(stamp) -> float:
    """Return a builtin_interfaces/Time as seconds."""
    return stamp.sec + stamp.nanosec * 1e-9
