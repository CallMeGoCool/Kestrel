"""
Strict, recursive dict -> dataclass conversion for registry models.

Why this exists: `Model(**payload)` leaves nested objects as plain dicts and
silently drops unknown keys. `from_dict` rebuilds nested dataclasses from the
type hints, reports every problem in one SchemaError, and can collect
unknown keys as warnings.
"""
from __future__ import annotations

import dataclasses
import types
import typing
from typing import Any, List, Optional


class SchemaError(ValueError):
    """Raised with every problem found, not just the first."""

    def __init__(self, problems: List[str]) -> None:
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


_UNION_TYPES = (typing.Union, getattr(types, "UnionType", typing.Union))


def from_dict(cls: type, data: Any, warnings: Optional[List[str]] = None) -> Any:
    """Build `cls` from a dict. Unknown keys are appended to `warnings`."""
    problems: List[str] = []
    obj = _build(cls, data, cls.__name__, problems, warnings)
    if problems:
        raise SchemaError(problems)
    return obj


def coerce_fields(obj: Any) -> None:
    """
    Convert nested dicts inside an already-constructed dataclass.
    Call from __post_init__ so Model(**payload) also yields typed children.
    """
    hints = typing.get_type_hints(type(obj))
    problems: List[str] = []
    for f in dataclasses.fields(obj):
        value = getattr(obj, f.name)
        setattr(
            obj,
            f.name,
            _convert(hints[f.name], value, f"{type(obj).__name__}.{f.name}", problems, None),
        )
    if problems:
        raise SchemaError(problems)


def _build(cls, data, path, problems, warnings):
    if isinstance(data, cls):
        return data
    if not isinstance(data, dict):
        problems.append(f"{path}: expected an object, got {type(data).__name__}")
        return None

    hints = typing.get_type_hints(cls)
    start = len(problems)
    kwargs = {}
    known = set()

    for f in dataclasses.fields(cls):
        if not f.init:
            continue
        known.add(f.name)
        if f.name in data:
            kwargs[f.name] = _convert(
                hints[f.name], data[f.name], f"{path}.{f.name}", problems, warnings
            )
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            problems.append(f"{path}: missing required field {f.name!r}")

    if warnings is not None:
        for key in data:
            if key not in known:
                warnings.append(f"{path}: unknown field {key!r}")

    if len(problems) > start:
        return None

    try:
        return cls(**kwargs)
    except SchemaError as exc:
        problems.extend(exc.problems)
        return None


def _convert(tp, value, path, problems, warnings):
    if tp is Any:
        return value

    origin = typing.get_origin(tp)
    args = typing.get_args(tp)

    if origin in _UNION_TYPES:
        if value is None and type(None) in args:
            return None
        options = [a for a in args if a is not type(None)]
        if len(options) == 1:
            return _convert(options[0], value, path, problems, warnings)
        return value

    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        return _build(tp, value, path, problems, warnings)

    if tp is list or origin is list:
        if not isinstance(value, list):
            problems.append(f"{path}: expected a list, got {type(value).__name__}")
            return value
        item_tp = args[0] if args else Any
        return [
            _convert(item_tp, item, f"{path}[{i}]", problems, warnings)
            for i, item in enumerate(value)
        ]

    if tp is dict or origin is dict:
        if not isinstance(value, dict):
            problems.append(f"{path}: expected an object, got {type(value).__name__}")
            return value
        return dict(value)

    if tp is bool:
        if not isinstance(value, bool):
            problems.append(f"{path}: expected true/false, got {type(value).__name__}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            problems.append(f"{path}: expected an integer, got {type(value).__name__}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            problems.append(f"{path}: expected a number, got {type(value).__name__}")
            return value
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            problems.append(f"{path}: expected text, got {type(value).__name__}")
        return value

    return value