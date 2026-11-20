"""References between plan steps.

Arguments of a step may point at the output of earlier steps:

* ``{"$ref": "s1", "path": "data.items.0.full_name"}`` - a value (any type)
* ``{"$ref": "s1", "path": "data.items", "pluck": "full_name"}`` - list of one field
* ``{"$ref": "s2", "path": "data.free_slots.0.start", "add_minutes": 30}`` - ISO time + N min
* ``{"$template": "Top: {s1.data.items.0.full_name} ({s1.data.total_count})"}`` - text

``path`` is dotted; numeric segments index lists. Omitting ``path`` yields
the step's whole result.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any

REF_KEY = "$ref"
TEMPLATE_KEY = "$template"
_TEMPLATE_FIELD = re.compile(r"\{([A-Za-z][A-Za-z0-9_]*)(?:\.([^{}]+))?\}")
_REF_KEYS = {REF_KEY, "path", "pluck", "add_minutes"}


class ReferenceError_(ValueError):  # noqa: N801 - avoid shadowing builtin ReferenceError
    """A reference could not be resolved (missing step output or path)."""


ReferenceResolutionError = ReferenceError_


def find_refs(value: Any) -> set[str]:
    """Step ids referenced anywhere inside ``value``."""
    found: set[str] = set()

    if isinstance(value, dict):
        if REF_KEY in value and isinstance(value[REF_KEY], str):
            found.add(value[REF_KEY])
        elif TEMPLATE_KEY in value and isinstance(value[TEMPLATE_KEY], str):
            found.update(m.group(1) for m in _TEMPLATE_FIELD.finditer(value[TEMPLATE_KEY]))
        else:
            for item in value.values():
                found |= find_refs(item)
    elif isinstance(value, list):
        for item in value:
            found |= find_refs(item)

    return found


def get_path(data: Any, path: str | None, *, step: str = "") -> Any:
    if not path:
        return data

    current = data

    for segment in path.split("."):
        try:
            if isinstance(current, list):
                current = current[int(segment)]
            elif isinstance(current, dict):
                current = current[segment]
            else:
                raise KeyError(segment)
        except (KeyError, IndexError, ValueError):
            raise ReferenceError_(
                f"Reference '{step}.{path}' could not be resolved at '{segment}'."
            ) from None

    return current


def _render(value: Any) -> str:
    if isinstance(value, str):
        return value

    return json.dumps(value, default=str, ensure_ascii=False, separators=(",", ":"))


def _resolve_ref(spec: dict[str, Any], results: Mapping[str, Any]) -> Any:
    extra = set(spec) - _REF_KEYS

    if extra:
        raise ReferenceError_(f"Unsupported keys in reference: {', '.join(sorted(extra))}.")

    step = spec[REF_KEY]

    if step not in results:
        raise ReferenceError_(f"Step '{step}' has no result to reference.")

    value = get_path(results[step], spec.get("path"), step=step)
    pluck = spec.get("pluck")

    if pluck is not None:
        if not isinstance(value, list):
            raise ReferenceError_(f"Cannot pluck '{pluck}' from a non-list at '{step}'.")

        value = [get_path(item, pluck, step=step) for item in value]

    minutes = spec.get("add_minutes")

    if minutes is not None:
        try:
            moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            value = (moment + timedelta(minutes=float(minutes))).isoformat()
        except (TypeError, ValueError):
            raise ReferenceError_(
                f"Cannot add minutes: '{step}' value is not an ISO datetime."
            ) from None

    return value


def resolve(value: Any, results: Mapping[str, Any]) -> Any:
    """Deep-copy ``value`` with every reference replaced by its target."""
    if isinstance(value, dict):
        if REF_KEY in value:
            return _resolve_ref(value, results)

        if TEMPLATE_KEY in value:
            template = value[TEMPLATE_KEY]

            def _fill(match: re.Match[str]) -> str:
                step = match.group(1)

                if step not in results:
                    raise ReferenceError_(f"Step '{step}' has no result to reference.")

                return _render(get_path(results[step], match.group(2), step=step))

            return _TEMPLATE_FIELD.sub(_fill, template)

        return {key: resolve(item, results) for key, item in value.items()}

    if isinstance(value, list):
        return [resolve(item, results) for item in value]

    return value
