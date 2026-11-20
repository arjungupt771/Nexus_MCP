
"""Normalize and validate MCP tool input schemas."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from jsonschema.validators import validator_for

from core.exceptions import DiscoveryError


class SchemaNormalizer:
    """Validate and normalize MCP tool input schemas."""

    def normalize(
        self,
        schema: Any,
        tool_name: str,
    ) -> dict[str, Any]:
        """Return a safe, independent JSON Schema dictionary."""

        if not isinstance(schema, dict):
            raise DiscoveryError(
                f"Tool '{tool_name}' has an invalid input schema.",
                details={
                    "tool_name": tool_name,
                    "reason": "schema_not_object",
                },
            )

        normalized = deepcopy(schema)

        if normalized.get("type") != "object":
            raise DiscoveryError(
                f"Tool '{tool_name}' must have an object input schema.",
                details={
                    "tool_name": tool_name,
                    "reason": "invalid_root_type",
                },
            )

        properties = normalized.get("properties", {})
        required = normalized.get("required", [])

        if not isinstance(properties, dict):
            raise DiscoveryError(
                f"Tool '{tool_name}' has invalid schema properties.",
                details={
                    "tool_name": tool_name,
                    "reason": "invalid_properties",
                },
            )

        if not isinstance(required, list):
            raise DiscoveryError(
                f"Tool '{tool_name}' has an invalid required field list.",
                details={
                    "tool_name": tool_name,
                    "reason": "invalid_required",
                },
            )

        if not all(
            isinstance(key, str) and key
            for key in properties
        ):
            raise DiscoveryError(
                f"Tool '{tool_name}' has invalid property names.",
                details={
                    "tool_name": tool_name,
                    "reason": "invalid_property_name",
                },
            )

        if not all(
            isinstance(item, str) and item
            for item in required
        ):
            raise DiscoveryError(
                f"Tool '{tool_name}' has invalid required field names.",
                details={
                    "tool_name": tool_name,
                    "reason": "invalid_required_name",
                },
            )

        if len(required) != len(set(required)):
            raise DiscoveryError(
                f"Tool '{tool_name}' has duplicate required fields.",
                details={
                    "tool_name": tool_name,
                    "reason": "duplicate_required",
                },
            )

        missing = set(required) - set(properties)

        if missing:
            raise DiscoveryError(
                f"Tool '{tool_name}' requires undefined properties.",
                details={
                    "tool_name": tool_name,
                    "reason": "required_property_missing",
                    "missing_properties": sorted(missing),
                },
            )

        normalized.setdefault("properties", {})
        normalized.setdefault("required", [])

        try:
            json.dumps(normalized, allow_nan=False)

            # Respect the JSON Schema dialect declared by the server.
            validator_class = validator_for(normalized)
            validator_class.check_schema(normalized)

        except Exception as exc:
            raise DiscoveryError(
                f"Tool '{tool_name}' has an invalid JSON Schema.",
                details={
                    "tool_name": tool_name,
                    "reason": "invalid_json_schema",
                    "cause_type": type(exc).__name__,
                },
            ) from exc

        return normalized
