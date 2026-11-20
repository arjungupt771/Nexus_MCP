
"""Validate MCP tool arguments against discovered JSON Schemas."""

from __future__ import annotations

import json
from typing import Any

from jsonschema import FormatChecker
from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for

from core.exceptions import DiscoveryError
from core.models.tool import ToolDefinition


class ArgumentValidator:
    """Validate tool arguments before sending them to an MCP server."""

    def validate(
        self,
        tool: ToolDefinition,
        arguments: dict[str, Any],
    ) -> None:
        """Raise ValueError when arguments do not satisfy the tool schema."""

        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be a JSON object.")

        schema = tool.input_schema

        if not isinstance(schema, dict):
            raise DiscoveryError(
                f"Tool '{tool.qualified_name}' has an invalid schema.",
                details={"reason": "schema_not_object"},
            )

        try:
            # Reject values that cannot safely be sent as JSON.
            json.dumps(arguments, allow_nan=False)

            validator_class = validator_for(schema)
            validator_class.check_schema(schema)

        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Tool arguments must contain JSON-compatible values."
            ) from exc

        except SchemaError as exc:
            raise DiscoveryError(
                f"Tool '{tool.qualified_name}' has an invalid JSON Schema.",
                details={
                    "tool_name": tool.name,
                    "reason": "invalid_schema",
                },
            ) from exc

        validator = validator_class(
            schema,
            format_checker=FormatChecker(),
        )

        errors = sorted(
            validator.iter_errors(arguments),
            key=lambda error: (
                tuple(str(part) for part in error.absolute_path),
                error.message,
            ),
        )

        if errors:
            messages = []

            for error in errors[:10]:
                location = ".".join(
                    str(part) for part in error.absolute_path
                ) or "<root>"

                messages.append(
                    f"{location}: {error.message}"
                )

            raise ValueError(
                "Invalid tool arguments: " + "; ".join(messages)
            )
