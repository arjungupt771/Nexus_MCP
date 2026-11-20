from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ToolDefinition(BaseModel):
    """Normalized representation of a discovered MCP tool."""

    model_config = ConfigDict(extra="forbid")

    server_id: str = Field(min_length=1)
    server_name: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=150)
    description: str = ""
    input_schema: dict[str, Any]

    @field_validator("server_id", "server_name", "name")
    @classmethod
    def validate_identifiers(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Identifier cannot be empty.")

        return value

    @field_validator("input_schema")
    @classmethod
    def validate_input_schema(
        cls,
        value: dict[str, Any],
    ) -> dict[str, Any]:
        if value.get("type") != "object":
            raise ValueError("MCP tool input schema must have type 'object'.")

        properties = value.get("properties", {})
        required = value.get("required", [])

        if not isinstance(properties, dict):
            raise ValueError("Schema properties must be an object.")

        if not isinstance(required, list):
            raise ValueError("Schema required fields must be a list.")

        for field in required:
            if not isinstance(field, str):
                raise ValueError("Required field names must be strings.")

            if field not in properties:
                raise ValueError(f"Required field '{field}' is missing from properties.")

        return value

    @property
    def qualified_name(self) -> str:
        """Return a server-qualified tool identifier."""

        return f"{self.server_id}.{self.name}"


class ToolCatalogDiff(BaseModel):
    """Difference between two tool snapshots of one server."""

    model_config = ConfigDict(extra="forbid")

    server_id: str
    added: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    changed: list[str] = Field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.removed or self.changed)

    @classmethod
    def compute(
        cls,
        server_id: str,
        previous: list[ToolDefinition],
        current: list[ToolDefinition],
    ) -> ToolCatalogDiff:
        """Compare snapshots by tool name; schema/description edits are 'changed'."""
        before = {tool.name: tool for tool in previous}
        after = {tool.name: tool for tool in current}

        return cls(
            server_id=server_id,
            added=sorted(after.keys() - before.keys()),
            removed=sorted(before.keys() - after.keys()),
            changed=sorted(
                name
                for name in before.keys() & after.keys()
                if (before[name].description, before[name].input_schema)
                != (after[name].description, after[name].input_schema)
                or before[name].server_name != after[name].server_name
            ),
        )
