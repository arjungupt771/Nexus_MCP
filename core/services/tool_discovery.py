
"""Discover MCP tools and populate the Nexus tool catalog."""

from __future__ import annotations

from core.exceptions import DiscoveryError
from core.interfaces.tool import ToolCatalog
from core.interfaces.transport import MCPTransport
from core.models.tool import ToolDefinition
from core.services.schema_normalizer import SchemaNormalizer


class ToolDiscoveryService:
    """Discover and normalize tools exposed by an MCP server."""

    def __init__(
        self,
        catalog: ToolCatalog,
        normalizer: SchemaNormalizer | None = None,
        *,
        max_pages: int = 100,
    ) -> None:
        if max_pages < 1:
            raise ValueError("max_pages must be at least 1")

        self._catalog = catalog
        self._normalizer = normalizer or SchemaNormalizer()
        self._max_pages = max_pages

    async def discover(
        self,
        server_id: str,
        server_name: str,
        transport: MCPTransport,
    ) -> list[ToolDefinition]:
        """Discover every page and atomically update the tool catalog."""

        if not isinstance(server_id, str) or not server_id.strip():
            raise ValueError("server_id cannot be empty")

        if not isinstance(server_name, str) or not server_name.strip():
            raise ValueError("server_name cannot be empty")

        discovered: list[ToolDefinition] = []
        seen_names: set[str] = set()
        seen_cursors: set[str] = set()
        cursor: str | None = None

        try:
            for _ in range(self._max_pages):
                if cursor is None:
                    page = await transport.list_tools()
                else:
                    page = await transport.list_tools(cursor=cursor)

                sdk_tools = getattr(page, "tools", None)

                if not isinstance(sdk_tools, list):
                    raise DiscoveryError(
                        f"Server '{server_id}' returned an invalid tools response.",
                        details={
                            "server_id": server_id,
                            "reason": "invalid_tools_response",
                        },
                    )

                for sdk_tool in sdk_tools:
                    name = getattr(sdk_tool, "name", None)

                    if not isinstance(name, str) or not name.strip():
                        raise DiscoveryError(
                            f"Server '{server_id}' returned an invalid tool name.",
                            details={
                                "server_id": server_id,
                                "reason": "invalid_tool_name",
                            },
                        )

                    name = name.strip()

                    if name in seen_names:
                        raise DiscoveryError(
                            f"Server '{server_id}' returned duplicate tool '{name}'.",
                            details={
                                "server_id": server_id,
                                "tool_name": name,
                                "reason": "duplicate_tool",
                            },
                        )

                    seen_names.add(name)

                    description = getattr(
                        sdk_tool,
                        "description",
                        None,
                    )

                    if description is None:
                        description = ""

                    if not isinstance(description, str):
                        raise DiscoveryError(
                            f"Tool '{name}' has an invalid description.",
                            details={
                                "server_id": server_id,
                                "tool_name": name,
                                "reason": "invalid_description",
                            },
                        )

                    schema = self._normalizer.normalize(
                        getattr(sdk_tool, "input_schema", None),
                        name,
                    )

                    discovered.append(
                        ToolDefinition(
                            server_id=server_id,
                            server_name=server_name,
                            name=name,
                            description=description,
                            input_schema=schema,
                        )
                    )

                next_cursor = getattr(page, "next_cursor", None)

                if next_cursor is None:
                    break

                if (
                    not isinstance(next_cursor, str)
                    or not next_cursor
                ):
                    raise DiscoveryError(
                        f"Server '{server_id}' returned an invalid pagination cursor.",
                        details={
                            "server_id": server_id,
                            "reason": "invalid_cursor",
                        },
                    )

                if next_cursor in seen_cursors:
                    raise DiscoveryError(
                        f"Server '{server_id}' repeated a pagination cursor.",
                        details={
                            "server_id": server_id,
                            "reason": "repeated_cursor",
                        },
                    )

                seen_cursors.add(next_cursor)
                cursor = next_cursor

            else:
                raise DiscoveryError(
                    f"Server '{server_id}' exceeded the discovery page limit.",
                    details={
                        "server_id": server_id,
                        "max_pages": self._max_pages,
                        "reason": "page_limit_exceeded",
                    },
                )

            # Replace the catalog snapshot only after complete discovery.
            return await self._catalog.upsert_tools(
                server_id,
                discovered,
            )

        except DiscoveryError:
            raise

        except Exception as exc:
            raise DiscoveryError(
                f"Tool discovery failed for server '{server_id}'.",
                details={
                    "server_id": server_id,
                    "cause_type": type(exc).__name__,
                },
            ) from exc
