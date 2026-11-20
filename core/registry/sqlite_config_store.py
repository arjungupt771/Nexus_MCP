"""SQLite-backed ServerConfigStore.

Uses SQLAlchemy Core (already a project dependency) behind the
``ServerConfigStore`` interface; blocking calls run in worker threads so the
event loop is never blocked. Pointing the URL at PostgreSQL later requires
no change to callers.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, TypeVar

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    delete,
    insert,
    select,
    update,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.pool import StaticPool

from core.exceptions import (
    RegistryPersistenceError,
    ServerAlreadyRegisteredError,
    ServerNotFoundError,
)
from core.models.server_config import ServerRecord, ServerRegistration

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
T = TypeVar("T")

metadata = MetaData()

schema_meta = Table(
    "nexus_schema_meta",
    metadata,
    Column("key", String(50), primary_key=True),
    Column("value", String(100), nullable=False),
)

server_configs = Table(
    "nexus_server_configs",
    metadata,
    Column("server_id", String(100), primary_key=True),
    Column("name", String(150), nullable=False),
    Column("transport", String(30), nullable=False),
    Column("description", Text, nullable=False, default=""),
    Column("tags", Text, nullable=False, default="[]"),
    Column("metadata", Text, nullable=False, default="{}"),
    Column("capabilities", Text, nullable=False, default="[]"),
    Column("enabled", Boolean, nullable=False, default=True),
    Column("config", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("row_version", Integer, nullable=False, default=1),
)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=UTC)


class SQLiteServerConfigStore:
    """Persist server registrations in SQLite."""

    def __init__(self, url: str = "sqlite:///./nexus.db") -> None:
        self._url = url
        self._engine: Engine | None = None
        self._init_lock = asyncio.Lock()
        # In-memory SQLite shares one connection across threads: serialize access.
        self._serial = asyncio.Lock() if url in {"sqlite://", "sqlite:///:memory:"} else None

    @classmethod
    def in_memory(cls) -> SQLiteServerConfigStore:
        return cls("sqlite://")

    def _create_engine(self) -> Engine:
        if self._url in {"sqlite://", "sqlite:///:memory:"}:
            # One shared connection so every thread sees the same database.
            return create_engine(
                "sqlite://",
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )

        return create_engine(self._url, connect_args={"check_same_thread": False})

    async def _run(self, fn: Callable[[Engine], T]) -> T:
        engine = await self._ensure_engine()

        try:
            if self._serial is not None:
                async with self._serial:
                    return await asyncio.to_thread(fn, engine)

            return await asyncio.to_thread(fn, engine)
        except (ServerAlreadyRegisteredError, ServerNotFoundError):
            raise
        except SQLAlchemyError as exc:
            # Log the exception type only; SQL parameters could hold config.
            logger.error("Server config store failure: %s", type(exc).__name__)
            raise RegistryPersistenceError(
                "Server configuration storage failed.",
                details={"cause_type": type(exc).__name__},
            ) from exc

    async def _ensure_engine(self) -> Engine:
        async with self._init_lock:
            if self._engine is None:
                engine = self._create_engine()

                def _setup() -> None:
                    metadata.create_all(engine)

                    with engine.begin() as conn:
                        existing = conn.execute(
                            select(schema_meta.c.value).where(schema_meta.c.key == "version")
                        ).scalar()

                        if existing is None:
                            conn.execute(
                                insert(schema_meta).values(key="version", value=str(SCHEMA_VERSION))
                            )

                try:
                    await asyncio.to_thread(_setup)
                except SQLAlchemyError as exc:
                    engine.dispose()
                    raise RegistryPersistenceError(
                        "Server configuration storage could not be initialized.",
                        details={"cause_type": type(exc).__name__},
                    ) from exc

                self._engine = engine

            return self._engine

    async def initialize(self) -> None:
        await self._ensure_engine()

    @staticmethod
    def _to_row(record: ServerRecord) -> dict[str, Any]:
        reg = record.registration
        return {
            "server_id": reg.server_id,
            "name": reg.name,
            "transport": reg.config.transport.value,
            "description": reg.description,
            "tags": json.dumps(reg.tags),
            "metadata": json.dumps(reg.metadata),
            "capabilities": json.dumps(reg.capabilities),
            "enabled": reg.enabled,
            "config": json.dumps(reg.config.model_dump(mode="json")),
            "created_at": record.created_at,
            "updated_at": record.updated_at,
        }

    @staticmethod
    def _from_row(row: Any) -> ServerRecord:
        registration = ServerRegistration.model_validate(
            {
                "server_id": row.server_id,
                "name": row.name,
                "description": row.description,
                "tags": json.loads(row.tags),
                "metadata": json.loads(row.metadata),
                "capabilities": json.loads(row.capabilities),
                "enabled": bool(row.enabled),
                "config": json.loads(row.config),
            }
        )
        return ServerRecord(
            registration=registration,
            created_at=_aware(row.created_at),
            updated_at=_aware(row.updated_at),
        )

    async def add(self, record: ServerRecord) -> ServerRecord:
        values = self._to_row(record)

        def _op(engine: Engine) -> None:
            try:
                with engine.begin() as conn:
                    conn.execute(insert(server_configs).values(**values, row_version=1))
            except IntegrityError as exc:
                raise ServerAlreadyRegisteredError(
                    f"Server '{values['server_id']}' is already registered.",
                    details={"server_id": values["server_id"]},
                ) from exc

        await self._run(_op)
        return record

    async def update(self, record: ServerRecord) -> ServerRecord:
        values = self._to_row(record)
        server_id = values.pop("server_id")
        values.pop("created_at")

        def _op(engine: Engine) -> None:
            with engine.begin() as conn:
                result = conn.execute(
                    update(server_configs)
                    .where(server_configs.c.server_id == server_id)
                    .values(**values, row_version=server_configs.c.row_version + 1)
                )

                if result.rowcount == 0:
                    raise ServerNotFoundError(
                        f"Server '{server_id}' is not registered.",
                        details={"server_id": server_id},
                    )

        await self._run(_op)
        return record

    async def get(self, server_id: str) -> ServerRecord | None:
        def _op(engine: Engine) -> ServerRecord | None:
            with engine.connect() as conn:
                row = conn.execute(
                    select(server_configs).where(server_configs.c.server_id == server_id)
                ).first()

            return self._from_row(row) if row else None

        return await self._run(_op)

    async def list_records(self) -> list[ServerRecord]:
        def _op(engine: Engine) -> list[ServerRecord]:
            with engine.connect() as conn:
                rows = conn.execute(
                    select(server_configs).order_by(
                        server_configs.c.created_at, server_configs.c.server_id
                    )
                ).all()

            records: list[ServerRecord] = []

            for row in rows:
                try:
                    records.append(self._from_row(row))
                except (ValueError, TypeError, KeyError):
                    # ValidationError is a ValueError. Log the id only.
                    logger.warning("Skipping unreadable server record '%s'.", row.server_id)

            return records

        return await self._run(_op)

    async def delete(self, server_id: str) -> bool:
        def _op(engine: Engine) -> bool:
            with engine.begin() as conn:
                result = conn.execute(
                    delete(server_configs).where(server_configs.c.server_id == server_id)
                )
                return result.rowcount > 0

        return await self._run(_op)

    async def close(self) -> None:
        async with self._init_lock:
            engine, self._engine = self._engine, None

        if engine is not None:
            await asyncio.to_thread(engine.dispose)
