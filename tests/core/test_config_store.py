from __future__ import annotations

import asyncio
import sqlite3

import pytest
from phase5_support import http_payload, registration

from core.exceptions import (
    RegistryPersistenceError,
    ServerAlreadyRegisteredError,
    ServerNotFoundError,
)
from core.interfaces import ServerConfigStore
from core.models.server_config import ServerRecord
from core.registry import InMemoryServerConfigStore, SQLiteServerConfigStore


@pytest.fixture(params=["memory", "sqlite", "sqlite_file"])
async def store(request, tmp_path):
    if request.param == "memory":
        instance = InMemoryServerConfigStore()
    elif request.param == "sqlite":
        instance = SQLiteServerConfigStore.in_memory()
    else:
        instance = SQLiteServerConfigStore(f"sqlite:///{tmp_path / 'reg.db'}")

    await instance.initialize()
    yield instance
    await instance.close()


def record(server_id: str = "notes", **overrides) -> ServerRecord:
    return ServerRecord(registration=registration(server_id, **overrides))


def test_stores_satisfy_the_interface():
    assert isinstance(InMemoryServerConfigStore(), ServerConfigStore)
    assert isinstance(SQLiteServerConfigStore.in_memory(), ServerConfigStore)


async def test_add_get_roundtrip(store):
    await store.add(record())
    loaded = await store.get("notes")

    assert loaded is not None
    assert loaded.registration == registration()
    assert await store.get("missing") is None


async def test_duplicate_add_is_rejected(store):
    await store.add(record())

    with pytest.raises(ServerAlreadyRegisteredError):
        await store.add(record())


async def test_update_replaces_and_missing_update_fails(store):
    await store.add(record())
    updated = ServerRecord(
        registration=registration("notes", description="changed", enabled=False),
        created_at=(await store.get("notes")).created_at,
    )
    await store.update(updated)

    loaded = await store.get("notes")
    assert loaded.registration.description == "changed"
    assert loaded.registration.enabled is False

    with pytest.raises(ServerNotFoundError):
        await store.update(record("ghost"))


async def test_delete_reports_existence(store):
    await store.add(record())

    assert await store.delete("notes") is True
    assert await store.delete("notes") is False
    assert await store.get("notes") is None


async def test_list_is_ordered_by_creation(store):
    for server_id in ("c", "a", "b"):
        await store.add(record(server_id))
        await asyncio.sleep(0.002)

    assert [r.registration.server_id for r in await store.list_records()] == ["c", "a", "b"]


async def test_concurrent_adds_of_one_id_have_exactly_one_winner(store):
    results = await asyncio.gather(
        *(store.add(record("same")) for _ in range(10)),
        return_exceptions=True,
    )

    assert sum(not isinstance(r, Exception) for r in results) == 1
    assert all(
        isinstance(r, ServerAlreadyRegisteredError) for r in results if isinstance(r, Exception)
    )


async def test_http_registration_roundtrips_through_storage(store):
    from core.models.server_config import ServerRegistration

    http = ServerRegistration.from_payload(http_payload())
    await store.add(ServerRecord(registration=http))

    assert (await store.get("remote")).registration == http


async def test_sqlite_data_survives_a_new_store_instance(tmp_path):
    url = f"sqlite:///{tmp_path / 'persist.db'}"
    first = SQLiteServerConfigStore(url)
    await first.add(record())
    await first.close()

    second = SQLiteServerConfigStore(url)
    assert (await second.get("notes")).registration.name == "Notes Server"
    await second.close()


async def test_sqlite_file_contains_references_not_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "super-secret-runtime-value")
    path = tmp_path / "secrets.db"
    store = SQLiteServerConfigStore(f"sqlite:///{path}")
    await store.add(
        record(
            config={
                "transport": "stdio",
                "command": "python",
                "env": {"GITHUB_TOKEN": "${GITHUB_TOKEN}"},
            }
        )
    )
    await store.close()

    raw = "\n".join(sqlite3.connect(path).iterdump())
    assert "${GITHUB_TOKEN}" in raw
    assert "super-secret-runtime-value" not in raw


async def test_unreadable_rows_are_skipped_not_fatal(tmp_path):
    path = tmp_path / "corrupt.db"
    store = SQLiteServerConfigStore(f"sqlite:///{path}")
    await store.add(record("good"))
    await store.add(record("bad"))
    await store.close()

    connection = sqlite3.connect(path)
    connection.execute("UPDATE nexus_server_configs SET config = 'not json' WHERE server_id='bad'")
    connection.commit()
    connection.close()

    reopened = SQLiteServerConfigStore(f"sqlite:///{path}")
    assert [r.registration.server_id for r in await reopened.list_records()] == ["good"]
    await reopened.close()


async def test_storage_failures_become_structured_errors(tmp_path):
    store = SQLiteServerConfigStore(f"sqlite:///{tmp_path / 'ok.db'}")
    await store.initialize()
    # Drop the table underneath the store to simulate a storage fault.
    connection = sqlite3.connect(tmp_path / "ok.db")
    connection.execute("DROP TABLE nexus_server_configs")
    connection.commit()
    connection.close()

    with pytest.raises(RegistryPersistenceError) as exc_info:
        await store.get("anything")

    assert exc_info.value.retryable is True
    assert exc_info.value.details == {"cause_type": "OperationalError"}
    await store.close()
