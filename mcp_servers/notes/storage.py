import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "notes.db"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class NotesStore:
    """SQLite-backed persistence layer for notes."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or DEFAULT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS notes (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    archived INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS note_tags (
                    note_id TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    PRIMARY KEY (note_id, tag),
                    FOREIGN KEY (note_id)
                        REFERENCES notes(id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_notes_updated_at
                ON notes(updated_at);

                CREATE INDEX IF NOT EXISTS idx_notes_archived
                ON notes(archived);

                CREATE INDEX IF NOT EXISTS idx_note_tags_tag
                ON note_tags(tag);
                """
            )

    @staticmethod
    def _hydrate(
        connection: sqlite3.Connection,
        rows: list[sqlite3.Row],
    ) -> list[dict[str, Any]]:
        if not rows:
            return []

        note_ids = [row["id"] for row in rows]
        placeholders = ",".join("?" for _ in note_ids)

        tag_rows = connection.execute(
            f"""
            SELECT note_id, tag
            FROM note_tags
            WHERE note_id IN ({placeholders})
            ORDER BY tag
            """,
            note_ids,
        ).fetchall()

        tags_by_note: dict[str, list[str]] = {note_id: [] for note_id in note_ids}

        for tag_row in tag_rows:
            tags_by_note[tag_row["note_id"]].append(tag_row["tag"])

        return [
            {
                "id": row["id"],
                "title": row["title"],
                "content": row["content"],
                "tags": tags_by_note[row["id"]],
                "archived": bool(row["archived"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def create(
        self,
        title: str,
        content: str,
        tags: list[str],
    ) -> dict[str, Any]:
        note_id = str(uuid.uuid4())
        now = utc_now()

        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO notes (
                    id, title, content, archived, created_at, updated_at
                )
                VALUES (?, ?, ?, 0, ?, ?)
                """,
                (note_id, title, content, now, now),
            )

            connection.executemany(
                "INSERT INTO note_tags (note_id, tag) VALUES (?, ?)",
                [(note_id, tag) for tag in tags],
            )

            rows = connection.execute(
                "SELECT * FROM notes WHERE id = ?",
                (note_id,),
            ).fetchall()

            return self._hydrate(connection, rows)[0]

    def get(self, note_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM notes WHERE id = ?",
                (note_id,),
            ).fetchall()

            hydrated = self._hydrate(connection, rows)
            return hydrated[0] if hydrated else None

    def list_notes(
        self,
        include_archived: bool,
        limit: int,
        offset: int,
    ) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM notes
                WHERE (? = 1 OR archived = 0)
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
                """,
                (int(include_archived), limit, offset),
            ).fetchall()

            return self._hydrate(connection, rows)

    def search(
        self,
        query: str,
        include_archived: bool,
        limit: int,
    ) -> list[dict[str, Any]]:
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"%{escaped}%"

        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT n.*
                FROM notes n
                LEFT JOIN note_tags t ON t.note_id = n.id
                WHERE (? = 1 OR n.archived = 0)
                  AND (
                    n.title LIKE ? ESCAPE '\\'
                    OR n.content LIKE ? ESCAPE '\\'
                    OR t.tag LIKE ? ESCAPE '\\'
                  )
                ORDER BY n.updated_at DESC
                LIMIT ?
                """,
                (
                    int(include_archived),
                    pattern,
                    pattern,
                    pattern,
                    limit,
                ),
            ).fetchall()

            return self._hydrate(connection, rows)

    def update(
        self,
        note_id: str,
        title: str | None,
        content: str | None,
        tags: list[str] | None,
    ) -> dict[str, Any] | None:
        with self._connection() as connection:
            existing = connection.execute(
                "SELECT id FROM notes WHERE id = ?",
                (note_id,),
            ).fetchone()

            if existing is None:
                return None

            updates: list[str] = []
            values: list[Any] = []

            if title is not None:
                updates.append("title = ?")
                values.append(title)

            if content is not None:
                updates.append("content = ?")
                values.append(content)

            if updates or tags is not None:
                updates.append("updated_at = ?")
                values.append(utc_now())

            if updates:
                values.append(note_id)
                connection.execute(
                    f"UPDATE notes SET {', '.join(updates)} WHERE id = ?",
                    values,
                )

            if tags is not None:
                connection.execute(
                    "DELETE FROM note_tags WHERE note_id = ?",
                    (note_id,),
                )
                connection.executemany(
                    "INSERT INTO note_tags (note_id, tag) VALUES (?, ?)",
                    [(note_id, tag) for tag in tags],
                )

            rows = connection.execute(
                "SELECT * FROM notes WHERE id = ?",
                (note_id,),
            ).fetchall()

            return self._hydrate(connection, rows)[0]

    def add_tags(
        self,
        note_id: str,
        tags: list[str],
    ) -> dict[str, Any] | None:
        with self._connection() as connection:
            existing = connection.execute(
                "SELECT id FROM notes WHERE id = ?",
                (note_id,),
            ).fetchone()

            if existing is None:
                return None

            connection.executemany(
                """
                INSERT OR IGNORE INTO note_tags (note_id, tag)
                VALUES (?, ?)
                """,
                [(note_id, tag) for tag in tags],
            )

            connection.execute(
                "UPDATE notes SET updated_at = ? WHERE id = ?",
                (utc_now(), note_id),
            )

            rows = connection.execute(
                "SELECT * FROM notes WHERE id = ?",
                (note_id,),
            ).fetchall()

            return self._hydrate(connection, rows)[0]

    def get_by_tag(
        self,
        tag: str,
        include_archived: bool,
        limit: int,
    ) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT n.*
                FROM notes n
                INNER JOIN note_tags t ON t.note_id = n.id
                WHERE t.tag = ?
                  AND (? = 1 OR n.archived = 0)
                ORDER BY n.updated_at DESC
                LIMIT ?
                """,
                (tag, int(include_archived), limit),
            ).fetchall()

            return self._hydrate(connection, rows)

    def set_archived(
        self,
        note_id: str,
        archived: bool,
    ) -> dict[str, Any] | None:
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE notes
                SET archived = ?, updated_at = ?
                WHERE id = ?
                """,
                (int(archived), utc_now(), note_id),
            )

            if cursor.rowcount == 0:
                return None

            rows = connection.execute(
                "SELECT * FROM notes WHERE id = ?",
                (note_id,),
            ).fetchall()

            return self._hydrate(connection, rows)[0]

    def delete(self, note_id: str) -> bool:
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM notes WHERE id = ?",
                (note_id,),
            )
            return cursor.rowcount > 0
