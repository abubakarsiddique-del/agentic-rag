from __future__ import annotations

import json
import sqlite3
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .models import Conversation, DocumentRecord, IndexedDocument, MemoryTurn, MessageRecord


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_dumps(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_loads(value: str | None) -> Any:
    if value is None or value == "":
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


class AbstractConversationStore(ABC):
    @abstractmethod
    def create_conversation(self, title: str | None = None, *, user_id: str | None = None) -> Conversation:
        raise NotImplementedError

    @abstractmethod
    def list_conversations(self, *, include_empty: bool = False, user_id: str | None = None) -> list[Conversation]:
        raise NotImplementedError

    @abstractmethod
    def get_conversation(self, conversation_id: str) -> Conversation | None:
        raise NotImplementedError

    @abstractmethod
    def append_message(self, conversation_id: str, message: MessageRecord | dict[str, Any]) -> MessageRecord:
        raise NotImplementedError

    @abstractmethod
    def get_messages(self, conversation_id: str) -> list[MessageRecord]:
        raise NotImplementedError

    @abstractmethod
    def rename_conversation(self, conversation_id: str, title: str) -> Conversation | None:
        raise NotImplementedError

    @abstractmethod
    def delete_conversation(self, conversation_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def register_documents(
        self,
        conversation_id: str,
        documents: Iterable[IndexedDocument | dict[str, Any]],
    ) -> list[IndexedDocument]:
        raise NotImplementedError

    @abstractmethod
    def get_documents(self, conversation_id: str) -> list[IndexedDocument]:
        raise NotImplementedError


class SQLiteConversationStore(AbstractConversationStore):
    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            project_root = Path(__file__).resolve().parent.parent
            db_path = project_root / ".rag_history.db"
        self.db_path = str(db_path)
        self._initialize()

        if not Path(self.db_path).exists():
            raise RuntimeError(f"SQLite DB not created at {self.db_path}")

    def __repr__(self) -> str:
        return f"<SQLiteConversationStore path={Path(self.db_path).resolve()}>"

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _initialize(self) -> None:
        conn = self._connect()
        try:
            existing_tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            migration_pending = (
                "schema_migrations" not in existing_tables
                or not conn.execute(
                    "SELECT 1 FROM schema_migrations WHERE version = 8"
                ).fetchone()
            )
            if migration_pending and "users" in existing_tables:
                password_column = next(
                    (
                        row
                        for row in conn.execute("PRAGMA table_info(users)")
                        if row[1] == "password_hash"
                    ),
                    None,
                )
                if password_column is not None and password_column[3]:
                    conn.execute("PRAGMA foreign_keys=OFF")
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversations (
                        id TEXT PRIMARY KEY,
                        title TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        message_count INTEGER NOT NULL DEFAULT 0
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS messages (
                        id TEXT PRIMARY KEY,
                        conversation_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        reasoning TEXT,
                        trace TEXT,
                        sources TEXT,
                        FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS indexed_documents (
                        id TEXT PRIMARY KEY,
                        conversation_id TEXT NOT NULL,
                        document_id TEXT NOT NULL,
                        filename TEXT NOT NULL,
                        source_path TEXT,
                        chunk_count INTEGER NOT NULL DEFAULT 0,
                        metadata TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        UNIQUE (conversation_id, document_id),
                        FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_messages_conversation_created ON messages(conversation_id, created_at)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_indexed_documents_conversation ON indexed_documents(conversation_id, document_id)"
                )
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
                )
                self._ensure_columns(conn, "messages", {
                    "sources_json": "TEXT",
                    "trace_json": "TEXT",
                    "rating": "INTEGER",
                    "feedback": "TEXT",
                    "status": "TEXT NOT NULL DEFAULT 'complete'",
                })
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        id TEXT PRIMARY KEY,
                        conversation_id TEXT NOT NULL,
                        filename TEXT NOT NULL,
                        sha256 TEXT,
                        size_bytes INTEGER NOT NULL DEFAULT 0,
                        pages INTEGER NOT NULL DEFAULT 0,
                        chunks INTEGER NOT NULL DEFAULT 0,
                        status TEXT NOT NULL DEFAULT 'queued'
                            CHECK (status IN ('queued', 'processing', 'ready', 'failed')),
                        error_code TEXT,
                        error_message TEXT,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_documents_conversation ON documents(conversation_id, created_at)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_documents_sha ON documents(conversation_id, sha256)"
                )
                self._migrate_legacy_records(conn)
            conn.execute("PRAGMA foreign_keys=ON")
            foreign_key_errors = conn.execute("PRAGMA foreign_key_check").fetchall()
            if foreign_key_errors:
                raise RuntimeError("SQLite foreign-key validation failed during initialization")
        finally:
            conn.close()

    @staticmethod
    def _ensure_columns(
        conn: sqlite3.Connection,
        table: str,
        columns: dict[str, str],
    ) -> None:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for name, declaration in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")

    @staticmethod
    def _migrate_legacy_records(conn: sqlite3.Connection) -> None:
        if not conn.execute("SELECT 1 FROM schema_migrations WHERE version = 1").fetchone():
            conn.execute(
                "UPDATE messages SET sources_json = sources WHERE sources_json IS NULL AND sources IS NOT NULL"
            )
            conn.execute(
                "UPDATE messages SET trace_json = trace WHERE trace_json IS NULL AND trace IS NOT NULL"
            )

            grouped: dict[tuple[str, str], dict[str, Any]] = {}
            rows = conn.execute(
                "SELECT conversation_id, filename, chunk_count, metadata, created_at FROM indexed_documents"
            ).fetchall()
            for row in rows:
                key = (row["conversation_id"], row["filename"])
                item = grouped.setdefault(key, {
                    "pages": set(),
                    "chunks": 0,
                    "created_at": row["created_at"],
                })
                metadata = _json_loads(row["metadata"]) or {}
                page = metadata.get("page")
                if page is not None:
                    item["pages"].add(str(page))
                item["chunks"] = max(item["chunks"], int(row["chunk_count"] or 0))
                item["created_at"] = min(item["created_at"], row["created_at"])

            for (conversation_id, filename), item in grouped.items():
                document_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"legacy:{conversation_id}:{filename}"))
                conn.execute(
                    """
                    INSERT OR IGNORE INTO documents
                        (id, conversation_id, filename, sha256, size_bytes, pages, chunks, status, created_at)
                    VALUES (?, ?, ?, NULL, 0, ?, ?, 'ready', ?)
                    """,
                    (
                        document_id,
                        conversation_id,
                        filename,
                        len(item["pages"]),
                        item["chunks"],
                        item["created_at"] or _utc_now_iso(),
                    ),
                )

            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (1, ?)",
                (_utc_now_iso(),),
            )

        if not conn.execute("SELECT 1 FROM schema_migrations WHERE version = 2").fetchone():
            for table in ("conversations", "messages", "indexed_documents", "documents"):
                rows = conn.execute(f"SELECT rowid, created_at FROM {table}").fetchall()
                for row in rows:
                    normalized = SQLiteConversationStore._normalize_timestamp(row["created_at"])
                    if normalized != row["created_at"]:
                        conn.execute(f"UPDATE {table} SET created_at = ? WHERE rowid = ?", (normalized, row["rowid"]))
                if table == "conversations":
                    rows = conn.execute("SELECT rowid, updated_at FROM conversations").fetchall()
                    for row in rows:
                        normalized = SQLiteConversationStore._normalize_timestamp(row["updated_at"])
                        if normalized != row["updated_at"]:
                            conn.execute("UPDATE conversations SET updated_at = ? WHERE rowid = ?", (normalized, row["rowid"]))
            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (2, ?)",
                (_utc_now_iso(),),
            )

        if not conn.execute("SELECT 1 FROM schema_migrations WHERE version = 3").fetchone():
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_memory_settings (
                    conversation_id TEXT PRIMARY KEY,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_turns (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    document_names TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_turns_conversation_created ON memory_turns(conversation_id, created_at)"
            )
            conn.execute(
                "INSERT OR IGNORE INTO app_settings(key, value) VALUES ('memory_enabled', '0')"
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO conversation_memory_settings(conversation_id, enabled)
                SELECT id, 1 FROM conversations
                """
            )
            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (3, ?)",
                (_utc_now_iso(),),
            )

        if not conn.execute("SELECT 1 FROM schema_migrations WHERE version = 4").fetchone():
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    password_hash TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_expiry ON sessions(expires_at)")
            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (4, ?)",
                (_utc_now_iso(),),
            )

        if not conn.execute("SELECT 1 FROM schema_migrations WHERE version = 5").fetchone():
            SQLiteConversationStore._ensure_columns(conn, "users", {"is_admin": "INTEGER NOT NULL DEFAULT 0"})
            SQLiteConversationStore._ensure_columns(conn, "conversations", {"user_id": "TEXT REFERENCES users(id) ON DELETE SET NULL"})
            SQLiteConversationStore._ensure_columns(conn, "documents", {"user_id": "TEXT REFERENCES users(id) ON DELETE SET NULL"})
            SQLiteConversationStore._ensure_columns(conn, "indexed_documents", {"user_id": "TEXT REFERENCES users(id) ON DELETE SET NULL"})
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_user_updated ON conversations(user_id, updated_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_user_conversation ON documents(user_id, conversation_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_indexed_documents_user ON indexed_documents(user_id)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_settings (
                    user_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    PRIMARY KEY (user_id, key),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (5, ?)",
                (_utc_now_iso(),),
            )

        if not conn.execute("SELECT 1 FROM schema_migrations WHERE version = 6").fetchone():
            SQLiteConversationStore._ensure_columns(
                conn,
                "memory_turns",
                {"user_id": "TEXT REFERENCES users(id) ON DELETE SET NULL"},
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_turns_user_created ON memory_turns(user_id, created_at)")
            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (6, ?)",
                (_utc_now_iso(),),
            )

        if not conn.execute("SELECT 1 FROM schema_migrations WHERE version = 7").fetchone():
            SQLiteConversationStore._ensure_columns(conn, "sessions", {"csrf_token_hash": "TEXT"})
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS preauth_csrf_tokens (
                    token_hash TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS password_reset_tokens (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_password_reset_user ON password_reset_tokens(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_password_reset_expiry ON password_reset_tokens(expires_at)")
            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (7, ?)",
                (_utc_now_iso(),),
            )

        if not conn.execute("SELECT 1 FROM schema_migrations WHERE version = 8").fetchone():
            columns = {
                row[1]: row
                for row in conn.execute("PRAGMA table_info(users)")
            }
            password_column = columns.get("password_hash")
            if password_column is not None and password_column[3]:
                conn.execute(
                    """
                    CREATE TABLE users_oauth_migration (
                        id TEXT PRIMARY KEY,
                        email TEXT NOT NULL COLLATE NOCASE UNIQUE,
                        password_hash TEXT,
                        created_at TEXT NOT NULL,
                        is_admin INTEGER NOT NULL DEFAULT 0
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO users_oauth_migration(id, email, password_hash, created_at, is_admin)
                    SELECT id, email, password_hash, created_at, is_admin FROM users
                    """
                )
                conn.execute("DROP TABLE users")
                conn.execute("ALTER TABLE users_oauth_migration RENAME TO users")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS oauth_identities (
                    provider TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    email TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (provider, subject),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_oauth_identities_user ON oauth_identities(user_id)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS oauth_states (
                    state_hash TEXT PRIMARY KEY,
                    flow_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_oauth_states_expiry ON oauth_states(expires_at)"
            )
            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (8, ?)",
                (_utc_now_iso(),),
            )

    @staticmethod
    def _normalize_timestamp(value: str | None) -> str:
        if not value:
            return _utc_now_iso()
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return value
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    def create_user(self, email: str, password_hash: str | None, *, is_admin: bool = False) -> dict[str, str] | None:
        user = {
            "id": str(uuid.uuid4()),
            "email": email.strip().casefold(),
            "created_at": _utc_now_iso(),
        }
        user["is_admin"] = bool(is_admin)
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO users(id, email, password_hash, created_at, is_admin) VALUES (?, ?, ?, ?, ?)",
                    (user["id"], user["email"], password_hash, user["created_at"], int(is_admin)),
                )
        except sqlite3.IntegrityError:
            return None
        return user

    def get_user_by_email(self, email: str) -> dict[str, str] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, email, password_hash, created_at, is_admin FROM users WHERE email = ? COLLATE NOCASE",
                (email.strip().casefold(),),
            ).fetchone()
        return dict(row) if row is not None else None

    def user_has_password(self, user_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT password_hash FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
        return row is not None and row["password_hash"] is not None

    def set_password_if_missing(self, user_id: str, password_hash: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE users SET password_hash = ? WHERE id = ? AND password_hash IS NULL",
                (password_hash, user_id),
            )
        return cursor.rowcount == 1

    def resolve_google_user(
        self,
        email: str,
        subject: str,
        *,
        is_admin: bool = False,
    ) -> dict[str, Any] | None:
        normalized_email = email.strip().casefold()
        created_at = _utc_now_iso()
        try:
            with self._connect() as conn:
                linked = conn.execute(
                    """
                    SELECT users.id, users.email, users.password_hash, users.created_at, users.is_admin
                    FROM oauth_identities
                    JOIN users ON users.id = oauth_identities.user_id
                    WHERE oauth_identities.provider = 'google'
                      AND oauth_identities.subject = ?
                    """,
                    (subject,),
                ).fetchone()
                if linked is not None:
                    return dict(linked)

                user = conn.execute(
                    "SELECT id, email, password_hash, created_at, is_admin FROM users WHERE email = ? COLLATE NOCASE",
                    (normalized_email,),
                ).fetchone()
                if user is None:
                    user_id = str(uuid.uuid4())
                    conn.execute(
                        """
                        INSERT INTO users(id, email, password_hash, created_at, is_admin)
                        VALUES (?, ?, NULL, ?, ?)
                        """,
                        (user_id, normalized_email, created_at, int(is_admin)),
                    )
                    user = conn.execute(
                        "SELECT id, email, password_hash, created_at, is_admin FROM users WHERE id = ?",
                        (user_id,),
                    ).fetchone()
                conn.execute(
                    """
                    INSERT INTO oauth_identities(provider, subject, user_id, email, created_at)
                    VALUES ('google', ?, ?, ?, ?)
                    """,
                    (subject, user["id"], normalized_email, created_at),
                )
                return dict(user)
        except sqlite3.IntegrityError:
            linked = self.get_google_user_by_subject(subject)
            if linked is not None:
                return linked
            return None

    def get_google_user_by_subject(self, subject: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT users.id, users.email, users.password_hash, users.created_at, users.is_admin
                FROM oauth_identities
                JOIN users ON users.id = oauth_identities.user_id
                WHERE oauth_identities.provider = 'google'
                  AND oauth_identities.subject = ?
                """,
                (subject,),
            ).fetchone()
        return dict(row) if row is not None else None

    def create_oauth_state(
        self,
        state_hash: str,
        flow_hash: str,
        created_at: str,
        expires_at: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM oauth_states WHERE expires_at <= ?", (created_at,))
            conn.execute(
                "INSERT INTO oauth_states(state_hash, flow_hash, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (state_hash, flow_hash, created_at, expires_at),
            )

    def consume_oauth_state(self, state_hash: str, flow_hash: str, now: str) -> bool:
        with self._connect() as conn:
            conn.execute("DELETE FROM oauth_states WHERE expires_at <= ?", (now,))
            cursor = conn.execute(
                """
                DELETE FROM oauth_states
                WHERE state_hash = ? AND flow_hash = ? AND expires_at > ?
                """,
                (state_hash, flow_hash, now),
            )
        return cursor.rowcount == 1

    def create_session(
        self,
        token_hash: str,
        user_id: str,
        created_at: str,
        expires_at: str,
        csrf_token_hash: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions(token_hash, user_id, created_at, expires_at, csrf_token_hash) VALUES (?, ?, ?, ?, ?)",
                (token_hash, user_id, created_at, expires_at, csrf_token_hash),
            )

    def get_session_details(self, token_hash: str, now: str) -> dict[str, str] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT s.token_hash, s.user_id, s.csrf_token_hash, s.expires_at,
                       u.email, u.created_at, u.is_admin
                FROM sessions s JOIN users u ON u.id = s.user_id
                WHERE s.token_hash = ? AND s.expires_at > ?
                """,
                (token_hash, now),
            ).fetchone()
            conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
        return dict(row) if row is not None else None

    def set_session_csrf_token(self, session_hash: str, csrf_hash: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE sessions SET csrf_token_hash = ? WHERE token_hash = ?",
                (csrf_hash, session_hash),
            )
        return cursor.rowcount == 1

    def add_preauth_csrf_token(self, token_hash: str, created_at: str, expires_at: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM preauth_csrf_tokens WHERE expires_at <= ?", (created_at,))
            conn.execute(
                "INSERT INTO preauth_csrf_tokens(token_hash, created_at, expires_at) VALUES (?, ?, ?)",
                (token_hash, created_at, expires_at),
            )

    def consume_preauth_csrf_token(self, token_hash: str, now: str) -> bool:
        with self._connect() as conn:
            conn.execute("DELETE FROM preauth_csrf_tokens WHERE expires_at <= ?", (now,))
            cursor = conn.execute(
                "DELETE FROM preauth_csrf_tokens WHERE token_hash = ? AND expires_at > ?",
                (token_hash, now),
            )
        return cursor.rowcount == 1

    def create_password_reset_token(
        self,
        token_hash: str,
        user_id: str,
        created_at: str,
        expires_at: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM password_reset_tokens WHERE user_id = ?", (user_id,))
            conn.execute(
                "INSERT INTO password_reset_tokens(token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (token_hash, user_id, created_at, expires_at),
            )

    def reset_password_with_token(self, token_hash: str, password_hash: str, now: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT user_id FROM password_reset_tokens WHERE token_hash = ? AND expires_at > ?",
                (token_hash, now),
            ).fetchone()
            if row is None:
                conn.execute("DELETE FROM password_reset_tokens WHERE expires_at <= ?", (now,))
                return None
            user_id = str(row["user_id"])
            updated = conn.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (password_hash, user_id),
            ).rowcount
            if updated != 1:
                return None
            conn.execute("DELETE FROM password_reset_tokens WHERE token_hash = ?", (token_hash,))
            conn.execute("DELETE FROM password_reset_tokens WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        return user_id

    def delete_expired_sessions(self, now: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
        return cursor.rowcount

    def delete_expired_auth_tokens(self, now: str) -> dict[str, int]:
        with self._connect() as conn:
            sessions = conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,)).rowcount
            reset_tokens = conn.execute(
                "DELETE FROM password_reset_tokens WHERE expires_at <= ?",
                (now,),
            ).rowcount
            csrf_tokens = conn.execute(
                "DELETE FROM preauth_csrf_tokens WHERE expires_at <= ?",
                (now,),
            ).rowcount
        return {"sessions": sessions, "password_reset_tokens": reset_tokens, "preauth_csrf_tokens": csrf_tokens}

    def get_user_for_session(self, token_hash: str, now: str) -> dict[str, str] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT u.id, u.email, u.created_at, u.is_admin
                FROM sessions s JOIN users u ON u.id = s.user_id
                WHERE s.token_hash = ? AND s.expires_at > ?
                """,
                (token_hash, now),
            ).fetchone()
            conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
        return dict(row) if row is not None else None

    def reassign_legacy_data(self, user_id: str) -> dict[str, int]:
        with self._connect() as conn:
            conversations = conn.execute(
                "UPDATE conversations SET user_id = ? WHERE user_id IS NULL",
                (user_id,),
            ).rowcount
            documents = conn.execute(
                "UPDATE documents SET user_id = ? WHERE user_id IS NULL AND conversation_id IN "
                "(SELECT id FROM conversations WHERE user_id = ?)",
                (user_id, user_id),
            ).rowcount
            indexed_documents = conn.execute(
                "UPDATE indexed_documents SET user_id = ? WHERE user_id IS NULL AND conversation_id IN "
                "(SELECT id FROM conversations WHERE user_id = ?)",
                (user_id, user_id),
            ).rowcount
            memory_turns = conn.execute(
                "UPDATE memory_turns SET user_id = ? WHERE user_id IS NULL AND conversation_id IN "
                "(SELECT id FROM conversations WHERE user_id = ?)",
                (user_id, user_id),
            ).rowcount
            conn.execute(
                """
                INSERT OR IGNORE INTO user_settings(user_id, key, value)
                SELECT ?, 'memory_enabled', value FROM app_settings WHERE key = 'memory_enabled'
                """,
                (user_id,),
            )
        return {
            "conversations": conversations,
            "documents": documents,
            "indexed_documents": indexed_documents,
            "memory_turns": memory_turns,
        }

    def delete_session(self, token_hash: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))
        return cursor.rowcount > 0

    @staticmethod
    def _ensure_message_record(value: MessageRecord | dict[str, Any]) -> MessageRecord:
        if isinstance(value, MessageRecord):
            return value

        if not isinstance(value, dict):
            raise TypeError("messages must be MessageRecord or dict")

        return MessageRecord(
            id=str(value.get("id") or uuid.uuid4().hex),
            conversation_id=str(value["conversation_id"]),
            role=str(value.get("role", "user")),
            content=str(value.get("content", "")),
            created_at=str(value.get("created_at") or _utc_now_iso()),
            reasoning=value.get("reasoning"),
            trace=value.get("trace"),
            sources=value.get("sources"),
            rating=value.get("rating"),
            feedback=value.get("feedback"),
            status=value.get("status", "complete"),
        )

    @staticmethod
    def _ensure_indexed_document(value: IndexedDocument | dict[str, Any]) -> IndexedDocument:
        if isinstance(value, IndexedDocument):
            return value

        if not isinstance(value, dict):
            raise TypeError("documents must be IndexedDocument or dict")

        return IndexedDocument(
            id=str(value.get("id") or uuid.uuid4().hex),
            conversation_id=str(value["conversation_id"]),
            document_id=str(value.get("document_id") or value.get("id") or uuid.uuid4().hex),
            filename=str(value.get("filename") or "unknown_file"),
            source_path=value.get("source_path"),
            chunk_count=int(value.get("chunk_count", 0) or 0),
            metadata=dict(value.get("metadata") or {}),
            created_at=str(value.get("created_at") or _utc_now_iso()),
            user_id=value.get("user_id"),
        )

    def create_conversation(self, title: str | None = None, *, user_id: str | None = None) -> Conversation:
        conversation_id = str(uuid.uuid4())
        now = _utc_now_iso()
        safe_title = (title or "New conversation").strip() or "New conversation"
        conv = Conversation(
            id=conversation_id,
            title=safe_title,
            created_at=now,
            updated_at=now,
            message_count=0,
            user_id=user_id,
        )

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at, message_count, user_id) VALUES (?, ?, ?, ?, ?, ?)",
                (conv.id, conv.title, conv.created_at, conv.updated_at, conv.message_count, conv.user_id),
            )

        return conv

    def list_conversations(
        self,
        *,
        include_empty: bool = False,
        user_id: str | None = None,
    ) -> list[Conversation]:
        filters = []
        parameters: list[Any] = []
        if user_id is not None:
            filters.append("user_id = ?")
            parameters.append(user_id)
        if not include_empty:
            filters.append("(message_count > 0 OR document_count > 0)")
        empty_filter = f"WHERE {' AND '.join(filters)}" if filters else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, title, created_at, updated_at, message_count, document_count, user_id
                FROM (
                    SELECT c.id, c.title, c.created_at, c.updated_at, c.user_id,
                        (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count,
                        (SELECT COUNT(*) FROM documents d WHERE d.conversation_id = c.id) AS document_count
                    FROM conversations c
                )
                {empty_filter}
                ORDER BY updated_at DESC, created_at DESC
                """,
                parameters,
            ).fetchall()

        return [
            Conversation(
                id=row["id"],
                title=row["title"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                message_count=int(row["message_count"] or 0),
                document_count=int(row["document_count"] or 0),
                user_id=row["user_id"],
            )
            for row in rows
        ]

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT c.id, c.title, c.created_at, c.updated_at, c.user_id,
                    (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count,
                    (SELECT COUNT(*) FROM documents d WHERE d.conversation_id = c.id) AS document_count
                FROM conversations c WHERE c.id = ?
                """,
                (conversation_id,),
            ).fetchone()

        if row is None:
            return None

        return Conversation(
            id=row["id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            message_count=int(row["message_count"] or 0),
            document_count=int(row["document_count"] or 0),
            user_id=row["user_id"],
        )

    def append_message(self, conversation_id: str, message: MessageRecord | dict[str, Any]) -> MessageRecord:
        if self.get_conversation(conversation_id) is None:
            raise KeyError(f"Conversation not found: {conversation_id}")

        record = self._ensure_message_record(message)
        if record.conversation_id and record.conversation_id != conversation_id:
            record = MessageRecord(
                id=record.id,
                conversation_id=conversation_id,
                role=record.role,
                content=record.content,
                created_at=record.created_at,
                reasoning=record.reasoning,
                trace=record.trace,
                sources=record.sources,
                rating=record.rating,
                feedback=record.feedback,
                status=record.status,
            )

        now = _utc_now_iso()
        created_at = record.created_at or now
        payload = MessageRecord(
            id=record.id or str(uuid.uuid4()),
            conversation_id=conversation_id,
            role=record.role,
            content=record.content,
            created_at=created_at,
            reasoning=record.reasoning,
            trace=record.trace,
            sources=record.sources,
            rating=record.rating,
            feedback=record.feedback,
            status=record.status,
        )

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO messages (id, conversation_id, role, content, created_at, reasoning, trace, sources, sources_json, trace_json, rating, feedback, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    conversation_id=excluded.conversation_id,
                    role=excluded.role,
                    content=excluded.content,
                    created_at=excluded.created_at,
                    reasoning=excluded.reasoning,
                    trace=excluded.trace,
                    sources=excluded.sources,
                    sources_json=excluded.sources_json,
                    trace_json=excluded.trace_json,
                    rating=excluded.rating,
                    feedback=excluded.feedback,
                    status=excluded.status
                """,
                (
                    payload.id,
                    payload.conversation_id,
                    payload.role,
                    payload.content,
                    payload.created_at,
                    _json_dumps(payload.reasoning),
                    _json_dumps(payload.trace),
                    _json_dumps(payload.sources),
                    _json_dumps(payload.sources),
                    _json_dumps(payload.trace),
                    payload.rating,
                    payload.feedback,
                    payload.status,
                ),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ?, message_count = (SELECT COUNT(*) FROM messages WHERE conversation_id = ?) WHERE id = ?",
                (now, conversation_id, conversation_id),
            )
            if payload.role == "user":
                conn.execute(
                    """
                    UPDATE conversations SET title = ?
                    WHERE id = ? AND (title IS NULL OR title = '' OR title = 'New conversation')
                        AND (SELECT COUNT(*) FROM messages WHERE conversation_id = ?) = 1
                    """,
                    (self._derive_title(payload.content), conversation_id, conversation_id),
                )

        return payload

    def get_messages(self, conversation_id: str) -> list[MessageRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, conversation_id, role, content, created_at, reasoning,
                    COALESCE(trace_json, trace) AS trace_value,
                    COALESCE(sources_json, sources) AS sources_value, rating, feedback, status
                FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC, rowid ASC
                """,
                (conversation_id,),
            ).fetchall()

        items: list[MessageRecord] = []
        for row in rows:
            items.append(
                MessageRecord(
                    id=row["id"],
                    conversation_id=row["conversation_id"],
                    role=row["role"],
                    content=row["content"],
                    created_at=row["created_at"],
                    reasoning=_json_loads(row["reasoning"]),
                    trace=_json_loads(row["trace_value"]),
                    sources=_json_loads(row["sources_value"]),
                    rating=row["rating"],
                    feedback=row["feedback"],
                    status=row["status"] or "complete",
                )
            )
        return items

    def set_message_rating(self, conversation_id: str, message_id: str, rating: int, feedback: str | None = None) -> MessageRecord:
        if self.get_conversation(conversation_id) is None:
            raise KeyError(f"Conversation not found: {conversation_id}")

        normalized = int(rating)
        if normalized not in (-1, 0, 1):
            raise ValueError("rating must be one of -1, 0, or 1")

        existing = next((item for item in self.get_messages(conversation_id) if item.id == message_id), None)
        if existing is None:
            raise KeyError(f"Message not found: {message_id}")

        updated = MessageRecord(
            id=existing.id,
            conversation_id=existing.conversation_id,
            role=existing.role,
            content=existing.content,
            created_at=existing.created_at,
            reasoning=existing.reasoning,
            trace=existing.trace,
            sources=existing.sources,
            rating=normalized,
            feedback=(feedback if feedback is not None else existing.feedback),
            status=existing.status,
        )

        with self._connect() as conn:
            conn.execute(
                "UPDATE messages SET rating = ?, feedback = ? WHERE id = ? AND conversation_id = ?",
                (normalized, updated.feedback, message_id, conversation_id),
            )

        return updated

    def rename_conversation(self, conversation_id: str, title: str) -> Conversation | None:
        safe_title = (title or "New conversation").strip() or "New conversation"
        with self._connect() as conn:
            row = conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ? RETURNING id, title, created_at, updated_at, message_count",
                (safe_title, _utc_now_iso(), conversation_id),
            ).fetchone()

        if row is None:
            return None

        return self.get_conversation(conversation_id)

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            conn.execute("DELETE FROM indexed_documents WHERE conversation_id = ?", (conversation_id,))
            cursor = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            return cursor.rowcount > 0

    def get_global_memory_enabled(self, user_id: str | None = None) -> bool:
        with self._connect() as conn:
            if user_id is None:
                row = conn.execute(
                    "SELECT value FROM app_settings WHERE key = 'memory_enabled'"
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT value FROM user_settings WHERE user_id = ? AND key = 'memory_enabled'",
                    (user_id,),
                ).fetchone()
        return bool(row and row["value"] == "1")

    def set_global_memory_enabled(self, enabled: bool, user_id: str | None = None) -> bool:
        value = "1" if enabled else "0"
        with self._connect() as conn:
            if user_id is None:
                conn.execute(
                    "INSERT INTO app_settings(key, value) VALUES ('memory_enabled', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (value,),
                )
            else:
                conn.execute(
                    "INSERT INTO user_settings(user_id, key, value) VALUES (?, 'memory_enabled', ?) "
                    "ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value",
                    (user_id, value),
                )
        return enabled

    def list_conversation_ids_for_owner(self, user_id: str) -> set[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT id FROM conversations WHERE user_id = ?", (user_id,)).fetchall()
        return {str(row["id"]) for row in rows}

    def get_conversation_memory_enabled(self, conversation_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT enabled FROM conversation_memory_settings WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return True if row is None else bool(row["enabled"])

    def set_conversation_memory_enabled(self, conversation_id: str, enabled: bool) -> bool:
        if self.get_conversation(conversation_id) is None:
            raise KeyError(f"Conversation not found: {conversation_id}")
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversation_memory_settings(conversation_id, enabled) VALUES (?, ?) "
                "ON CONFLICT(conversation_id) DO UPDATE SET enabled = excluded.enabled",
                (conversation_id, int(enabled)),
            )
        return enabled

    def append_memory_turn(self, turn: MemoryTurn) -> MemoryTurn:
        conversation = self.get_conversation(turn.conversation_id)
        if conversation is None:
            raise KeyError(f"Conversation not found: {turn.conversation_id}")
        if turn.user_id is None:
            turn.user_id = conversation.user_id
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_turns(id, conversation_id, question, summary, document_names, created_at, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    conversation_id=excluded.conversation_id,
                    question=excluded.question,
                    summary=excluded.summary,
                    document_names=excluded.document_names,
                    created_at=excluded.created_at,
                    user_id=excluded.user_id
                """,
                (
                    turn.id,
                    turn.conversation_id,
                    turn.question,
                    turn.summary,
                    _json_dumps(turn.document_names) or "[]",
                    turn.created_at,
                    turn.user_id,
                ),
            )
        return turn

    def list_memory_turns(
        self,
        conversation_id: str | None = None,
        *,
        limit: int = 100,
        owner_id: str | None = None,
    ) -> list[MemoryTurn]:
        filters = []
        parameters: list[Any] = []
        if conversation_id is not None:
            filters.append("conversation_id = ?")
            parameters.append(conversation_id)
        if owner_id is not None:
            filters.append("user_id = ?")
            parameters.append(owner_id)
        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        parameters.append(max(1, limit))
        query = f"SELECT * FROM memory_turns {where_clause} ORDER BY created_at DESC LIMIT ?"
        with self._connect() as conn:
            rows = conn.execute(query, parameters).fetchall()
        return [
            MemoryTurn(
                id=row["id"],
                conversation_id=row["conversation_id"],
                question=row["question"],
                summary=row["summary"],
                document_names=_json_loads(row["document_names"]) or [],
                created_at=row["created_at"],
                user_id=row["user_id"],
            )
            for row in rows
        ]

    def delete_memory_turns(
        self,
        conversation_id: str | None = None,
        *,
        owner_id: str | None = None,
    ) -> list[str]:
        with self._connect() as conn:
            if conversation_id is not None:
                ids = [
                    row["id"]
                    for row in conn.execute(
                        "SELECT id FROM memory_turns WHERE conversation_id = ?",
                        (conversation_id,),
                    ).fetchall()
                ]
                conn.execute("DELETE FROM memory_turns WHERE conversation_id = ?", (conversation_id,))
            elif owner_id is not None:
                ids = [
                    row["id"]
                    for row in conn.execute(
                        "SELECT id FROM memory_turns WHERE user_id = ?",
                        (owner_id,),
                    ).fetchall()
                ]
                conn.execute("DELETE FROM memory_turns WHERE user_id = ?", (owner_id,))
            else:
                ids = [row["id"] for row in conn.execute("SELECT id FROM memory_turns").fetchall()]
                conn.execute("DELETE FROM memory_turns")
        return ids

    @staticmethod
    def _derive_title(content: str, maximum: int = 50) -> str:
        normalized = " ".join((content or "").split())
        if len(normalized) <= maximum:
            return normalized or "New conversation"
        prefix = normalized[:maximum + 1]
        boundary = prefix.rfind(" ")
        return (prefix[:boundary] if boundary > 0 else prefix[:maximum]).rstrip()

    def create_document(self, document: DocumentRecord) -> DocumentRecord:
        if document.user_id is None:
            conversation = self.get_conversation(document.conversation_id)
            document.user_id = conversation.user_id if conversation else None
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents
                    (id, conversation_id, filename, sha256, size_bytes, pages, chunks, status,
                     error_code, error_message, created_at, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document.id, document.conversation_id, document.filename, document.sha256,
                    document.size_bytes, document.pages, document.chunks, document.status,
                    document.error_code, document.error_message, document.created_at, document.user_id,
                ),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (_utc_now_iso(), document.conversation_id),
            )
        return document

    def update_document(self, document_id: str, **changes: Any) -> DocumentRecord | None:
        allowed = {"filename", "sha256", "size_bytes", "pages", "chunks", "status", "error_code", "error_message"}
        updates = {key: value for key, value in changes.items() if key in allowed}
        if not updates:
            return self.get_document(document_id)
        assignments = ", ".join(f"{key} = ?" for key in updates)
        with self._connect() as conn:
            row = conn.execute(
                f"UPDATE documents SET {assignments} WHERE id = ? RETURNING conversation_id",
                (*updates.values(), document_id),
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (_utc_now_iso(), row["conversation_id"]),
            )
        return self.get_document(document_id)

    def get_document(self, document_id: str) -> DocumentRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        return self._row_to_document(row) if row is not None else None

    def find_document_by_sha(self, conversation_id: str, sha256: str) -> DocumentRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE conversation_id = ? AND sha256 = ? LIMIT 1",
                (conversation_id, sha256),
            ).fetchone()
        return self._row_to_document(row) if row is not None else None

    def delete_document(self, conversation_id: str, document_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM documents WHERE conversation_id = ? AND id = ?",
                (conversation_id, document_id),
            )
            if cursor.rowcount:
                conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    (_utc_now_iso(), conversation_id),
                )
            return cursor.rowcount > 0

    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            id=row["id"], conversation_id=row["conversation_id"], filename=row["filename"],
            sha256=row["sha256"], size_bytes=int(row["size_bytes"] or 0),
            pages=int(row["pages"] or 0), chunks=int(row["chunks"] or 0), status=row["status"],
            error_code=row["error_code"], error_message=row["error_message"], created_at=row["created_at"],
            user_id=row["user_id"],
        )

    def list_document_records(self, conversation_id: str) -> list[DocumentRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE conversation_id = ? ORDER BY created_at, id",
                (conversation_id,),
            ).fetchall()
        return [self._row_to_document(row) for row in rows]

    def register_documents(
        self,
        conversation_id: str,
        documents: Iterable[IndexedDocument | dict[str, Any]],
    ) -> list[IndexedDocument]:
        if self.get_conversation(conversation_id) is None:
            raise KeyError(f"Conversation not found: {conversation_id}")

        items: list[IndexedDocument] = []
        for value in documents:
            doc = self._ensure_indexed_document(value)
            if doc.conversation_id and doc.conversation_id != conversation_id:
                doc = IndexedDocument(
                    id=doc.id,
                    conversation_id=conversation_id,
                    document_id=doc.document_id,
                    filename=doc.filename,
                    source_path=doc.source_path,
                    chunk_count=doc.chunk_count,
                    metadata=doc.metadata,
                    created_at=doc.created_at or _utc_now_iso(),
                    user_id=doc.user_id,
                )

            if doc.user_id is None:
                conversation = self.get_conversation(conversation_id)
                doc.user_id = conversation.user_id if conversation else None

            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO indexed_documents (
                        id, conversation_id, document_id, filename, source_path, chunk_count, metadata, created_at, user_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(conversation_id, document_id) DO UPDATE SET
                        filename=excluded.filename,
                        source_path=excluded.source_path,
                        chunk_count=excluded.chunk_count,
                        metadata=excluded.metadata,
                        created_at=excluded.created_at,
                        user_id=excluded.user_id
                    """,
                    (
                        doc.id,
                        doc.conversation_id,
                        doc.document_id,
                        doc.filename,
                        doc.source_path,
                        doc.chunk_count,
                        _json_dumps(doc.metadata),
                        doc.created_at or _utc_now_iso(),
                        doc.user_id,
                    ),
                )

            items.append(doc)

        return items

    def get_documents(self, conversation_id: str) -> list[IndexedDocument]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, conversation_id, document_id, filename, source_path, chunk_count, metadata, created_at, user_id
                FROM indexed_documents
                WHERE conversation_id = ?
                ORDER BY created_at ASC, filename ASC
                """,
                (conversation_id,),
            ).fetchall()

        documents: list[IndexedDocument] = []
        for row in rows:
            documents.append(
                IndexedDocument(
                    id=row["id"],
                    conversation_id=row["conversation_id"],
                    document_id=row["document_id"],
                    filename=row["filename"],
                    source_path=row["source_path"],
                    chunk_count=int(row["chunk_count"]),
                    metadata=_json_loads(row["metadata"]) or {},
                    created_at=row["created_at"],
                    user_id=row["user_id"],
                )
            )
        return documents


__all__ = [
    "AbstractConversationStore",
    "SQLiteConversationStore",
]
