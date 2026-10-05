import asyncio
import re
import sqlite3
from datetime import datetime, timedelta
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

import backend.app as backend_module
from backend.app import app
from backend.auth import (
    SESSION_COOKIE_NAME,
    SignInRateLimiter,
    hash_password,
    hash_session_token,
)
from persistence.store import SQLiteConversationStore


@pytest.fixture
def auth_store(tmp_path, monkeypatch):
    store = SQLiteConversationStore(db_path=tmp_path / "auth-test.db")
    monkeypatch.setattr(backend_module, "_get_store", lambda: store)
    monkeypatch.setattr(backend_module, "signin_rate_limiter", SignInRateLimiter())
    monkeypatch.setattr(backend_module, "reset_request_limiter", SignInRateLimiter(limit=3, window_seconds=3600))
    return store


@pytest.fixture
def https_client(auth_store):
    with TestClient(app, base_url="https://testserver") as client:
        yield client


def test_signup_me_signin_logout_cookie_and_hashed_session(https_client, auth_store):
    signup = https_client.post(
        "/api/auth/signup",
        json={"email": "  User@Example.com ", "password": "a-long-test-password"},
    )
    assert signup.status_code == 201
    user = signup.json()
    assert user["email"] == "user@example.com"
    assert "password_hash" not in user

    cookie_header = signup.headers["set-cookie"].lower()
    assert "httponly" in cookie_header
    assert ("secure" in cookie_header) is backend_module.COOKIE_SECURE
    assert "samesite=lax" in cookie_header
    assert "path=/" in cookie_header
    assert "max-age=604800" in cookie_header
    assert "domain=" not in cookie_header

    token = https_client.cookies.get(SESSION_COOKIE_NAME)
    assert https_client.get("/api/auth/me").json() == user
    with sqlite3.connect(auth_store.db_path) as connection:
        stored_hash = connection.execute("SELECT password_hash FROM users").fetchone()[0]
        stored_token_hash = connection.execute("SELECT token_hash FROM sessions").fetchone()[0]
        session_expiry = connection.execute("SELECT expires_at FROM sessions").fetchone()[0]
    assert stored_hash != "a-long-test-password"
    assert stored_token_hash == hash_session_token(token)
    assert token not in stored_token_hash
    with sqlite3.connect(auth_store.db_path) as connection:
        assert connection.execute(
            "SELECT expires_at FROM sessions WHERE token_hash = ?",
            (stored_token_hash,),
        ).fetchone()[0] == session_expiry

    signin = https_client.post(
        "/api/auth/signin",
        json={"email": "USER@example.com", "password": "a-long-test-password"},
    )
    assert signin.status_code == 200
    assert signin.json() == user
    csrf = https_client.get("/api/auth/csrf").json()["csrf_token"]
    logout = https_client.post("/api/auth/logout", headers={"X-CSRF-Token": csrf})
    assert logout.json() == {"logged_out": True}
    assert https_client.get("/api/auth/me").status_code == 401


def test_signup_validates_email_password_and_duplicate_email(https_client):
    assert https_client.post(
        "/api/auth/signup",
        json={"email": "not-an-email", "password": "a-long-test-password"},
    ).status_code == 422
    assert https_client.post(
        "/api/auth/signup",
        json={"email": "a@example.com", "password": "short"},
    ).status_code == 422
    valid = {"email": "a@example.com", "password": "a-long-test-password"}
    assert https_client.post("/api/auth/signup", json=valid).status_code == 201
    assert https_client.post("/api/auth/signup", json=valid).status_code == 409


def test_signin_failures_are_generic_and_throttled(https_client, auth_store):
    auth_store.create_user("person@example.com", hash_password("a-long-test-password"))
    bad_password = https_client.post(
        "/api/auth/signin",
        json={"email": "person@example.com", "password": "incorrect-password"},
    )
    unknown_user = https_client.post(
        "/api/auth/signin",
        json={"email": "missing@example.com", "password": "incorrect-password"},
    )
    assert bad_password.status_code == unknown_user.status_code == 401
    assert bad_password.json()["detail"] == unknown_user.json()["detail"]

    limiter = backend_module.signin_rate_limiter
    key = ("person@example.com", "testclient")
    for _ in range(limiter.limit - 1):
        response = https_client.post(
            "/api/auth/signin",
            json={"email": "person@example.com", "password": "incorrect-password"},
        )
        assert response.status_code == 401
    limited = https_client.post(
        "/api/auth/signin",
        json={"email": "person@example.com", "password": "incorrect-password"},
    )
    assert limited.status_code == 429
    assert limiter.is_limited(key)


def test_expired_session_is_rejected_and_removed(https_client, auth_store):
    user = auth_store.create_user("expired@example.com", hash_password("a-long-test-password"))
    raw_token = "expired-session-token"
    auth_store.create_session(
        hash_session_token(raw_token),
        user["id"],
        "2020-01-01T00:00:00Z",
        "2020-01-02T00:00:00Z",
    )
    response = https_client.get(
        "/api/auth/me",
        headers={"cookie": f"{SESSION_COOKIE_NAME}={raw_token}"},
    )

    assert response.status_code == 401
    with sqlite3.connect(auth_store.db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0


def test_signup_never_claims_ownerless_legacy_conversations(https_client, auth_store, monkeypatch):
    monkeypatch.setenv("RAG_LEGACY_OWNER_EMAIL", "designated-owner@example.com")
    legacy_conversation = auth_store.create_conversation("Legacy chat")

    response = https_client.post(
        "/api/auth/signup",
        json={"email": "ordinary-user@example.com", "password": "a-long-test-password"},
    )

    assert response.status_code == 201
    assert auth_store.get_conversation(legacy_conversation.id).user_id is None


def test_auth_migration_preserves_existing_conversations(tmp_path):
    database = tmp_path / "legacy.db"
    store = SQLiteConversationStore(db_path=database)
    conversation = store.create_conversation("Existing chat")

    migrated = SQLiteConversationStore(db_path=database)

    assert migrated.get_conversation(conversation.id).title == "Existing chat"
    with migrated._connect() as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert {"users", "sessions"} <= tables


def test_oauth_migration_makes_password_nullable_and_preserves_users(tmp_path):
    database = tmp_path / "nullable-password.db"
    store = SQLiteConversationStore(db_path=database)
    user = store.create_user("legacy@example.com", hash_password("legacy-password"))
    with store._connect() as connection:
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute(
            """
            CREATE TABLE users_legacy_test (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL COLLATE NOCASE UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                is_admin INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        connection.execute(
            """
            INSERT INTO users_legacy_test(id, email, password_hash, created_at, is_admin)
            SELECT id, email, password_hash, created_at, is_admin FROM users
            """
        )
        connection.execute("DROP TABLE users")
        connection.execute("ALTER TABLE users_legacy_test RENAME TO users")
        connection.execute("DELETE FROM schema_migrations WHERE version = 8")

    migrated = SQLiteConversationStore(db_path=database)

    assert migrated.get_user_by_email("legacy@example.com")["id"] == user["id"]
    assert migrated.get_user_by_email("legacy@example.com")["password_hash"] is not None
    with migrated._connect() as connection:
        password_column = next(row for row in connection.execute("PRAGMA table_info(users)") if row[1] == "password_hash")
        assert password_column[3] == 0
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


async def _fake_google_authorization_url(state):
    return f"https://accounts.google.com/o/oauth2/v2/auth?state={state}"


def test_google_authorization_uses_code_flow_and_required_scopes(monkeypatch, caplog):
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "test-client-secret")
    monkeypatch.setenv("GOOGLE_REDIRECT_URI", "https://app.example.com/api/auth/google/callback")

    with caplog.at_level("INFO", logger=backend_module.logger.name):
        authorization_url = asyncio.run(backend_module._google_authorization_url("random-state"))
    query = parse_qs(urlparse(authorization_url).query)

    assert urlparse(authorization_url).scheme == "https"
    assert query["client_id"] == ["test-client-id"]
    assert query["redirect_uri"] == ["https://app.example.com/api/auth/google/callback"]
    assert query["response_type"] == ["code"]
    assert set(query["scope"][0].split()) == {"openid", "email", "profile"}
    assert query["state"] == ["random-state"]
    assert "Google OAuth outgoing redirect_uri=https://app.example.com/api/auth/google/callback" in caplog.text


def _begin_google_flow(client, monkeypatch):
    monkeypatch.setattr(backend_module, "_google_authorization_url", _fake_google_authorization_url)
    response = client.get("/api/auth/google/login", follow_redirects=False)
    state = parse_qs(urlparse(response.headers["location"]).query)["state"][0]
    return response, state


def test_google_login_creates_verified_google_user_and_opaque_session(
    https_client,
    auth_store,
    monkeypatch,
):
    async def userinfo(code):
        assert code == "google-code"
        return {
            "sub": "google-sub-123",
            "email": "New.User@example.com",
            "email_verified": True,
            "name": "New User",
        }

    monkeypatch.setattr(backend_module, "_fetch_google_userinfo", userinfo)
    login, state = _begin_google_flow(https_client, monkeypatch)
    assert login.status_code == 302
    flow_cookie = login.headers["set-cookie"].lower()
    assert backend_module.GOOGLE_FLOW_COOKIE_NAME in flow_cookie
    assert "httponly" in flow_cookie
    assert "samesite=lax" in flow_cookie

    callback = https_client.get(
        "/api/auth/google/callback",
        params={"state": state, "code": "google-code"},
        follow_redirects=False,
    )

    assert callback.status_code == 303
    assert callback.headers["location"] == f"{backend_module.FRONTEND_ORIGINS[0]}/app"
    assert any(
        header.startswith(f"{SESSION_COOKIE_NAME}=")
        for header in callback.headers.get_list("set-cookie")
    )
    current_user = https_client.get("/api/auth/me")
    assert current_user.status_code == 200
    assert current_user.json()["email"] == "new.user@example.com"
    user = auth_store.get_user_by_email("new.user@example.com")
    assert user["password_hash"] is None
    assert auth_store.get_google_user_by_subject("google-sub-123")["id"] == user["id"]
    with sqlite3.connect(auth_store.db_path) as connection:
        token = https_client.cookies.get(SESSION_COOKIE_NAME)
        assert connection.execute(
            "SELECT token_hash FROM sessions WHERE user_id = ?",
            (user["id"],),
        ).fetchone()[0] == hash_session_token(token)
    csrf_token = https_client.get("/api/auth/csrf").json()["csrf_token"]
    assert https_client.post(
        "/api/auth/logout",
        headers={"X-CSRF-Token": csrf_token},
    ).status_code == 200
    assert https_client.get("/api/auth/me").status_code == 401


def test_google_callback_rejects_state_mismatch_before_exchange(
    https_client,
    auth_store,
    monkeypatch,
):
    exchange_calls = []

    async def userinfo(code):
        exchange_calls.append(code)
        return {"sub": "unused", "email": "unused@example.com", "email_verified": True}

    monkeypatch.setattr(backend_module, "_fetch_google_userinfo", userinfo)
    _, state = _begin_google_flow(https_client, monkeypatch)

    response = https_client.get(
        "/api/auth/google/callback",
        params={"state": f"{state}-mismatch", "code": "google-code"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == (
        f"{backend_module.FRONTEND_ORIGINS[0]}/login?auth_error=google_signin_failed"
    )
    assert exchange_calls == []
    assert auth_store.get_user_by_email("unused@example.com") is None
    assert https_client.get("/api/auth/me").status_code == 401


def test_google_callback_requires_a_prior_state_and_flow_cookie(
    https_client,
    auth_store,
    monkeypatch,
):
    exchange_calls = []

    async def userinfo(code):
        exchange_calls.append(code)
        return {"sub": "not-used", "email": "bypass@example.com", "email_verified": True}

    monkeypatch.setattr(backend_module, "_fetch_google_userinfo", userinfo)

    direct_callback = https_client.get(
        "/api/auth/google/callback",
        params={"state": "invented-state", "code": "google-code"},
        follow_redirects=False,
    )

    assert direct_callback.status_code == 303
    assert direct_callback.headers["location"].endswith(
        "/login?auth_error=google_signin_failed"
    )
    assert exchange_calls == []
    assert auth_store.get_user_by_email("bypass@example.com") is None
    assert https_client.get("/api/auth/me").status_code == 401

    _, valid_state = _begin_google_flow(https_client, monkeypatch)
    client_without_flow_cookie = TestClient(app, base_url="https://testserver")
    callback_without_cookie = client_without_flow_cookie.get(
        "/api/auth/google/callback",
        params={"state": valid_state, "code": "google-code"},
        follow_redirects=False,
    )

    assert callback_without_cookie.status_code == 303
    assert exchange_calls == []
    assert auth_store.get_user_by_email("bypass@example.com") is None


def test_google_oauth_state_is_random_hashed_and_short_lived(
    https_client,
    auth_store,
    monkeypatch,
):
    monkeypatch.setattr(backend_module, "_google_authorization_url", _fake_google_authorization_url)
    first = https_client.get("/api/auth/google/login", follow_redirects=False)
    first_state = parse_qs(urlparse(first.headers["location"]).query)["state"][0]
    first_flow = https_client.cookies.get(backend_module.GOOGLE_FLOW_COOKIE_NAME)
    second = https_client.get("/api/auth/google/login", follow_redirects=False)
    second_state = parse_qs(urlparse(second.headers["location"]).query)["state"][0]
    second_flow = https_client.cookies.get(backend_module.GOOGLE_FLOW_COOKIE_NAME)

    assert len(first_state) >= 43
    assert len(second_state) >= 43
    assert first_state != second_state
    assert first_flow != second_flow
    with sqlite3.connect(auth_store.db_path) as connection:
        stored = connection.execute(
            "SELECT state_hash, flow_hash, created_at, expires_at FROM oauth_states"
        ).fetchall()
    assert len(stored) == 2
    assert all(row[0] not in {first_state, second_state} for row in stored)
    assert all(row[1] not in {first_flow, second_flow} for row in stored)
    assert all(
        datetime.fromisoformat(row[3].replace("Z", "+00:00"))
        - datetime.fromisoformat(row[2].replace("Z", "+00:00"))
        == timedelta(seconds=backend_module.GOOGLE_OAUTH_STATE_TTL_SECONDS)
        for row in stored
    )


def test_google_callback_rejects_expired_state_before_exchange(
    https_client,
    auth_store,
    monkeypatch,
):
    exchange_calls = []

    async def userinfo(code):
        exchange_calls.append(code)
        return {"sub": "expired", "email": "expired@example.com", "email_verified": True}

    monkeypatch.setattr(backend_module, "_fetch_google_userinfo", userinfo)
    _, state = _begin_google_flow(https_client, monkeypatch)
    with sqlite3.connect(auth_store.db_path) as connection:
        connection.execute(
            "UPDATE oauth_states SET expires_at = ? WHERE state_hash = ?",
            ("2000-01-01T00:00:00Z", hash_session_token(state)),
        )

    response = https_client.get(
        "/api/auth/google/callback",
        params={"state": state, "code": "google-code"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert exchange_calls == []
    assert auth_store.get_user_by_email("expired@example.com") is None


def test_google_verified_email_auto_links_existing_password_user(
    https_client,
    auth_store,
    monkeypatch,
):
    existing = auth_store.create_user(
        "same@example.com",
        hash_password("existing-password"),
    )

    async def userinfo(_code):
        return {
            "sub": "google-sub-existing",
            "email": "SAME@example.com",
            "email_verified": True,
        }

    monkeypatch.setattr(backend_module, "_fetch_google_userinfo", userinfo)
    _, state = _begin_google_flow(https_client, monkeypatch)
    callback = https_client.get(
        "/api/auth/google/callback",
        params={"state": state, "code": "google-code"},
        follow_redirects=False,
    )

    assert callback.status_code == 303
    assert https_client.get("/api/auth/me").json()["id"] == existing["id"]
    linked = auth_store.get_google_user_by_subject("google-sub-existing")
    assert linked["id"] == existing["id"]
    assert auth_store.get_user_by_email("same@example.com")["password_hash"] is not None


def test_google_callback_rejects_unverified_email_and_replayed_state(
    https_client,
    auth_store,
    monkeypatch,
):
    existing = auth_store.create_user(
        "unverified@example.com",
        hash_password("existing-password"),
    )
    exchanges = []

    async def userinfo(code):
        exchanges.append(code)
        return {
            "sub": "google-sub-unverified",
            "email": "unverified@example.com",
            "email_verified": False,
        }

    monkeypatch.setattr(backend_module, "_fetch_google_userinfo", userinfo)
    _, state = _begin_google_flow(https_client, monkeypatch)
    first = https_client.get(
        "/api/auth/google/callback",
        params={"state": state, "code": "google-code"},
        follow_redirects=False,
    )
    replay = https_client.get(
        "/api/auth/google/callback",
        params={"state": state, "code": "google-code"},
        follow_redirects=False,
    )

    assert first.status_code == replay.status_code == 303
    assert exchanges == ["google-code"]
    assert auth_store.get_user_by_email("unverified@example.com")["id"] == existing["id"]
    assert auth_store.get_google_user_by_subject("google-sub-unverified") is None
    assert https_client.get("/api/auth/me").status_code == 401


def test_google_only_account_can_set_password_through_password_reset_flow(
    https_client,
    auth_store,
    caplog,
):
    user = auth_store.resolve_google_user(
        "google-reset@example.com",
        "google-sub-reset",
    )
    assert user is not None
    assert user["password_hash"] is None

    csrf = https_client.get("/api/auth/csrf").json()["csrf_token"]
    requested = https_client.post(
        "/api/auth/request-reset",
        json={"email": "google-reset@example.com"},
        headers={"X-CSRF-Token": csrf},
    )

    assert requested.status_code == 202
    assert requested.json() == backend_module.GENERIC_RESET_RESPONSE
    reset_token_match = re.search(r"reset_token=([A-Za-z0-9_-]+)", caplog.text)
    assert reset_token_match is not None
    reset_token = reset_token_match.group(1)

    reset_csrf = https_client.get("/api/auth/csrf").json()["csrf_token"]
    changed = https_client.post(
        "/api/auth/reset-password",
        json={"token": reset_token, "password": "google-account-password"},
        headers={"X-CSRF-Token": reset_csrf},
    )

    assert changed.status_code == 200
    updated_user = auth_store.get_user_by_email("google-reset@example.com")
    assert backend_module.verify_password("google-account-password", updated_user["password_hash"])
    assert https_client.post(
        "/api/auth/signin",
        json={"email": "google-reset@example.com", "password": "google-account-password"},
    ).status_code == 200


def test_google_only_account_password_signin_fails_generically(https_client, auth_store):
    auth_store.resolve_google_user("google-only@example.com", "google-sub-only")

    response = https_client.post(
        "/api/auth/signin",
        json={"email": "google-only@example.com", "password": "some-password"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid email or password"


def test_google_only_account_can_add_password_from_authenticated_session(
    https_client,
    auth_store,
    monkeypatch,
):
    async def userinfo(_code):
        return {
            "sub": "google-sub-add-password",
            "email": "google-add-password@example.com",
            "email_verified": True,
        }

    monkeypatch.setattr(backend_module, "_fetch_google_userinfo", userinfo)
    _, state = _begin_google_flow(https_client, monkeypatch)
    callback = https_client.get(
        "/api/auth/google/callback",
        params={"state": state, "code": "google-code"},
        follow_redirects=False,
    )
    assert callback.status_code == 303
    assert https_client.get("/api/auth/password-status").json() == {"has_password": False}

    csrf = https_client.get("/api/auth/csrf").json()["csrf_token"]
    added = https_client.post(
        "/api/auth/add-password",
        json={"password": "new-google-account-password"},
        headers={"X-CSRF-Token": csrf},
    )

    assert added.status_code == 200
    assert added.json() == {"password_added": True}
    assert https_client.get("/api/auth/password-status").json() == {"has_password": True}
    user = auth_store.get_user_by_email("google-add-password@example.com")
    assert backend_module.verify_password("new-google-account-password", user["password_hash"])
    assert https_client.post(
        "/api/auth/add-password",
        json={"password": "attempt-to-replace-password"},
        headers={"X-CSRF-Token": csrf},
    ).status_code == 409
    user = auth_store.get_user_by_email("google-add-password@example.com")
    assert backend_module.verify_password("new-google-account-password", user["password_hash"])


def test_password_status_requires_an_authenticated_session(https_client):
    assert https_client.get("/api/auth/password-status").status_code == 401


def test_csrf_missing_and_mismatched_tokens_are_rejected(https_client):
    signup = https_client.post(
        "/api/auth/signup",
        json={"email": "csrf@example.com", "password": "a-long-test-password"},
    )
    assert signup.status_code == 201

    missing = https_client.post("/api/auth/logout")
    csrf = https_client.get("/api/auth/csrf").json()["csrf_token"]
    mismatched = https_client.post(
        "/api/auth/logout",
        headers={"X-CSRF-Token": "wrong-token"},
    )
    accepted = https_client.post("/api/auth/logout", headers={"X-CSRF-Token": csrf})

    assert missing.status_code == mismatched.status_code == 403
    assert accepted.status_code == 200


def test_authenticated_csrf_token_remains_valid_after_another_tab_fetches_it(https_client):
    signup = https_client.post(
        "/api/auth/signup",
        json={"email": "multi-tab@example.com", "password": "a-long-test-password"},
    )
    assert signup.status_code == 201

    first_tab_token = https_client.get("/api/auth/csrf").json()["csrf_token"]
    second_tab_token = https_client.get("/api/auth/csrf").json()["csrf_token"]
    response = https_client.post("/api/auth/logout", headers={"X-CSRF-Token": first_tab_token})

    assert first_tab_token == second_tab_token
    assert response.status_code == 200


def test_credentialed_cors_accepts_only_configured_frontend_origin(https_client):
    allowed_origin = backend_module.FRONTEND_ORIGINS[0]
    accepted = https_client.get("/api/auth/me", headers={"Origin": allowed_origin})
    rejected = https_client.get("/api/auth/me", headers={"Origin": "https://attacker.invalid"})

    assert accepted.headers.get("access-control-allow-origin") == allowed_origin
    assert accepted.headers.get("access-control-allow-credentials") == "true"
    assert "access-control-allow-origin" not in rejected.headers


def test_reset_request_is_generic_and_stores_only_a_token_hash(https_client, auth_store, caplog):
    user = auth_store.create_user("reset@example.com", hash_password("old-test-password"))

    def request_reset(email):
        csrf = https_client.get("/api/auth/csrf").json()["csrf_token"]
        return https_client.post(
            "/api/auth/request-reset",
            json={"email": email},
            headers={"X-CSRF-Token": csrf},
        )

    known = request_reset("reset@example.com")
    unknown = request_reset("missing@example.com")

    assert known.status_code == unknown.status_code == 202
    assert known.json() == unknown.json() == backend_module.GENERIC_RESET_RESPONSE
    with sqlite3.connect(auth_store.db_path) as connection:
        stored_token_hash = connection.execute("SELECT token_hash FROM password_reset_tokens").fetchone()[0]
    match = re.search(r"reset_token=([A-Za-z0-9_-]+)", caplog.text)
    assert match is not None
    reset_token = match.group(1)
    assert stored_token_hash == hash_session_token(reset_token)
    assert reset_token not in stored_token_hash
    assert auth_store.get_user_by_email("reset@example.com")["id"] == user["id"]


def test_password_reset_token_is_single_use_and_revokes_sessions(https_client, auth_store, caplog):
    signup = https_client.post(
        "/api/auth/signup",
        json={"email": "reset-flow@example.com", "password": "old-test-password"},
    )
    user = signup.json()
    request_csrf = https_client.get("/api/auth/csrf").json()["csrf_token"]
    requested = https_client.post(
        "/api/auth/request-reset",
        json={"email": user["email"]},
        headers={"X-CSRF-Token": request_csrf},
    )
    assert requested.status_code == 202
    reset_token = re.search(r"reset_token=([A-Za-z0-9_-]+)", caplog.text).group(1)

    reset_csrf = https_client.get("/api/auth/csrf").json()["csrf_token"]
    changed = https_client.post(
        "/api/auth/reset-password",
        json={"token": reset_token, "password": "new-test-password"},
        headers={"X-CSRF-Token": reset_csrf},
    )
    assert changed.status_code == 200
    assert https_client.get("/api/auth/me").status_code == 401

    with sqlite3.connect(auth_store.db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM password_reset_tokens").fetchone()[0] == 0
    updated_user = auth_store.get_user_by_email(user["email"])
    assert updated_user["password_hash"] != "new-test-password"
    assert backend_module.verify_password("new-test-password", updated_user["password_hash"])
    assert not backend_module.verify_password("old-test-password", updated_user["password_hash"])

    second_use_csrf = https_client.get("/api/auth/csrf").json()["csrf_token"]
    second_use = https_client.post(
        "/api/auth/reset-password",
        json={"token": reset_token, "password": "another-new-password"},
        headers={"X-CSRF-Token": second_use_csrf},
    )
    assert second_use.status_code == 400
    assert https_client.post(
        "/api/auth/signin",
        json={"email": user["email"], "password": "new-test-password"},
    ).status_code == 200


def test_production_cookie_is_always_secure(https_client, monkeypatch):
    monkeypatch.setattr(backend_module, "COOKIE_SECURE", True)
    response = https_client.post(
        "/api/auth/signup",
        json={"email": "production-cookie@example.com", "password": "a-long-test-password"},
    )
    assert response.status_code == 201
    assert "secure" in response.headers["set-cookie"].lower()


def test_auth_cleanup_removes_only_expired_records(auth_store):
    expired_user = auth_store.create_user("expired-cleanup@example.com", "hash")
    active_user = auth_store.create_user("active-cleanup@example.com", "hash")
    auth_store.create_session("expired-session", expired_user["id"], "2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z")
    auth_store.create_session("active-session", active_user["id"], "2020-01-01T00:00:00Z", "2099-01-02T00:00:00Z")
    auth_store.create_password_reset_token("expired-reset", expired_user["id"], "2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z")
    auth_store.add_preauth_csrf_token("expired-csrf", "2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z")

    deleted = auth_store.delete_expired_auth_tokens("2026-09-30T00:00:00Z")

    assert deleted == {"sessions": 1, "password_reset_tokens": 1, "preauth_csrf_tokens": 1}
    with sqlite3.connect(auth_store.db_path) as connection:
        assert connection.execute("SELECT token_hash FROM sessions").fetchall() == [("active-session",)]