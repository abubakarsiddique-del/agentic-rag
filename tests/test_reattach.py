from datetime import datetime, timedelta, timezone
from fastapi.testclient import TestClient
import pytest

import backend.app as backend_module
from backend.app import app, _SERVICE_REGISTRY
from backend.auth import SESSION_COOKIE_NAME, hash_password, hash_session_token, new_session_token
from persistence.store import SQLiteConversationStore


@pytest.fixture(autouse=True)
def authenticated_backend(tmp_path, monkeypatch):
    store = SQLiteConversationStore(db_path=tmp_path / "reattach-test.db")
    user = store.create_user("reattach-test@example.com", hash_password("reattach-test-password-123"))
    token = new_session_token()
    now = datetime.now(timezone.utc)
    store.create_session(
        hash_session_token(token),
        user["id"],
        now.isoformat(timespec="seconds").replace("+00:00", "Z"),
        (now + timedelta(days=1)).isoformat(timespec="seconds").replace("+00:00", "Z"),
    )
    monkeypatch.setattr(backend_module, "_get_store", lambda: store)
    original_request = TestClient.request

    def authenticated_request(self, method, url, **kwargs):
        headers = dict(kwargs.pop("headers", {}) or {})
        headers.setdefault("cookie", f"{SESSION_COOKIE_NAME}={token}")
        path_parts = url.split("?", 1)[0].strip("/").split("/")
        csrf_required = method.upper() in {"POST", "PUT", "PATCH", "DELETE"} or (
            method.upper() == "GET"
            and path_parts[:2] == ["api", "conversations"]
            and (len(path_parts) == 3 or (len(path_parts) == 4 and path_parts[3] in {"documents", "status"}))
        )
        if csrf_required and url not in {
            "/api/auth/signup", "/api/auth/signin",
        }:
            csrf_response = original_request(self, "GET", "/api/auth/csrf", headers=headers)
            headers.setdefault("x-csrf-token", csrf_response.json()["csrf_token"])
        kwargs["headers"] = headers
        return original_request(self, method, url, **kwargs)

    monkeypatch.setattr(TestClient, "request", authenticated_request)


def test_reattach_reuses_chroma(tmp_path, monkeypatch):
    calls = []

    class FakeService:
        def has_documents(self):
            return True

    service = FakeService()

    def ensure_service(conversation_id):
        calls.append(conversation_id)
        return service

    monkeypatch.setattr(backend_module, "_ensure_service", ensure_service)
    with TestClient(app) as client:
        resp = client.post("/api/conversations")
        assert resp.status_code == 200
        conv = resp.json()["id"]
        _SERVICE_REGISTRY.clear()
        r = client.get(f"/api/conversations/{conv}/status")
        assert r.status_code == 200
        data = r.json()
        assert data.get("chroma_exists") is True

        _SERVICE_REGISTRY.clear()
        r2 = client.get(f"/api/conversations/{conv}/status")
        assert r2.status_code == 200
        assert calls == [conv, conv]
