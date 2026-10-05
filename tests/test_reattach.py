import os
import shutil
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
    # Create conversation and monkeypatch RAGService.build_index to create a chroma dir
    with TestClient(app) as client:
        resp = client.post("/api/conversations")
        assert resp.status_code == 200
        conv = resp.json()["id"]

        # Upload a dummy document to build the index (monkeypatching heavy operations)
        class FakeUpload:
            def __init__(self, name, content):
                self.filename = name
                self._content = content

            async def read(self):
                return self._content

        # Use the real endpoint but simulate persistence by creating the .chroma_store/<conv> directory
        # Ensure registry empty and simulate an existing chroma dir
        _SERVICE_REGISTRY.clear()
        project_root = __import__('pathlib').Path(__file__).resolve().parent.parent
        store_dir = project_root / ".chroma_store" / conv
        store_dir.mkdir(parents=True, exist_ok=True)

        # Now ensure that a new ensure_service will create a RAGService and reuse the dir
        # by calling the backend endpoint that triggers ensure
        r = client.get(f"/api/conversations/{conv}/status")
        assert r.status_code == 200
        data = r.json()
        assert data.get("chroma_exists") is True

        # Simulate backend restart by clearing registry and calling questions endpoint
        _SERVICE_REGISTRY.clear()
        # The service should be re-created on demand; call status to trigger creation
        r2 = client.get(f"/api/conversations/{conv}/status")
        assert r2.status_code == 200

        # Clean up
        shutil.rmtree(store_dir)
