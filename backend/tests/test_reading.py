"""Reading records use an isolated database; never touch the user's library."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("LLM_FAKE", "1")
os.environ.setdefault("ARIA_DATA_DIR", tempfile.mkdtemp(prefix="aria-reading-"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from app import db
from app.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Only this fixture's connections are redirected. Existing suite globals stay intact.
    monkeypatch.setattr(db, "db_path", lambda: str(tmp_path / "reading.sqlite"))
    monkeypatch.setattr(db, "_initialized", False)
    db.init_db()
    with TestClient(app) as c:
        yield c


def paper(client, title="Reading paper"):
    response = client.post("/api/papers", json={"title": title, "authors": ["Test Author"]})
    assert response.status_code == 200, response.text
    return response.json()["paper"]["id"]


def record(client, pid):
    return next(r for r in client.get("/api/reading").json() if r["paper_id"] == pid)


def test_old_paper_without_pdf_defaults_to_unknown(client):
    pid = paper(client)
    assert record(client, pid) == {"paper_id": pid, "status": "unknown", "read_through_page": None,
                                   "total_pages": None, "updated_at": None}
    # Read-only list never manufactures a record (nor changes status).
    with db.get_conn() as conn:
        assert conn.execute("SELECT COUNT(*) FROM paper_reading").fetchone()[0] == 0


def test_progress_persists_across_database_reinitialization(client, monkeypatch):
    pid = paper(client)
    result = client.patch(f"/api/reading/{pid}", json={"status": "reading", "read_through_page": 3, "total_pages": 10})
    assert result.status_code == 200
    monkeypatch.setattr(db, "_initialized", False)
    db.init_db()
    assert record(client, pid) == result.json()
    assert result.json()["updated_at"]


def test_correct_clear_and_mark_unread(client):
    pid = paper(client)
    url = f"/api/reading/{pid}"
    client.patch(url, json={"status": "reading", "read_through_page": 5, "total_pages": 10})
    assert client.patch(url, json={"read_through_page": 2}).json()["read_through_page"] == 2
    cleared = client.patch(url, json={"read_through_page": None, "total_pages": None}).json()
    assert cleared["read_through_page"] is None and cleared["total_pages"] is None
    client.patch(url, json={"read_through_page": 3})
    unread = client.patch(url, json={"status": "unread"}).json()
    assert unread["read_through_page"] is None
    assert unread["status"] == "unread"
    assert client.patch(url, json={"read_through_page": 1}).status_code == 422
    # External reading can be confirmed without a PDF or known page count.
    assert client.patch(url, json={"status": "read"}).json()["total_pages"] is None
    assert client.patch(url, json={"status": "unknown"}).json()["status"] == "unknown"


@pytest.mark.parametrize("fields", [
    {"status": "done"}, {"status": None}, {"read_through_page": 0}, {"read_through_page": -1},
    {"read_through_page": 1.5}, {"read_through_page": True}, {"total_pages": 0}, {"total_pages": "10"},
    {"total_pages": 100001}, {"read_through_page": 11}, {"total_pages": 2}, {"unexpected": "field"},
])
def test_invalid_patch_preserves_saved_record(client, fields):
    pid = paper(client)
    url = f"/api/reading/{pid}"
    old = client.patch(url, json={"status": "reading", "read_through_page": 3, "total_pages": 10}).json()
    response = client.patch(url, json=fields)
    assert response.status_code == 422, response.text
    assert record(client, pid) == old


def test_missing_paper_and_delete_cascade(client):
    assert client.patch("/api/reading/missing", json={"status": "read"}).status_code == 404
    pid = paper(client)
    client.patch(f"/api/reading/{pid}", json={"status": "read"})
    assert client.delete(f"/api/papers/{pid}").status_code == 200
    assert not client.get("/api/reading").json()
    with db.get_conn() as conn:
        assert conn.execute("SELECT COUNT(*) FROM paper_reading").fetchone()[0] == 0
    assert record(client, paper(client))["status"] == "unknown"


def test_notes_and_document_remain_independent_of_reading(client):
    pid = paper(client)
    note = client.post("/api/notes", json={"paper_id": pid, "page": 1, "quote": "A highlight"}).json()
    content = {"type": "doc", "content": [{"type": "paragraph", "content": [
        {"type": "citation", "attrs": {"ids": [pid]}}
    ]}]}
    doc = client.post("/api/documents", json={"content": content}).json()
    client.patch(f"/api/reading/{pid}", json={"status": "read"})
    assert client.get(f"/api/documents/{doc['id']}").json()["content"] == content
    assert client.get("/api/notes", params={"paper_id": pid}).json() == [note]
    client.delete(f"/api/notes/{note['id']}")
    assert client.get("/api/notes", params={"paper_id": pid}).json() == []
    assert record(client, pid)["status"] == "read"


def test_upgrade_legacy_database_without_reading_table(client, monkeypatch):
    pid = paper(client)
    with db.get_conn() as conn:
        conn.execute("DROP TABLE paper_reading")
    monkeypatch.setattr(db, "_initialized", False)
    db.init_db()
    assert record(client, pid)["status"] == "unknown"
    db.init_db()  # idempotent
