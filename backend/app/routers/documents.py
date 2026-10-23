from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..db import get_conn
from ..document_assets import validate_content

router = APIRouter(prefix="/api/documents", tags=["documents"])

EMPTY_DOC = {"type": "doc", "content": [{"type": "paragraph"}]}


class DocIn(BaseModel):
    title: str | None = None
    content: dict | None = None
    cite_style: str | None = None


def _row(r) -> dict:
    d = dict(r)
    d["content"] = json.loads(d["content"])
    return d


@router.get("")
def docs_list():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, title, cite_style, created_at, updated_at FROM documents ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


@router.post("")
def docs_create(body: DocIn):
    did = uuid.uuid4().hex[:12]
    if body.content:
        validate_content(did, body.content)
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO documents (id, title, content, cite_style) VALUES (?,?,?,?)",
            (did, body.title or "未命名论文", json.dumps(body.content or EMPTY_DOC, ensure_ascii=False),
             body.cite_style or "numeric"),
        )
    return docs_get(did)


@router.get("/{did}")
def docs_get(did: str):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM documents WHERE id=?", (did,)).fetchone()
    if not row:
        raise HTTPException(404, "文档不存在")
    return _row(row)


@router.put("/{did}")
def docs_update(did: str, body: DocIn):
    sets, vals = ["updated_at=datetime('now')"], []
    if body.title is not None:
        sets.append("title=?")
        vals.append(body.title)
    if body.content is not None:
        validate_content(did, body.content)
        sets.append("content=?")
        vals.append(json.dumps(body.content, ensure_ascii=False))
    if body.cite_style is not None:
        sets.append("cite_style=?")
        vals.append(body.cite_style)
    with get_conn() as conn:
        cur = conn.execute(f"UPDATE documents SET {', '.join(sets)} WHERE id=?", (*vals, did))
    if cur.rowcount == 0:
        raise HTTPException(404, "文档不存在")
    return {"ok": True}


@router.delete("/{did}")
def docs_delete(did: str):
    with get_conn() as conn:
        conn.execute("DELETE FROM documents WHERE id=?", (did,))
    return {"ok": True}
