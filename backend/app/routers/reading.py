"""用户主动确认的阅读记录；打开/滚动 PDF 不意味着已读。"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from ..db import get_conn

router = APIRouter(prefix="/api/reading", tags=["reading"])


class ReadingPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["unknown", "unread", "reading", "read"] | None = None
    read_through_page: int | None = Field(default=None, ge=1, le=100000, strict=True)
    total_pages: int | None = Field(default=None, ge=1, le=100000, strict=True)


def empty_record(pid: str) -> dict:
    return {"paper_id": pid, "status": "unknown", "read_through_page": None,
            "total_pages": None, "updated_at": None}


@router.get("")
def reading_list():
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT p.id AS paper_id, COALESCE(r.status, 'unknown') AS status,
                   r.read_through_page, r.total_pages, r.updated_at
            FROM papers p LEFT JOIN paper_reading r ON r.paper_id=p.id
            ORDER BY p.id
        """).fetchall()
    return [dict(r) for r in rows]


@router.patch("/{pid}")
def reading_update(pid: str, body: ReadingPatch):
    fields = body.model_dump(exclude_unset=True)
    if "status" in fields and fields["status"] is None:
        raise HTTPException(422, "阅读状态不能为空")
    with get_conn() as conn:
        # Serialize read-modify-write so partial updates do not clobber other fields.
        conn.execute("BEGIN IMMEDIATE")
        if not conn.execute("SELECT 1 FROM papers WHERE id=?", (pid,)).fetchone():
            raise HTTPException(404, "文献不存在")
        row = conn.execute("SELECT * FROM paper_reading WHERE paper_id=?", (pid,)).fetchone()
        record = dict(row) if row else empty_record(pid)
        record.update(fields)
        if record["status"] in {"unknown", "unread"}:
            if fields.get("read_through_page") is not None:
                raise HTTPException(422, "记录读至页码前请确认阅读中或已读")
            record["read_through_page"] = None
        page, total = record["read_through_page"], record["total_pages"]
        if page is not None and total is not None and page > total:
            raise HTTPException(422, "读至页码不能超过总页数")
        conn.execute("""
            INSERT INTO paper_reading (paper_id, status, read_through_page, total_pages)
            VALUES (?,?,?,?) ON CONFLICT(paper_id) DO UPDATE SET
                status=excluded.status, read_through_page=excluded.read_through_page,
                total_pages=excluded.total_pages, updated_at=datetime('now')
        """, (pid, record["status"], page, total))
        return dict(conn.execute("SELECT * FROM paper_reading WHERE paper_id=?", (pid,)).fetchone())
