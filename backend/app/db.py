"""SQLite 存储：文献元数据、全文切块、PDF 笔记、论文文档。单文件、零运维，适合本地使用。"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from typing import Iterator

from .config import get_settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id            TEXT PRIMARY KEY,
    bibkey        TEXT UNIQUE NOT NULL,
    title         TEXT NOT NULL,
    authors       TEXT NOT NULL DEFAULT '[]',
    year          INTEGER,
    venue         TEXT,
    doi           TEXT,
    arxiv_id      TEXT,
    url           TEXT,
    pdf_url       TEXT,
    abstract      TEXT,
    source        TEXT,
    citation_count INTEGER,
    has_fulltext  INTEGER NOT NULL DEFAULT 0,
    parser        TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS chunks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id   TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    idx        INTEGER NOT NULL,
    section    TEXT,
    page       INTEGER,
    text       TEXT NOT NULL,
    embedding  BLOB
);
CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id);
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    content     TEXT NOT NULL,
    cite_style  TEXT NOT NULL DEFAULT 'numeric',
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS llm_calls (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                TEXT NOT NULL DEFAULT (datetime('now')),
    tier              TEXT NOT NULL,
    model             TEXT NOT NULL,
    stream            INTEGER NOT NULL DEFAULT 0,
    ok                INTEGER NOT NULL,
    latency_ms        INTEGER,
    first_token_ms    INTEGER,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    cached_tokens     INTEGER,
    cost_usd          REAL,
    error             TEXT,
    thread_id         TEXT
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_tier ON llm_calls(tier, ts);
CREATE TABLE IF NOT EXISTS search_cache (
    key         TEXT PRIMARY KEY,
    results     TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS notes (
    id          TEXT PRIMARY KEY,
    paper_id    TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    page        INTEGER NOT NULL,
    quote       TEXT NOT NULL,
    comment     TEXT NOT NULL DEFAULT '',
    rects       TEXT NOT NULL DEFAULT '[]',
    color       TEXT NOT NULL DEFAULT 'yellow',
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_notes_paper ON notes(paper_id, page);
CREATE TABLE IF NOT EXISTS paper_reading (
    paper_id TEXT PRIMARY KEY REFERENCES papers(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'unknown' CHECK(status IN ('unknown','unread','reading','read')),
    read_through_page INTEGER CHECK(read_through_page >= 1),
    total_pages INTEGER CHECK(total_pages >= 1),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    CHECK(read_through_page IS NULL OR total_pages IS NULL OR read_through_page <= total_pages)
);
CREATE TABLE IF NOT EXISTS agent_threads (
    thread_id   TEXT PRIMARY KEY,
    doc_id      TEXT,
    topic       TEXT NOT NULL,
    kind        TEXT NOT NULL DEFAULT 'review',
    status      TEXT NOT NULL DEFAULT 'running',
    stats       TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

_lock = threading.Lock()
_initialized = False


def db_path() -> str:
    return str(get_settings().data_dir / "aria.sqlite")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(db_path(), check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db() -> None:
    global _initialized
    with _lock:
        if _initialized:
            return
        conn = _connect()
        conn.executescript(_SCHEMA)
        # 旧数据库补列：命中缓存的输入 tokens（用于按缓存价计费）
        cols = {r[1] for r in conn.execute("PRAGMA table_info(llm_calls)")}
        if "cached_tokens" not in cols:
            conn.execute("ALTER TABLE llm_calls ADD COLUMN cached_tokens INTEGER")
        # 旧数据库补列：调用属于哪个 Agent 任务（运行统计、排查问题）
        if "thread_id" not in cols:
            conn.execute("ALTER TABLE llm_calls ADD COLUMN thread_id TEXT")
        # 旧数据库补列：Agent 任务类型（review=综述→大纲，write=分节写作）
        cols = {r[1] for r in conn.execute("PRAGMA table_info(agent_threads)")}
        if "kind" not in cols:
            conn.execute("ALTER TABLE agent_threads ADD COLUMN kind TEXT NOT NULL DEFAULT 'review'")
        # 旧数据库补列：Agent 运行统计（token、速度、缓存命中、上下文占用）
        if "stats" not in cols:
            conn.execute("ALTER TABLE agent_threads ADD COLUMN stats TEXT")
        # 旧数据库补列：全文是哪个解析器解析的（pymupdf / grobid / mineru）
        cols = {r[1] for r in conn.execute("PRAGMA table_info(papers)")}
        if "parser" not in cols:
            conn.execute("ALTER TABLE papers ADD COLUMN parser TEXT")
        conn.commit()
        conn.close()
        _initialized = True


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    init_db()
    conn = _connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
