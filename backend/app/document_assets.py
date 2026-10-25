"""Document-owned images; stable IDs, no external URLs or paths in exports."""
from __future__ import annotations

import hashlib
import io
import re
import uuid
from pathlib import Path
from typing import Any

from PIL import Image
from fastapi import HTTPException

from .config import get_settings
from .db import get_conn
from .security import max_upload_bytes

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 25_000_000
ASSET_ID = re.compile(r"^[0-9a-f]{32}$")
UNSAFE_TEX = re.compile(r"\\(?:input|include|includegraphics|write\d*|openout|openin|read|catcode|csname|endcsname|def|gdef|edef|xdef|newcommand|renewcommand|usepackage|documentclass|href|url|html\w*|special|directlua|loop|repeat)\b", re.I)


def require_document(doc_id: str):
    with get_conn() as conn:
        if not conn.execute("SELECT 1 FROM documents WHERE id=?", (doc_id,)).fetchone():
            raise HTTPException(404, "文档不存在")


def asset_path(row: dict) -> Path:
    return get_settings().data_dir / "images" / f"{row['id']}.{row['extension']}"


def asset(doc_id: str, asset_id: str) -> dict:
    if not ASSET_ID.fullmatch(asset_id):
        raise HTTPException(422, "无效的图片资源 ID")
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM document_assets WHERE id=? AND doc_id=?", (asset_id, doc_id)).fetchone()
    if not row:
        raise HTTPException(404, "图片不存在或不属于当前文档，请重新上传")
    out = dict(row)
    if not asset_path(out).is_file():
        raise HTTPException(404, "图片文件缺失，请重新上传")
    return out


def view(row: dict) -> dict:
    return {"id": row["id"], "src": f"/api/documents/{row['doc_id']}/assets/{row['id']}",
            "width": row["width"], "height": row["height"], "extension": row["extension"], "size": row["size"]}


def store_image(doc_id: str, data: bytes) -> dict:
    require_document(doc_id)
    if len(data) > min(MAX_IMAGE_BYTES, max_upload_bytes()):
        raise HTTPException(413, "图片超过上传大小上限（单张最多 10 MB）")
    ext = "png" if data.startswith(b"\x89PNG\r\n\x1a\n") else "jpg" if data.startswith(b"\xff\xd8\xff") else None
    if ext is None:
        raise HTTPException(422, "仅支持有效的 PNG、JPEG 图片")
    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            if width * height > MAX_PIXELS:
                raise HTTPException(413, "图片超过 2500 万像素，请缩小后上传")
            if image.format not in {"PNG", "JPEG"}:
                raise ValueError("unsupported format")
            image.verify()
        # Decode, rather than trusting the extension, MIME type or image header.
        with Image.open(io.BytesIO(data)) as image:
            image.load()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(422, "图片损坏或格式不受支持") from e
    digest = hashlib.sha256(data).hexdigest()
    with get_conn() as conn:
        existing = conn.execute("SELECT * FROM document_assets WHERE doc_id=? AND digest=?", (doc_id, digest)).fetchone()
        if existing and asset_path(dict(existing)).is_file():
            return view(dict(existing))
        if existing:
            row = dict(existing)
        else:
            row = {"id": uuid.uuid4().hex, "doc_id": doc_id, "digest": digest, "extension": ext,
                   "width": width, "height": height, "size": len(data)}
        dest = asset_path(row)
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            temp.write_bytes(data)
            temp.replace(dest)
            conn.execute("INSERT OR IGNORE INTO document_assets(id,doc_id,digest,extension,width,height,size) VALUES (?,?,?,?,?,?,?)",
                         tuple(row[k] for k in ("id", "doc_id", "digest", "extension", "width", "height", "size")))
            saved = dict(conn.execute("SELECT * FROM document_assets WHERE doc_id=? AND digest=?", (doc_id, digest)).fetchone())
            if saved["id"] != row["id"]:
                dest.unlink(missing_ok=True)
            return view(saved)
        finally:
            temp.unlink(missing_ok=True)


def validate_content(doc_id: str, content: dict) -> None:
    """Only validate new node types; old document structures remain compatible."""
    pending: list[tuple[Any, int]] = [(content, 0)]
    count = 0
    while pending:
        node, depth = pending.pop()
        count += 1
        if depth > 30 or count > 50000 or not isinstance(node, dict):
            raise HTTPException(422, "文档结构过大或无效")
        kind, attrs = node.get("type"), node.get("attrs") or {}
        if not isinstance(attrs, dict):
            raise HTTPException(422, "文档节点属性无效")
        if kind == "image":
            a = asset(doc_id, str(attrs.get("assetId", "")))
            if attrs.get("src") != view(a)["src"]:
                raise HTTPException(422, "图片地址与资源不匹配，请重新上传图片")
            if attrs.get("extension") != a["extension"]:
                raise HTTPException(422, "图片格式与资源不匹配")
            width = attrs.get("widthPercent", 100)
            if not isinstance(width, (int, float)) or not 10 <= width <= 100:
                raise HTTPException(422, "图片宽度需要在 10% 到 100% 之间")
        elif kind in ("inlineMath", "blockMath"):
            tex = attrs.get("latex")
            if not isinstance(tex, str) or not tex.strip() or len(tex) > 8000 or UNSAFE_TEX.search(tex):
                raise HTTPException(422, "公式为空、过长或包含不支持的命令")
        elif kind == "table":
            rows = node.get("content") or []
            if not isinstance(rows, list) or any(not isinstance(r, dict) or r.get("type") != "tableRow" for r in rows):
                raise HTTPException(422, "表格行结构无效")
            widths = [len(r.get("content") or []) for r in rows]
            if not rows or len(rows) > 50 or not widths or not 1 <= widths[0] <= 10 or len(set(widths)) != 1:
                raise HTTPException(422, "表格需要为规则矩形，最多 50 行、10 列")
        elif kind in ("tableCell", "tableHeader"):
            if attrs.get("colspan", 1) != 1 or attrs.get("rowspan", 1) != 1:
                raise HTTPException(422, "第一版暂不支持合并单元格")
            if any(c.get("type") != "paragraph" for c in node.get("content") or []):
                raise HTTPException(422, "表格单元格仅支持文字、引用和行内公式")
        pending.extend((child, depth + 1) for child in node.get("content") or [])


def export_assets(doc_id: str | None, ids: list[str]) -> dict[str, bytes]:
    if not ids:
        return {}
    if not doc_id or len(ids) > 200:
        raise HTTPException(422, "导出图片需要指定文档，且最多包含 200 张图片")
    files = {}
    total = 0
    for aid in set(ids):
        row = asset(doc_id, aid)
        data = asset_path(row).read_bytes()
        total += len(data)
        if total > 100 * 1024 * 1024:
            raise HTTPException(413, "导出图片总大小超过 100 MB，请缩小图片后重试")
        files[f"images/{row['id']}.{row['extension']}"] = data
    return files
