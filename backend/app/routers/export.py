from __future__ import annotations

from urllib.parse import quote
import io
import zipfile
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ..export_docx import CSL, PandocMissing, to_docx
from ..document_assets import export_assets
from ..library import bibtex_for

router = APIRouter(prefix="/api/export", tags=["export"])

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


class DocxIn(BaseModel):
    markdown: str = Field(..., max_length=2_000_000)
    title: str = "论文"
    paper_ids: list[str] = Field(default_factory=list)
    style: str = "gbt-numeric"  # gbt-numeric / gbt-author-date / apa
    doc_id: str | None = None
    asset_ids: list[str] = Field(default_factory=list, max_length=200)


@router.post("/docx")
def export_docx(body: DocxIn, request: Request):
    if body.style not in CSL:
        raise HTTPException(422, f"不支持的引用格式：{body.style}")
    try:
        files = export_assets(body.doc_id, body.asset_ids)
        data = to_docx(body.markdown, body.paper_ids, body.style, assets=files)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    except PandocMissing as e:
        raise HTTPException(503, str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, str(e)) from e
    name = (body.title.strip() or "论文")[:80] + ".docx"
    headers = {} if request.headers.get("x-ariadne-export") == "blob" else {"Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}"}
    return Response(data, media_type=DOCX, headers=headers)


class BundleIn(BaseModel):
    source: str = Field(..., max_length=2_000_000)
    format: Literal["tex", "md"]
    title: str = "论文"
    doc_id: str | None = None
    asset_ids: list[str] = Field(default_factory=list, max_length=200)
    paper_ids: list[str] = Field(default_factory=list, max_length=1000)


@router.post("/bundle")
def bundle(body: BundleIn, request: Request):
    files = export_assets(body.doc_id, body.asset_ids)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"main.{body.format}", body.source)
        archive.writestr("refs.bib", bibtex_for(body.paper_ids) if body.paper_ids else "")
        for name, data in files.items():
            archive.writestr(name, data)
        archive.writestr("README.txt", "LaTeX: use XeLaTeX + BibTeX (for Chinese, install ctex or use Overleaf).\n"
                         "Markdown: Pandoc Markdown with refs.bib and relative images/.\n"
                         "Figures stay in document order; no floating layouts or merged table cells.\n")
    filename = quote((body.title.strip() or "论文")[:80] + f"-{body.format}.zip")
    headers = {} if request.headers.get("x-ariadne-export") == "blob" else {"Content-Disposition": f"attachment; filename*=UTF-8''{filename}"}
    return Response(stream.getvalue(), media_type="application/zip", headers=headers)


@router.get("/docx/status")
def docx_status():
    """前端据此决定 Word 导出是否可用"""
    try:
        from ..export_docx import pandoc_path

        pandoc_path()
        return {"ok": True}
    except PandocMissing as e:
        return {"ok": False, "message": str(e)}
