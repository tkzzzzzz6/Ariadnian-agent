from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from ..document_assets import MAX_IMAGE_BYTES, asset, asset_path, require_document, store_image
from ..security import max_upload_bytes

router = APIRouter(prefix="/api/documents", tags=["document images"])


@router.post("/{doc_id}/assets")
def upload(doc_id: str, file: UploadFile = File(...)):
    require_document(doc_id)
    limit = min(MAX_IMAGE_BYTES, max_upload_bytes())
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, "图片超过上传大小上限（单张最多 10 MB）")
    return store_image(doc_id, data)


@router.get("/{doc_id}/assets/{asset_id}")
def image(doc_id: str, asset_id: str):
    row = asset(doc_id, asset_id)
    return FileResponse(asset_path(row), media_type="image/png" if row["extension"] == "png" else "image/jpeg",
                        headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "private, max-age=3600"})
