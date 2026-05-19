import os
from pathlib import Path


def _load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _load_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf is required for PDF loading. Install with: uv pip install pypdf")
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _load_docx(path: str) -> str:
    try:
        from docx import Document
    except ImportError:
        raise ImportError("python-docx is required for DOCX loading. Install with: uv pip install python-docx")
    doc = Document(path)
    return "\n".join(para.text for para in doc.paragraphs)


def _load_md(path: str) -> str:
    return _load_txt(path)


_EXTENSION_LOADERS = {
    ".txt": _load_txt,
    ".md": _load_md,
    ".markdown": _load_md,
    ".pdf": _load_pdf,
    ".docx": _load_docx,
}


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        start = end - overlap
        if start <= end - chunk_size and start < text_len:
            start = end
    return [c.strip() for c in chunks if c.strip()]


class DocumentLoader:
    """Load and chunk documents for RAG."""

    supported_extensions: tuple[str, ...] = tuple(_EXTENSION_LOADERS.keys())

    @classmethod
    def load(cls, file_path: str | Path) -> str:
        path = Path(file_path)
        ext = path.suffix.lower()
        if ext not in _EXTENSION_LOADERS:
            raise ValueError(f"Unsupported file type: {ext}. Supported: {cls.supported_extensions}")
        return _EXTENSION_LOADERS[ext](str(path))

    @classmethod
    def load_and_chunk(cls, file_path: str | Path, chunk_size: int = 800, overlap: int = 100) -> list[str]:
        text = cls.load(file_path)
        return chunk_text(text, chunk_size=chunk_size, overlap=overlap)
