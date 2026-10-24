"""论文全文解析与切块。

- PyMuPDF 按文本块抽取，双栏论文先读左栏再读右栏；
- 识别章节标题，切块时记录 section / page，检索结果可以精确到出处；
- 遇到 References / 参考文献 就停止，避免参考文献列表污染检索；
- 切块循环保证一定前进（修复了旧版 chunk_text 的死循环）。
"""
from __future__ import annotations

from . import tracing

import re
from dataclasses import dataclass
from pathlib import Path

from .db import get_conn
from .embeddings import embed_texts, embeddings_enabled, to_blob

_HEADING = re.compile(
    r"^\s*(?:"
    r"(?:\d{1,2}(?:\.\d{1,2}){0,2}\.?\s+[A-Z一-鿿][^\n]{1,80})"  # 1 Introduction / 2.1 Method
    r"|(?:[IVX]{1,4}\.\s+[A-Z][^\n]{1,80})"  # II. RELATED WORK
    r"|(?:Abstract|Introduction|Related Work|Background|Method(?:s|ology)?|Approach|Experiments?|Results|"
    r"Discussion|Conclusions?|Limitations|Acknowledg(?:e)?ments?)"
    r"|(?:摘\s*要|引\s*言|绪\s*论|相关工作|方法|实验|结论|讨论|致\s*谢)"
    r"|(?:第[一二三四五六七八九十\d]+章[^\n]{0,40})"
    r")\s*$"
)
_REFERENCES = re.compile(r"^\s*(?:\d+\.?\s*)?(?:References|REFERENCES|Bibliography|参考文献)\s*$")


@dataclass
class Chunk:
    text: str
    section: str | None
    page: int | None


def _clean(text: str) -> str:
    text = text.replace("­", "")
    text = re.sub(r"-\n(?=[a-z])", "", text)  # 断词连字符
    text = re.sub(r"[ \t]+", " ", text)
    return text


def _page_text(page) -> str:
    """按阅读顺序取一页的文字，处理双栏排版。

    get_text(sort=True) 只按纵坐标排序，双栏论文会把左右两栏同一高度的行交错拼在一起。
    这里按文本块处理：跨栏的块（标题、通栏图表说明）按原位置放；它们之间的部分先读完左栏再读右栏。
    """
    width = page.rect.width
    blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
    blocks.sort(key=lambda b: (b[1], b[0]))
    out: list[str] = []
    left: list[tuple] = []
    right: list[tuple] = []

    def flush() -> None:
        for col in (left, right):
            out.extend(b[4] for b in sorted(col, key=lambda b: (b[1], b[0])))
            col.clear()

    for b in blocks:
        x0, _, x1 = b[0], b[1], b[2]
        if x1 - x0 > width * 0.6:  # 通栏
            flush()
            out.append(b[4])
        elif (x0 + x1) / 2 < width / 2:
            left.append(b)
        else:
            right.append(b)
    flush()
    return "\n\n".join(t.strip("\n") for t in out)  # 一个文本块大致是一段，空行分隔


def extract_pdf_pages(path: str | Path) -> list[str]:
    import fitz  # PyMuPDF

    with fitz.open(str(path)) as doc:
        return [_clean(_page_text(page)) for page in doc]


def split_paragraphs(pages: list[str]) -> list[tuple[str, str | None, int]]:
    """返回 [(段落, 所在章节, 页码)]；到参考文献为止。"""
    out: list[tuple[str, str | None, int]] = []
    section: str | None = None
    for pno, page in enumerate(pages, start=1):
        buf: list[str] = []

        def flush() -> None:
            para = " ".join(s.strip() for s in buf if s.strip())
            if len(para) > 30:
                out.append((para, section, pno))
            buf.clear()

        for line in page.split("\n"):
            if _REFERENCES.match(line):
                flush()
                return out
            if _HEADING.match(line) and len(line.strip()) < 90:
                flush()
                section = line.strip()
                continue
            if not line.strip():
                flush()
            else:
                buf.append(line)
        flush()
    return out


def chunk_paragraphs(
    paras: list[tuple[str, str | None, int]], max_chars: int = 1200, overlap_chars: int = 200
) -> list[Chunk]:
    chunks: list[Chunk] = []
    cur: list[str] = []
    cur_len = 0
    has_new = False  # cur 里是否有重叠尾巴之外的新内容
    cur_section: str | None = None
    cur_page: int | None = None

    def emit() -> None:
        nonlocal cur, cur_len, has_new
        if cur and has_new:
            chunks.append(Chunk(" ".join(cur).strip(), cur_section, cur_page))
            tail = cur[-1][-overlap_chars:] if overlap_chars else ""
            cur = [tail] if tail else []
            cur_len = len(tail)
            has_new = False

    for text, section, page in paras:
        if section != cur_section and cur:
            emit()
            cur, cur_len = [], 0  # 换章节不带重叠
        if cur_section != section or cur_page is None:
            cur_page = page
        cur_section = section
        # 超长段落按句子硬切，保证每次循环都前进
        pieces = [text] if len(text) <= max_chars else _hard_split(text, max_chars)
        for piece in pieces:
            if cur_len + len(piece) > max_chars and has_new:
                emit()
                cur_page = page
            cur.append(piece)
            cur_len += len(piece) + 1
            has_new = True
    if cur and has_new:
        chunks.append(Chunk(" ".join(cur).strip(), cur_section, cur_page))
    return [c for c in chunks if c.text]


def _hard_split(text: str, max_chars: int) -> list[str]:
    sents = re.split(r"(?<=[.!?。！？；;])\s*", text)
    out, buf = [], ""
    for s in sents:
        while len(s) > max_chars:  # 极端情况：没有句号的超长串
            out.append(s[:max_chars])
            s = s[max_chars:]
        if len(buf) + len(s) > max_chars and buf:
            out.append(buf)
            buf = ""
        buf += (" " if buf and s else "") + s  # 切分时吃掉了句间空格，拼回去时补上（否则会变成 "facts.Purely"）
    if buf:
        out.append(buf)
    return out


def chunk_plain_text(text: str, max_chars: int = 1200, overlap_chars: int = 200) -> list[Chunk]:
    paras = [(p.strip(), None, 1) for p in re.split(r"\n\s*\n", text) if p.strip()]
    return chunk_paragraphs(paras, max_chars, overlap_chars)


def store_chunks(paper_id: str, chunks: list[Chunk], *, fulltext: bool) -> int:
    vectors = embed_texts([c.text for c in chunks]) if (chunks and embeddings_enabled()) else None
    with get_conn() as conn:
        if fulltext:
            conn.execute("DELETE FROM chunks WHERE paper_id=?", (paper_id,))
        for i, c in enumerate(chunks):
            conn.execute(
                "INSERT INTO chunks (paper_id, idx, section, page, text, embedding) VALUES (?,?,?,?,?,?)",
                (paper_id, i, c.section, c.page, c.text, to_blob(vectors[i]) if vectors else None),
            )
        if fulltext:
            conn.execute("UPDATE papers SET has_fulltext=1 WHERE id=?", (paper_id,))
    from .retrieval import invalidate_index

    invalidate_index()
    return len(chunks)


@tracing.traced("tool", "索引文献摘要")
def index_abstract(paper_id: str, title: str, abstract: str | None) -> int:
    """没有全文时，先把 标题+摘要 作为一个块，保证论文能被检索到。"""
    with get_conn() as conn:
        exists = conn.execute("SELECT count(*) FROM chunks WHERE paper_id=?", (paper_id,)).fetchone()[0]
    if exists:
        return 0
    text = f"{title}. {abstract or ''}".strip()
    return store_chunks(paper_id, [Chunk(text, "Abstract", None)], fulltext=False)


@tracing.traced("tool", "解析并索引全文")
def index_pdf(paper_id: str, path: str | Path, report: dict | None = None) -> int:
    """解析全文并入库。解析器按设置选择（PyMuPDF / GROBID / MinerU），外部服务不可用时自动回退 PyMuPDF。"""
    from .parsers import parse_pdf

    paras, parser, note = parse_pdf(path)
    if report is not None:
        report.update(parser=parser, note=note)
    chunks = chunk_paragraphs(paras)
    if not chunks:  # 扫描版 PDF 等
        hint = "（可能是扫描版：在设置里启用 MinerU 解析可以做 OCR）" if parser == "pymupdf" else ""
        raise ValueError(f"没能从 PDF 中提取到文本{hint}")
    n = store_chunks(paper_id, chunks, fulltext=True)
    with get_conn() as conn:
        conn.execute("UPDATE papers SET parser=? WHERE id=?", (parser, paper_id))
    return n


def guess_pdf_metadata(path: str | Path) -> dict:
    """从 PDF 元数据/首页猜标题和作者，用于直接上传的本地 PDF。"""
    import fitz

    with fitz.open(str(path)) as doc:
        meta = doc.metadata or {}
        title = (meta.get("title") or "").strip()
        author = (meta.get("author") or "").strip()
        if not title or len(title) < 6 or title.lower().endswith((".pdf", ".dvi", ".tex")):
            first = doc[0].get_text("dict") if len(doc) else {"blocks": []}
            best, best_size = "", 0.0
            for b in first.get("blocks", []):
                for line in b.get("lines", []):
                    for span in line.get("spans", []):
                        txt = span.get("text", "").strip()
                        if len(txt) > 8 and span.get("size", 0) > best_size:
                            best, best_size = txt, span["size"]
            title = best or Path(path).stem
    authors = [a.strip() for a in re.split(r",|;| and ", author) if a.strip()] if author else []
    return {"title": title, "authors": authors}
