"""Word 导出：编辑器生成的 Markdown（pandoc 引用语法 [@key]、[@key, p. 3]）+ 文献库生成的 BibTeX，
用 pandoc + citeproc 转成 .docx，参考文献按 CSL 格式排版（GB/T 7714—2015 / APA）。

- pandoc 来自 pypandoc-binary（随 Python 依赖一起安装，不用另装），找不到时再用系统里的 pandoc，或 PANDOC_PATH 指定；
- 版式来自 assets/reference-zh.docx：正文宋体 / Times New Roman 小四、1.5 倍行距、首行缩进 2 字符，标题黑体。
"""
from __future__ import annotations

import os
import io
import zipfile
from xml.dom import minidom
from urllib.parse import urlsplit
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .library import bibtex_for

ASSETS = Path(__file__).resolve().parent / "assets"
CSL = {
    "gbt-numeric": "china-national-standard-gb-t-7714-2015-numeric.csl",
    "gbt-author-date": "china-national-standard-gb-t-7714-2015-author-date.csl",
    "apa": "apa.csl",
}


class PandocMissing(RuntimeError):
    pass


def pandoc_path() -> str:
    if os.getenv("PANDOC_PATH"):
        return os.environ["PANDOC_PATH"]
    try:
        import pypandoc  # type: ignore

        p = pypandoc.get_pandoc_path()
        if p:
            return p
    except Exception:  # noqa: BLE001
        pass
    p = shutil.which("pandoc")
    if p:
        return p
    raise PandocMissing("没有找到 pandoc：请重新安装后端依赖（包含 pypandoc-binary），或安装 pandoc 后设置 PANDOC_PATH")


def to_docx(markdown: str, paper_ids: list[str], style: str = "gbt-numeric", refs_title: str = "参考文献",
            assets: dict[str, bytes] | None = None) -> bytes:
    exe = pandoc_path()
    csl = ASSETS / "csl" / CSL.get(style, CSL["gbt-numeric"])
    with tempfile.TemporaryDirectory(prefix="aria-docx-") as tmp:
        d = Path(tmp)
        body = markdown.rstrip()
        args = [exe, "in.json", "-f", "json", "-t", "docx", "-o", "out.docx", "--reference-doc", str(ASSETS / "reference-zh.docx")]
        for filename, data in (assets or {}).items():
            target = d / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        if paper_ids:
            (d / "refs.bib").write_text(bibtex_for(paper_ids), encoding="utf-8")
            # 参考文献放在文末，带标题
            body += f"\n\n# {refs_title}\n\n::: {{#refs}}\n:::\n"
            args += ["--citeproc", "--bibliography", "refs.bib", "--csl", str(csl), "-M", "link-citations=true"]
        (d / "in.md").write_text(body + "\n", encoding="utf-8")
        parsed = subprocess.run([exe, "in.md", "-f", "markdown", "-t", "json", "--sandbox"],
                                cwd=d, capture_output=True, text=True, encoding="utf-8", timeout=30)
        if parsed.returncode:
            raise ValueError(f"无法解析文档：{parsed.stderr[:300]}")
        document = json.loads(parsed.stdout)
        document["meta"] = {k: v for k, v in document.get("meta", {}).items() if k in {"title", "author", "date", "lang"}}

        highlight_styles: dict[str, str] = {}

        def check(node):
            if isinstance(node, dict):
                if node.get("t") == "Link":
                    url = node["c"][-1][0]
                    if not (url.startswith("#") or urlsplit(url).scheme.lower() in {"https", "http", "mailto"}):
                        raise ValueError("链接地址无效，仅支持网页、邮件和文档锚点")
                if node.get("t") == "Span" and "aria-highlight" in node["c"][0][1]:
                    color = dict(node["c"][0][2]).get("data-color", "#fff59d").lower()
                    if color not in {"#fff59d", "#c8e6c9", "#bbdefb"}:
                        raise ValueError("不支持的高亮颜色")
                    name = "AriaHighlight" + color[1:].upper()
                    highlight_styles[name] = color[1:].upper()
                    node["c"][0] = ["", [], [["custom-style", name]]]
                if node.get("t") == "Image":
                    src = node["c"][-1][0]
                    if src not in (assets or {}):
                        raise ValueError("图片缺失或使用了外部地址，请重新上传并导出")
                if node.get("t") in {"RawBlock", "RawInline"}:
                    raise ValueError("文档包含不支持的原始 HTML / LaTeX，请使用编辑器的插入功能")
                if node.get("t") == "Math":
                    from .document_assets import UNSAFE_TEX
                    if UNSAFE_TEX.search(node["c"][1]):
                        raise ValueError("公式包含不支持的命令")
                for value in node.values():
                    check(value)
            elif isinstance(node, list):
                for value in node:
                    check(value)
        check(document)
        (d / "in.json").write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
        args += ["-M", "lang=zh-CN"]
        r = subprocess.run(args, cwd=d, capture_output=True, text=True, encoding="utf-8", timeout=120)
        if r.returncode != 0:
            raise RuntimeError(f"pandoc 转换失败：{(r.stderr or r.stdout).strip()[:500]}")
        if re.search(r"could not convert.*(?:tex|math)|could not fetch resource", r.stderr, re.I):
            raise ValueError(f"部分公式或图片无法转换：{r.stderr.strip()[:500]}")
        output = (d / "out.docx").read_bytes()
        if highlight_styles:
            output = _highlight_docx(output, highlight_styles)
        return output


def _highlight_docx(data: bytes, styles: dict[str, str]) -> bytes:
    """Use native Word character shading; retain all other archive entries."""
    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    result = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as source, zipfile.ZipFile(result, "w", zipfile.ZIP_DEFLATED) as target:
        # Preserve all namespace declarations, including mc:Ignorable prefixes.
        root = minidom.parseString(source.read("word/styles.xml"))
        for name, color in styles.items():
            style = next((s for s in root.getElementsByTagNameNS(ns, "style") if s.getAttributeNS(ns, "styleId") == name), None)
            if style is None:
                style = root.createElementNS(ns, "w:style")
                style.setAttributeNS(ns, "w:type", "character")
                style.setAttributeNS(ns, "w:styleId", name)
                label = root.createElementNS(ns, "w:name")
                label.setAttributeNS(ns, "w:val", name)
                style.appendChild(label)
                root.documentElement.appendChild(style)
            candidates = style.getElementsByTagNameNS(ns, "rPr")
            properties = candidates[0] if candidates else root.createElementNS(ns, "w:rPr")
            if not candidates:
                style.appendChild(properties)
            shading = root.createElementNS(ns, "w:shd")
            for key, value in {"val": "clear", "color": "auto", "fill": color}.items():
                shading.setAttributeNS(ns, "w:" + key, value)
            properties.appendChild(shading)
        for entry in source.infolist():
            target.writestr(entry, root.toxml(encoding="utf-8") if entry.filename == "word/styles.xml" else source.read(entry.filename))
    return result.getvalue()
