"""Conservative source parsers: retain structure and code, never claim PDF layout gold."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET


def parse_markup(text: str, *, rst: bool = False) -> list[dict]:
    """Parse Markdown/RST paragraphs, heading paths and fenced/directive blocks.

    Source lines and original block text are retained. Includes are not expanded;
    references/images/directives are source markup, not rendered content.
    """
    lines = text.splitlines()
    blocks: list[dict] = []
    path: list[str] = []
    buf: list[str] = []
    start = 1
    fence: str | None = None
    kind = "paragraph"
    adornments: dict[str, int] = {}

    def emit() -> None:
        nonlocal buf, kind
        value = "\n".join(buf).strip("\n")
        if value.strip():
            inferred = kind
            if kind == "paragraph" and (value.lstrip().startswith("$$") or ".. math::" in value):
                inferred = "formula"
            if kind == "paragraph" and (re.search(r"^\s*\|.*\|\s*$", value, re.M) or ".. list-table::" in value or ".. csv-table::" in value):
                inferred = "table"
            blocks.append({"id": f"b{len(blocks):05d}", "type": inferred, "text": value,
                           "section_path": list(path), "page": None, "source_line": start})
        buf = []
        kind = "paragraph"

    i = 0
    while i < len(lines):
        line = lines[i]
        if fence:
            buf.append(line)
            if line.lstrip().startswith(fence):
                emit()
                fence = None
            i += 1
            continue
        fm = re.match(r"^\s*(`{3,}|~{3,})", line)
        if fm:
            emit()
            start = i + 1
            fence = fm.group(1)
            kind = "code"
            buf = [line]
            i += 1
            continue
        hm = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line) if not rst else None
        level, title, consumed = (len(hm.group(1)), hm.group(2), 1) if hm else (0, "", 0)
        if rst and i + 1 < len(lines) and line.strip() and not line[:1].isspace():
            underline = lines[i + 1].strip()
            if len(underline) >= max(3, len(line.strip()) - 2) and len(set(underline)) == 1 and underline[0] in "=-~^\"+#*":
                key = underline[0]
                if key not in adornments:
                    adornments[key] = len(adornments) + 1
                level, title, consumed = adornments[key], line.strip(), 2
        if level:
            emit()
            path = path[:level - 1] + [title]
            blocks.append({"id": f"b{len(blocks):05d}", "type": "heading", "text": title,
                           "section_path": list(path), "page": None, "source_line": i + 1, "level": level})
            i += consumed
            continue
        # RST directive payload belongs with its directive, including blank lines.
        if rst and re.match(r"^\s*\.\.\s+(?:code-block|code|math|list-table|csv-table)::", line):
            emit()
            start = i + 1
            kind = "formula" if "math::" in line else "table" if "table::" in line else "code"
            buf = [line]
            i += 1
            while i < len(lines) and (not lines[i].strip() or lines[i][:1].isspace()):
                buf.append(lines[i])
                i += 1
            emit()
            continue
        if not line.strip():
            emit()
        else:
            if not buf:
                start = i + 1
            buf.append(line)
        i += 1
    emit()
    return blocks


def _tag(el: ET.Element) -> str:
    return el.tag.rsplit("}", 1)[-1]


def plain(el: ET.Element | None) -> str:
    return "" if el is None else re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def parse_jats(xml: str) -> tuple[list[dict], dict]:
    """Extract JATS body/abstract/appendices; explicitly omit reference lists.

    Formula/table blocks retain XML, including MathML/table cell relationships.
    It is a source parser, not a visual layout quality annotation.
    """
    root = ET.fromstring(xml)
    if _tag(root) != "article":
        raise ValueError("Expected a JATS article")
    blocks: list[dict] = []

    def add(el: ET.Element, path: list[str], kind: str) -> None:
        value = ET.tostring(el, encoding="unicode") if kind in {"formula", "table"} else plain(el)
        if value:
            blocks.append({"id": f"b{len(blocks):05d}", "type": kind, "text": value,
                           "section_path": list(path), "page": None})
            if kind == "paragraph":
                # Prose is useful for retrieval, while MathML/table relationships
                # must remain recoverable rather than silently lost by itertext.
                nested = [node for node in el.iter() if node is not el and _tag(node) in
                          {"inline-formula", "disp-formula", "table-wrap", "table"}]
                if nested:
                    blocks[-1]["source_xml"] = ET.tostring(el, encoding="unicode")
                    blocks[-1]["inline_structures"] = [{"type": _tag(node), "xml": ET.tostring(node, encoding="unicode")} for node in nested]

    def visit(el: ET.Element, path: list[str]) -> None:
        tag = _tag(el)
        if tag in {"ref-list", "permissions", "article-meta", "journal-meta"}:
            return
        if tag in {"sec", "app", "abstract"}:
            title_el = next((c for c in el if _tag(c) == "title"), None)
            title = plain(title_el) or ("Abstract" if tag == "abstract" else "")
            if title:
                path = path + [title]
                blocks.append({"id": f"b{len(blocks):05d}", "type": "heading", "text": title,
                               "section_path": list(path), "page": None, "level": len(path)})
        if tag in {"p", "disp-quote", "list-item"}:
            add(el, path, "paragraph")
            return
        if tag in {"disp-formula", "inline-formula"}:
            add(el, path, "formula")
            return
        if tag == "table-wrap":
            add(el, path, "table")
            return
        if tag in {"fig", "boxed-text"}:
            add(el, path, "paragraph")
            return
        for child in el:
            if _tag(child) != "title":
                visit(child, path)

    for abstract in root.findall("./front/article-meta/abstract"):
        visit(abstract, [])
    for container in (root.find("body"), root.find("back/app-group")):
        if container is not None:
            visit(container, [])
    license_el = root.find(".//permissions/license")
    license_text = plain(license_el)
    license_links = [] if license_el is None else [v for e in license_el.iter() for k, v in e.attrib.items() if k.endswith("href")]
    return blocks, {"title": plain(root.find(".//article-title")), "license_text": license_text,
                    "license_links": license_links, "excluded_elements": ["ref-list"], "parser": "jats-v1"}
