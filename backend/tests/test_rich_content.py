"""Real image bytes, saved editor JSON, and editable Word/portable ZIP output."""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

os.environ.setdefault("LLM_FAKE", "1")
os.environ.setdefault("ARIA_DATA_DIR", tempfile.mkdtemp(prefix="aria-rich-test-"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def picture(format="PNG"):
    buffer = io.BytesIO()
    Image.new("RGB", (24, 16), (60, 120, 190)).save(buffer, format=format)
    return buffer.getvalue()


def document():
    return client.post("/api/documents", json={"title": "图表公式测试"}).json()["id"]


def upload(did, data=None, filename="figure.png"):
    return client.post(f"/api/documents/{did}/assets", files={"file": (filename, picture() if data is None else data, "image/png")})


def test_upload_validates_actual_images_and_scopes_resources():
    did, other = document(), document()
    result = upload(did)
    assert result.status_code == 200
    a = result.json()
    assert a["width"] == 24 and a["height"] == 16
    assert upload(did).json()["id"] == a["id"]
    assert upload(other).json()["id"] != a["id"]
    assert client.get(a["src"]).content == picture()
    assert client.get(f"/api/documents/{other}/assets/{a['id']}").status_code == 404
    assert upload(did, b"<svg><script>alert(1)</script></svg>").status_code == 422
    assert upload(did, b"\x89PNG\r\n\x1a\ncorrupt").status_code == 422
    assert upload("nonexistent").status_code == 404
    jpg = upload(did, picture("JPEG"), "renamed.png").json()
    assert jpg["extension"] == "jpg"  # actual format, not supplied MIME/extension


def test_image_size_limit(monkeypatch):
    monkeypatch.setenv("ARIA_MAX_UPLOAD_MB", "0.00001")
    assert upload(document()).status_code == 413


def test_rich_json_roundtrip_and_cross_document_images_are_rejected():
    did = document()
    a = upload(did).json()
    content = {"type": "doc", "content": [
        {"type": "paragraph", "content": [{"type": "text", "text": "正文 "}, {"type": "inlineMath", "attrs": {"latex": "x^2"}}]},
        {"type": "image", "attrs": {"assetId": a["id"], "extension": a["extension"], "src": a["src"], "widthPercent": 50, "caption": "图：实验结果"}},
        {"type": "blockMath", "attrs": {"latex": r"\sum_{i=1}^n x_i"}},
        {"type": "table", "content": [{"type": "tableRow", "content": [
            {"type": "tableCell", "content": [{"type": "paragraph", "content": [{"type": "text", "text": "cell"}]}]}
        ]}]},
    ]}
    assert client.put(f"/api/documents/{did}", json={"content": content}).status_code == 200
    assert client.get(f"/api/documents/{did}").json()["content"] == content
    other = document()
    assert client.put(f"/api/documents/{other}", json={"content": content}).status_code == 404
    bad = json.loads(json.dumps(content))
    bad["content"][1]["attrs"]["src"] = "https://external.invalid/image.png"
    assert client.put(f"/api/documents/{did}", json={"content": bad}).status_code == 422
    assert client.get(f"/api/documents/{did}").json()["content"] == content


@pytest.mark.parametrize("node", [
    {"type": "blockMath", "attrs": {"latex": r"\input{secret}"}},
    {"type": "inlineMath", "attrs": {"latex": ""}},
    {"type": "table", "content": []},
    {"type": "table", "content": [{"type": "tableRow", "content": [{"type": "tableCell", "attrs": {"colspan": 2}}]}]},
])
def test_invalid_rich_nodes_fail_without_saving(node):
    did = document()
    assert client.put(f"/api/documents/{did}", json={"content": {"type": "doc", "content": [node]}}).status_code == 422


def test_word_embeds_image_native_table_and_editable_math():
    did = document()
    a = upload(did).json()
    md = (f"# 混排报告\n\n前文 $x^2$。\n\n![实验图](images/{a['id']}.png){{width=50%}}\n\n"
          "| 方法 | 分数 |\n| --- | --- |\n| Baseline | 0.8 |\n\n$$\n\\frac{a}{b}\n$$\n\n末尾正文。\n")
    response = client.post("/api/export/docx", json={"markdown": md, "doc_id": did, "asset_ids": [a["id"]]})
    assert response.status_code == 200, response.text[:500] if response.status_code != 200 else ""
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    root = ET.fromstring(archive.read("word/document.xml"))
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main", "m": "http://schemas.openxmlformats.org/officeDocument/2006/math"}
    assert root.findall(".//w:tbl", ns)
    assert len(root.findall(".//m:oMath", ns)) >= 2
    assert any(name.startswith("word/media/") for name in archive.namelist())
    text = "".join(root.itertext())
    assert text.index("前文") < text.index("实验图") < text.index("Baseline") < text.index("末尾正文")


@pytest.mark.parametrize("src", ["https://example.com/a.png", "file:///etc/passwd", "../settings.json", "data:image/png;base64,AA=="])
def test_word_never_fetches_unmanaged_image_urls(src):
    r = client.post("/api/export/docx", json={"markdown": f"![image]({src})"})
    assert r.status_code == 422


def test_bundle_includes_images_bibtex_and_relative_filenames():
    did = document()
    a = upload(did).json()
    source = f"\\includegraphics{{images/{a['id']}.png}}"
    r = client.post("/api/export/bundle", json={"format": "tex", "source": source, "doc_id": did, "asset_ids": [a["id"]]})
    assert r.status_code == 200
    z = zipfile.ZipFile(io.BytesIO(r.content))
    assert {"main.tex", "refs.bib", "README.txt", f"images/{a['id']}.png"} == set(z.namelist())
    assert z.read("main.tex").decode() == source
    assert z.read("refs.bib") == b""  # never export the entire library when no paper is cited
    assert z.read(f"images/{a['id']}.png") == picture()
    assert client.post("/api/export/bundle", json={"format": "tex", "source": source, "doc_id": document(), "asset_ids": [a["id"]]}).status_code == 404


def test_old_plain_document_and_docx_still_work():
    did = document()
    content = {"type": "doc", "content": [{"type": "paragraph", "content": [{"type": "text", "text": "Old document"}]}]}
    assert client.put(f"/api/documents/{did}", json={"content": content}).status_code == 200
    assert client.post("/api/export/docx", json={"markdown": "Old document"}).status_code == 200


def test_blob_exports_do_not_trigger_a_second_attachment_download():
    response = client.post("/api/export/bundle", json={"format": "tex", "source": "test"}, headers={"X-Ariadne-Export": "blob"})
    assert response.status_code == 200 and response.content[:2] == b"PK"
    assert "content-disposition" not in response.headers
    response = client.post("/api/export/docx", json={"markdown": "test"}, headers={"X-Ariadne-Export": "blob"})
    assert response.status_code == 200 and response.content[:2] == b"PK"
    assert "content-disposition" not in response.headers


@pytest.mark.parametrize("empty_middle", [False, True])
def test_word_headerless_table_preserves_data_rows_without_a_fake_header(empty_middle):
    markdown = "|  |  |\n| --- | --- |\n| Method | Score |\n"
    if empty_middle:
        markdown += "|  |  |\n"
    markdown += "| Baseline | 0.8 |\n"
    response = client.post("/api/export/docx", json={"markdown": markdown})
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    tables = root.findall(".//w:tbl", ns)
    assert len(tables) == 1
    rows = tables[0].findall("w:tr", ns)
    assert len(rows) == (3 if empty_middle else 2)
    values = [["".join(cell.itertext()).strip() for cell in row.findall("w:tc", ns)] for row in rows]
    assert values[0] == ["Method", "Score"]
    assert values[-1] == ["Baseline", "0.8"]
    if empty_middle:
        assert values[1] == ["", ""]



def test_word_preserves_new_text_formats_and_safe_links():
    md = '[underline]{.underline} ~~removed~~ ^sup^ ~sub~ [mark]{.aria-highlight data-color="#c8e6c9"} [site](https://example.com)'
    response = client.post('/api/export/docx', json={'markdown': md})
    assert response.status_code == 200, response.text[:300] if response.status_code != 200 else ''
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        root = ET.fromstring(z.read('word/document.xml'))
        styles = ET.fromstring(z.read('word/styles.xml'))
        relations = z.read('word/_rels/document.xml.rels').decode()
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    assert root.findall('.//w:u', ns)
    assert root.findall('.//w:strike', ns)
    assert {n.get('{'+ns['w']+'}val') for n in root.findall('.//w:vertAlign', ns)} >= {'superscript', 'subscript'}
    assert any(n.get('{'+ns['w']+'}fill') == 'C8E6C9' for n in styles.findall('.//w:shd', ns))
    assert 'https://example.com' in relations


@pytest.mark.parametrize('url', ['javascript:alert(1)', 'data:text/html,hello', 'file:///secret'])
def test_word_rejects_executable_link_schemes(url):
    assert client.post('/api/export/docx', json={'markdown': f'[link]({url})'}).status_code == 422
