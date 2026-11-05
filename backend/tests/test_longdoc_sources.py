"""Source fidelity and pinning checks; never download from the network."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from longdoc_sources.__main__ import get, normalize_cc
from longdoc_sources.parsing import parse_jats, parse_markup
from longdoc_sources.study import compile_sources, qa_candidates


def test_markdown_nested_paths_and_code_not_interpreted_as_heading():
    blocks = parse_markup("# Model\n\nIntro.\n\n## Method\n\n```python\n# a code comment\nx = 3\n```\n\n| A | B |\n|---|---|\n| 1 | 2 |\n")
    code = next(b for b in blocks if b["type"] == "code")
    assert code["text"] == "```python\n# a code comment\nx = 3\n```"
    assert code["section_path"] == ["Model", "Method"]
    assert len([b for b in blocks if b["type"] == "heading"]) == 2
    assert next(b for b in blocks if b["type"] == "table")["text"].endswith("| 1 | 2 |")


def test_rst_directives_keep_indentation_and_nested_headings():
    source = "Guide\n=====\n\nSetup\n-----\n\n.. code-block:: python\n\n   x = 1\n   if x:\n       print(x)\n\n.. math::\n\n   E = mc^2\n\nText after formula.\n"
    blocks = parse_markup(source, rst=True)
    code = next(b for b in blocks if b["type"] == "code")
    assert "       print(x)" in code["text"]
    assert code["section_path"] == ["Guide", "Setup"]
    assert next(b for b in blocks if b["type"] == "formula")["text"].endswith("   E = mc^2")
    assert blocks[-1]["text"] == "Text after formula."


def test_jats_nested_sections_mathml_tables_and_reference_exclusion():
    xml = '''<article xmlns:xlink="http://www.w3.org/1999/xlink"><front><article-meta>
      <title-group><article-title>Crop study</article-title></title-group>
      <permissions><license xlink:href="https://creativecommons.org/licenses/by/4.0/"><license-p>CC BY</license-p></license></permissions>
      <abstract><p>Abstract facts.</p></abstract></article-meta></front>
      <body><sec><title>Methods</title><p>Setup.</p><sec><title>Measurement</title>
      <disp-formula><math><mi>x</mi><mo>=</mo><mn>2</mn></math></disp-formula>
      <table-wrap><caption><p>Results</p></caption><table><tr><td>Yield</td><td>3</td></tr></table></table-wrap>
      <p>Real evidence with <inline-formula><math><mfrac><mi>a</mi><mi>b</mi></mfrac></math></inline-formula>.</p></sec></sec></body>
      <back><ref-list><ref><mixed-citation>Not body text</mixed-citation></ref></ref-list>
      <app-group><app><title>Appendix</title><p>Additional evidence.</p></app></app-group></back></article>'''
    blocks, meta = parse_jats(xml)
    assert meta["title"] == "Crop study"
    assert normalize_cc(meta["license_links"], meta["license_text"]) == "CC-BY-4.0"
    assert next(b for b in blocks if b["type"] == "formula")["section_path"] == ["Methods", "Measurement"]
    assert "<td>Yield</td>" in next(b for b in blocks if b["type"] == "table")["text"]
    inline = next(b for b in blocks if b.get("inline_structures"))
    assert "<mfrac>" in inline["source_xml"]
    assert inline["inline_structures"][0]["type"] == "inline-formula"
    assert all("Not body text" not in b["text"] for b in blocks)
    assert blocks[-1]["section_path"] == ["Appendix"]


def test_cache_rechecks_pinned_hash_instead_of_trusting_old_download(tmp_path):
    url = "https://example.invalid/source"
    value = b"pinned source"
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=value)))
    expected = hashlib.sha256(value).hexdigest()
    assert get(client, url, tmp_path, expected_hash=expected) == value
    (tmp_path / hashlib.sha256(url.encode()).hexdigest()).write_bytes(b"changed")
    with pytest.raises(ValueError, match="mismatch"):
        get(client, url, tmp_path, expected_hash=expected)


def _doc(did, domain="biology", split="test"):
    return {"id": did, "family_id": did, "title": did, "source_url": "https://example.invalid/" + did,
            "license": "CC0-1.0", "revision": "controlled-fixture", "language": "en", "domain": domain,
            "kind": "controlled_fixture", "split": split,
            "blocks": [{"id": f"p{i}", "type": "paragraph", "text": (f"Unique paragraph {i} for {did}. " * 8),
                        "section_path": ["Methods"], "page": None} for i in range(2)]}


def test_qa_multiple_acceptable_paragraphs_do_not_claim_one_evidence_gold():
    task = qa_candidates([_doc("d1")])[0]
    assert len(task["reference_answers"]) == 2
    assert task["evidence"] == [] and task["evidence_review_pending"] is True
    assert len(task["candidate_evidence_options"]) == 2
    assert task["annotation_status"] == "ai_generated" and task["review_status"] == "unreviewed"


def test_compilations_balance_domains_and_preserve_source_lineage():
    docs = [_doc(f"a{i}", "agriculture") for i in range(8)] + [_doc(f"z{i}", "programming") for i in range(8)]
    # A small limit must not let alphabetically first domain monopolize output.
    compilations = compile_sources(docs, limit=2)
    assert {d["domain"] for d in compilations} == {"agriculture", "programming"}
    original = {d["id"]: d for d in docs}
    for compilation in compilations:
        assert compilation["training_eligible"] is False
        assert len(compilation["parent_family_ids"]) == 2
        assert all(original[did]["split"] == compilation["split"] for did in compilation["source_document_ids"])
        assert all(b["text"] == next(x for x in original[b["source_document_id"]]["blocks"] if x["id"] == b["source_block_id"])["text"] for b in compilation["blocks"] if b.get("source_document_id"))
