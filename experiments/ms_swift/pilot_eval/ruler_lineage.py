"""Attach source lineage to unchanged RULER qa_1 examples.

The upstream SQuAD generator selects answerable questions in original source
order; its seed changes distractors, not question selection. The 10-example
pilot therefore uses questions from one article, not 10 independent families.
This helper never changes the prompt, references, or generated haystack.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path


SQUAD_SHA256 = "80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8"
QUESTION_MARKER = "\n\nQuestion: "


def _answerable_records(document):
    """Match the nested iteration and impossible-question filter in read_squad."""
    result = []
    seen = set()
    for article in document["data"]:
        title = article["title"]
        if not isinstance(title, str) or not title:
            raise ValueError("SQuAD article requires a title")
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            if not isinstance(context, str) or not context:
                raise ValueError("SQuAD support paragraph is empty")
            for qa in paragraph["qas"]:
                if qa["is_impossible"]:
                    continue
                qid, question = qa["id"], qa["question"]
                answers = [a["text"] for a in qa["answers"]]
                if not isinstance(qid, str) or not qid or qid in seen:
                    raise ValueError("SQuAD answerable QA IDs must be unique")
                if not isinstance(question, str) or not question or not answers or any(not isinstance(a, str) or not a for a in answers):
                    raise ValueError("SQuAD answerable QA has missing query/references")
                seen.add(qid)
                result.append({"question": question, "answers": answers,
                               "source_qa_id": qid, "source_article": title,
                               "source_support_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
                               "source_answerable_index": len(result)})
    return result


@lru_cache(maxsize=2)
def _load_verified_records(absolute_path, expected_sha):
    content = Path(absolute_path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha:
        raise ValueError("pinned SQuAD source changed during lineage loading")
    return _answerable_records(json.loads(content))


def _annotate(raw, records):
    prompt, answers = raw.get("input"), raw.get("outputs")
    if not isinstance(prompt, str) or QUESTION_MARKER not in prompt:
        raise ValueError("RULER QA input lacks the exact official terminal question marker")
    query = prompt.rsplit(QUESTION_MARKER, 1)[1]
    # Do not strip punctuation, lowercase, reorder, or deduplicate references.
    # Those transformations could silently bind a modified generated example.
    if not isinstance(answers, list) or not answers or any(not isinstance(a, str) or not a for a in answers):
        raise ValueError("RULER QA outputs must be nonempty original reference strings")
    candidates = [row for row in records if row["question"] == query and row["answers"] == answers]
    if len(candidates) != 1:
        raise ValueError("RULER QA query/reference lineage is missing or ambiguous")
    row = candidates[0]
    index = raw.get("index")
    if type(index) is not int or not 0 <= index < 10 or row["source_answerable_index"] != index:
        raise ValueError("RULER pilot QA must match the first ten answerable upstream records")
    title_hash = hashlib.sha256(row["source_article"].encode("utf-8")).hexdigest()
    return {"cluster_id": "ruler-squad-article:" + title_hash,
            "source_qa_id": row["source_qa_id"], "source_article": row["source_article"],
            "source_support_sha256": row["source_support_sha256"],
            "source_answerable_index": row["source_answerable_index"],
            "selection_note": "Unchanged upstream first 10 answerable SQuAD records in source order; seed controls distractors, not QA selection. Clustered by article across questions and lengths; this pilot's first 10 queries cover only the Normans article, so it cannot supply an independent-source QA confidence interval."}


def annotate_ruler_qa(raw, squadpath):
    """Return conservative article-level lineage, refusing unpinned sources."""
    path = Path(squadpath).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != SQUAD_SHA256:
        raise ValueError("RULER QA lineage requires the pinned SQuAD dev-v2.0 bytes")
    return _annotate(raw, _load_verified_records(str(path), SQUAD_SHA256))
