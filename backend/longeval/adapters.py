"""Local benchmark import adapters; these do not implement official model runs."""
from __future__ import annotations

from typing import Any


def import_benchmark(records: list[dict], benchmark: str, *, license_name: str, source_url: str) -> tuple[list[dict], list[dict]]:
    """Preserve source labels/units without asserting local human verification.

    RULER accepts generated JSONL input/outputs records. LongBench v2 accepts
    dataset JSON/JSONL records with context, question, choice_A..D, answer.
    Evidence is deliberately empty: neither format provides block-span gold.
    """
    if benchmark not in {"ruler", "longbench-v2"}:
        raise ValueError("benchmark must be ruler or longbench-v2")
    documents, tasks = [], []
    for index, record in enumerate(records):
        sample_id = str(record.get("_id", record.get("index", record.get("id", index))))
        did = f"{benchmark}-{sample_id}"
        if benchmark == "longbench-v2":
            context = record.get("context", "")
            question = str(record.get("question", ""))
            choices = []
            for label in "ABCD":
                choice = record.get(f"choice_{label}")
                if choice is not None:
                    choices.append(f"{label}. {choice}")
            if choices:
                question += "\n" + "\n".join(choices)
            answer = str(record.get("answer", ""))
            references = [answer]
            status = "unreviewed"
            task_type = "multiple_choice"
            unit = "words"
            reported_length = record.get("length")
            domain = str(record.get("domain", record.get("sub_domain", "benchmark")))
        else:
            context = record.get("input", "")
            question = str(record.get("question") or "Complete the task specified in the supplied raw benchmark prompt.")
            references = record.get("outputs", record.get("answers", []))
            if isinstance(references, str):
                references = [references]
            if not isinstance(references, list) or not references or any(not isinstance(x, str) for x in references):
                raise ValueError(f"{did}: RULER outputs must be a nonempty list of strings")
            answer = references[0]
            status = "controlled_synthetic"
            task_type = "extractive_qa"
            unit = "generator_tokenizer_tokens"
            reported_length = record.get("length")
            domain = str(record.get("task", "synthetic"))
        if not isinstance(context, str) or not context or not question or not answer:
            raise ValueError(f"{did}: missing nonempty context/question/reference answer")
        metadata: dict[str, Any] = {"benchmark": benchmark, "source_sample_id": sample_id, "benchmark_reported_length": reported_length, "benchmark_reported_length_unit": unit, "official_full_run": False, "import_note": "Local adapter only; no claim of official full evaluation or independently verified labels."}
        documents.append({"id": did, "family_id": did, "title": did, "source_url": source_url, "license": license_name, "revision": str(record.get("revision", "imported-local-snapshot")), "language": str(record.get("language", "en")), "domain": domain, "kind": "benchmark_context", "split": "test", "blocks": [{"id": "context", "type": "paragraph", "text": context, "section_path": [], "page": None}], "metadata": metadata})
        tasks.append({"id": did + "-task", "task_type": task_type, "document_ids": [did], "question": question, "reference_answer": answer, "reference_answers": references, "evidence": [], "answerable": True, "language": str(record.get("language", "en")), "annotation_status": status, "split": "test", "label_origin": "original_benchmark", "metadata": metadata})
    return documents, tasks
