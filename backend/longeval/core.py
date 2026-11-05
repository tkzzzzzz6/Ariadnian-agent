"""Dataset contracts, honest retrieval diagnostics, and prediction scoring."""
from __future__ import annotations

import hashlib
import math
import random
import re
from collections import Counter, defaultdict
from typing import Any, Callable

SPLITS = {"train", "dev", "test"}
STATUSES = {"ai_generated", "unreviewed", "human_verified", "controlled_synthetic"}
ANSWER_TASKS = {"qa", "long_qa", "cross_document_qa", "cross_section_qa", "extractive_qa", "multiple_choice", "unanswerable"}


def terms(text: str) -> list[str]:
    """English words plus Chinese characters/bigrams; no downloaded segmenter."""
    text = text.lower()
    words = re.findall(r"[a-z0-9_]+", text)
    for part in re.findall(r"[\u3400-\u9fff]+", text):
        words.extend(part)
        words.extend(part[i:i + 2] for i in range(len(part) - 1))
    return words


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def quote_spans(text: str, evidence: dict) -> list[tuple[int, int]]:
    """All literal occurrences, or one validated explicit Unicode-char interval."""
    quote = evidence.get("quote", text)
    if not isinstance(quote, str) or not quote:
        raise ValueError("quote must be a nonempty string")
    if "start_char" in evidence or "end_char" in evidence:
        start = evidence.get("start_char")
        end = evidence.get("end_char")
        if start is None and type(end) is int:
            start = end - len(quote)
        if end is None and type(start) is int:
            end = start + len(quote)
        if type(start) is not int or type(end) is not int or start < 0 or end > len(text) or end <= start or text[start:end] != quote:
            raise ValueError("quote offsets must identify its exact original block substring")
        return [(start, end)]
    spans = []
    start = 0
    while True:
        found = text.find(quote, start)
        if found < 0:
            break
        spans.append((found, found + len(quote)))
        start = found + 1
    if not spans:
        raise ValueError("quote is absent from original block")
    return spans


def resolve_splits(documents: list[dict]) -> dict[str, str]:
    """Keep document families, declared duplicate clusters and exact copies together.

    Explicit split conflicts are rejected rather than silently overwritten. Inferred
    splits are stable hashes, not guarantees of balanced domain/length coverage.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: str, b: str) -> None:
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    seen: dict[str, str] = {}
    for doc in documents:
        did = str(doc.get("id", ""))
        find(did)
        body = "\n".join(str(b.get("text", "")) for b in doc.get("blocks", []) if isinstance(b, dict))
        keys = ["family:" + str(doc.get("family_id") or did)]
        keys.append("document:" + did)
        for parent_id in doc.get("source_document_ids", []):
            keys.append("document:" + parent_id)
        for parent_family in doc.get("parent_family_ids", []):
            keys.append("family:" + parent_family)
        for name in ("duplicate_cluster_id", "near_duplicate_cluster_id", "cluster_id"):
            if doc.get(name):
                keys.append(name + ":" + str(doc[name]))
        if body.strip():
            keys.append("exact:" + _digest(" ".join(body.split()).casefold()))
        for key in keys:
            if key in seen:
                union(did, seen[key])
            else:
                seen[key] = did
    groups: dict[str, list[dict]] = defaultdict(list)
    for doc in documents:
        groups[find(str(doc.get("id", "")))].append(doc)
    result = {}
    for group in groups.values():
        explicit = {d["split"] for d in group if d.get("split")}
        if explicit - SPLITS:
            raise ValueError(f"invalid document split: {sorted(explicit - SPLITS)}")
        if len(explicit) > 1:
            raise ValueError("split leakage within family/duplicate cluster: " + ", ".join(str(d.get("id")) for d in group))
        family = min(str(d.get("family_id") or d.get("id")) for d in group)
        bucket = int(_digest(family)[:8], 16) % 100
        split = next(iter(explicit)) if explicit else ("train" if bucket < 80 else "dev" if bucket < 90 else "test")
        result.update({str(d["id"]): split for d in group})
    return result


def validate(documents: list[dict], tasks: list[dict]) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    document_count = len(documents) if isinstance(documents, list) else 0
    task_count = len(tasks) if isinstance(tasks, list) else 0
    if not isinstance(documents, list):
        errors.append("documents must be a list of objects")
        documents = []
    if not isinstance(tasks, list):
        errors.append("tasks must be a list of objects")
        tasks = []
    if any(not isinstance(d, dict) for d in documents):
        errors.append("every document must be an object")
        documents = [d for d in documents if isinstance(d, dict)]
    if any(not isinstance(t, dict) for t in tasks):
        errors.append("every task must be an object")
        tasks = [t for t in tasks if isinstance(t, dict)]
    ids = [d.get("id") for d in documents]
    if any(not isinstance(x, str) or not x for x in ids):
        errors.append("documents require nonempty string IDs")
    if len(set(str(x) for x in ids)) != len(ids):
        errors.append("duplicate document IDs")
    docs = {d.get("id"): d for d in documents if isinstance(d.get("id"), str)}
    blocks: dict[tuple[str, str], dict] = {}
    for doc in documents:
        did = str(doc.get("id"))
        for key in ("family_id", "title", "language", "domain", "kind"):
            if not isinstance(doc.get(key), str) or not doc[key]:
                errors.append(f"{did}: missing nonempty string {key}")
        for key in ("source_url", "license", "revision"):
            if key not in doc or (doc[key] is not None and not isinstance(doc[key], str)):
                errors.append(f"{did}: {key} must be a string or null")
            elif not doc[key]:
                warnings.append(f"{did}: missing {key}; assessment only, no source/training rights inferred")
        if isinstance(doc.get("license"), str) and doc["license"].lower() in {"unknown", "unspecified"}:
            warnings.append(f"{did}: license is unknown; do not infer training/redistribution rights")
        for key in ("source_document_ids", "parent_family_ids"):
            if key in doc and (not isinstance(doc[key], list) or any(not isinstance(x, str) or not x for x in doc[key])):
                errors.append(f"{did}: {key} must be a list of nonempty strings")
        if "split" in doc and doc["split"] is not None and (not isinstance(doc["split"], str) or doc["split"] not in SPLITS):
            errors.append(f"{did}: invalid document split")
        if not isinstance(doc.get("blocks"), list) or not doc.get("blocks"):
            errors.append(f"{did}: blocks must be a nonempty list")
            continue
        for block in doc["blocks"]:
            if not isinstance(block, dict):
                errors.append(f"{did}: block is not an object")
                continue
            bid = block.get("id")
            if not isinstance(bid, str) or not bid:
                errors.append(f"{did}: block ID must be a nonempty string")
                continue
            if (did, bid) in blocks:
                errors.append(f"{did}: duplicate block ID {bid}")
            blocks[did, bid] = block
            if not isinstance(block.get("text"), str) or not block["text"]:
                errors.append(f"{did}/{bid}: block text must be nonempty")
            if not isinstance(block.get("type"), str):
                errors.append(f"{did}/{bid}: missing block type")
            if not isinstance(block.get("section_path"), list) or any(not isinstance(x, str) for x in block.get("section_path", [])):
                errors.append(f"{did}/{bid}: section_path must be a list of strings")
            page = block.get("page")
            if page is not None and (isinstance(page, bool) or not isinstance(page, int) or page < 1):
                errors.append(f"{did}/{bid}: page must be null or a positive integer")
    try:
        splits = resolve_splits(documents) if not errors else {}
    except (ValueError, TypeError, KeyError) as exc:
        errors.append(str(exc))
        splits = {}
    seen_tasks: set[str] = set()
    signatures: dict[str, str] = {}
    for task in tasks:
        tid = task.get("id")
        if not isinstance(tid, str) or not tid:
            errors.append("task requires nonempty string ID")
            tid = str(tid)
        if tid in seen_tasks:
            errors.append(f"duplicate task ID: {tid}")
        seen_tasks.add(tid)
        for key in ("task_type", "question", "language"):
            if not isinstance(task.get(key), str) or not task[key]:
                errors.append(f"{tid}: missing nonempty string {key}")
        if not isinstance(task.get("reference_answer"), str):
            errors.append(f"{tid}: reference_answer must be a string")
        if "reference_answers" in task and (not isinstance(task["reference_answers"], list) or not task["reference_answers"] or any(not isinstance(x, str) for x in task["reference_answers"])):
            errors.append(f"{tid}: reference_answers must be a nonempty string list")
        if not isinstance(task.get("answerable"), bool):
            errors.append(f"{tid}: answerable must be boolean")
        if not isinstance(task.get("annotation_status"), str) or task["annotation_status"] not in STATUSES:
            errors.append(f"{tid}: invalid annotation_status")
        if not isinstance(task.get("split"), str) or task["split"] not in SPLITS:
            errors.append(f"{tid}: invalid or missing split")
        dids = task.get("document_ids")
        if not isinstance(dids, list) or not dids or any(not isinstance(x, str) for x in dids):
            errors.append(f"{tid}: document_ids must be nonempty string list")
            dids = []
        if len(set(dids)) != len(dids):
            errors.append(f"{tid}: duplicate document_ids")
        for did in dids:
            if did not in docs:
                errors.append(f"{tid}: unknown document {did}")
            elif splits and task.get("split") != splits.get(did):
                errors.append(f"{tid}: task/document split mismatch for {did}")
        evidence = task.get("evidence")
        if not isinstance(evidence, list):
            errors.append(f"{tid}: evidence must be a list")
            evidence = []
        if task.get("answerable") is False and evidence:
            errors.append(f"{tid}: unanswerable task must have empty supporting evidence")
        for ev in evidence:
            if not isinstance(ev, dict):
                errors.append(f"{tid}: evidence must be an object")
                continue
            key = ev.get("document_id"), ev.get("block_id")
            if any(not isinstance(x, str) for x in key):
                errors.append(f"{tid}: evidence IDs must be strings")
                continue
            if key[0] not in dids:
                errors.append(f"{tid}: evidence document outside document_ids")
            block = blocks.get(key)
            quote = ev.get("quote")
            if block is None:
                errors.append(f"{tid}: unknown evidence block {key}")
            elif not isinstance(block.get("text"), str) or not isinstance(quote, str) or not quote or quote not in block["text"]:
                errors.append(f"{tid}: evidence quote is absent from original block {key}")
            else:
                try:
                    quote_spans(block["text"], ev)
                except ValueError as exc:
                    errors.append(f"{tid}: {key}: {exc}")
        signature = _digest(str(task.get("question", "")).strip().casefold())
        if signature in signatures and signatures[signature] != task.get("split"):
            errors.append(f"{tid}: identical question appears across splits")
        signatures[signature] = task.get("split")
        if task.get("answerable") and not evidence:
            warnings.append(f"{tid}: no evidence gold; excluded from evidence metrics")
    known_ids = {d["id"] for d in documents if isinstance(d.get("id"), str)}
    known_families = {d["family_id"] for d in documents if isinstance(d.get("family_id"), str)}
    for doc in documents:
        source_ids = doc.get("source_document_ids", [])
        family_ids = doc.get("parent_family_ids", [])
        if isinstance(source_ids, list) and any(isinstance(x, str) and x not in known_ids for x in source_ids):
            warnings.append(f"{doc.get('id')}: some parent documents absent; lineage split verification is partial")
        if isinstance(family_ids, list) and any(isinstance(x, str) and x not in known_families for x in family_ids):
            warnings.append(f"{doc.get('id')}: some parent families absent; lineage split verification is partial")
    return {"valid": not errors, "errors": errors, "warnings": warnings, "documents": document_count, "tasks": task_count, "resolved_splits": splits}


def build_diagnostics(documents: list[dict], max_tasks: int = 1000, seed: int = 13) -> list[dict]:
    """Generate location checks, not reasoning/QA gold or human annotations."""
    splits = resolve_splits(documents)
    tasks = []
    shuffled_documents = list(documents)
    rng = random.Random(seed)
    rng.shuffle(shuffled_documents)
    for doc in shuffled_documents:
        candidates = [b for b in doc["blocks"] if b.get("text", "").strip() and b.get("type") not in {"heading", "title"}]
        if not candidates:
            continue
        base = {"document_ids": [doc["id"]], "reference_answer": "", "answerable": True, "language": doc["language"], "annotation_status": "controlled_synthetic", "split": splits[doc["id"]], "label_origin": "deterministic_exact_quote_location", "diagnostic_only": True}
        block = candidates[len(candidates) // 2]
        quote = block["text"][:200]
        task = {**base, "id": f"diag-{doc['id']}-lookup", "task_type": "evidence_lookup", "question": f"Locate the original passage containing this exact quotation: {quote}", "evidence": [{"document_id": doc["id"], "block_id": block["id"], "quote": quote}]}
        tasks.append(task)
        first = candidates[0]
        last = next((b for b in reversed(candidates) if b.get("section_path") != first.get("section_path")), None)
        if last:
            evidence = [{"document_id": doc["id"], "block_id": b["id"], "quote": b["text"][:150]} for b in (first, last)]
            tasks.append({**base, "id": f"diag-{doc['id']}-cross-section", "task_type": "cross_section_evidence", "question": "Locate BOTH original passages (this is a controlled location check, not a reasoning task): " + " | ".join(e["quote"] for e in evidence), "evidence": evidence})
        marker = "ABSENT_" + _digest(doc["id"] + "negative")[:24]
        body = "\n".join(b["text"] for b in doc["blocks"])
        while marker in body:
            marker += "X"
        tasks.append({**base, "id": f"diag-{doc['id']}-absent", "task_type": "absent_marker", "question": marker, "reference_answer": "", "answerable": False, "evidence": [], "label_origin": "verified_absent_random_marker"})
    docs = {d["id"]: d for d in documents}
    grouped = defaultdict(list)
    for task in tasks:
        doc = docs[task["document_ids"][0]]
        size = sum(len(b["text"].encode("utf-8")) for b in doc["blocks"])
        bucket = "<8k" if size < 8192 else "8k-32k" if size < 32768 else "32k-128k" if size < 131072 else ">=128k"
        key = doc["language"], doc["domain"], doc["kind"], bucket
        task["sampling"] = {"method": "stratified_round_robin", "seed": seed, "stratum": list(key), "length_unit": "utf8_bytes"}
        grouped[key].append(task)
    keys = sorted(grouped)
    rng.shuffle(keys)
    selected = []
    cursor = 0
    while len(selected) < max_tasks:
        added = False
        for key in keys:
            if cursor < len(grouped[key]) and len(selected) < max_tasks:
                selected.append(grouped[key][cursor])
                added = True
        if not added:
            break
        cursor += 1
    return selected


def _truncate(text: str, budget: int) -> str:
    return text.encode("utf-8")[:budget].decode("utf-8", errors="ignore")


def select_context(task: dict, documents: dict[str, dict], strategy: str, budget_bytes: int, *, budget_tokens: int | None = None, length_counter: Callable[[str], int] | None = None) -> dict:
    candidates = [(did, block) for did in task["document_ids"] for block in documents[did]["blocks"]]
    tokens = [Counter(terms(b["text"])) for _, b in candidates]
    query = set(terms(task["question"]))
    average = sum(sum(t.values()) for t in tokens) / max(len(tokens), 1) or 1
    frequency = Counter(t for counts in tokens for t in counts)
    ranked = []
    for i, ((did, block), counts) in enumerate(zip(candidates, tokens)):
        length = sum(counts.values())
        score = 0.0
        for token in query & counts.keys():
            idf = math.log(1 + (len(tokens) - frequency[token] + .5) / (frequency[token] + .5))
            tf = counts[token]
            score += idf * tf * 2.2 / (tf + 1.2 * (.25 + .75 * length / average))
        if strategy == "structure":
            metadata = " ".join([documents[did]["title"], *block.get("section_path", [])])
            score += 2.0 * len(query & set(terms(metadata)))
        ranked.append((score, i, did, block))
    if strategy != "truncate":
        ranked.sort(key=lambda x: (-x[0], x[1]))
    remaining = budget_bytes
    evidence = []
    selected_text = ""
    selected_token_count = 0
    for score, _, did, block in ranked:
        if budget_tokens is None and remaining <= 0:
            break
        if strategy != "truncate" and score <= 0:
            continue
        if budget_tokens is not None:
            if length_counter is None:
                raise ValueError("token budgeting requires a real tokenizer length counter")
            prefix = selected_text + ("\n\n" if evidence else "")
            text = block["text"]
            candidate_tokens = length_counter(prefix + text)
            truncated_last_block = candidate_tokens > budget_tokens
            if truncated_last_block:
                low, high = 0, len(text)
                while low < high:
                    middle = (low + high + 1) // 2
                    if length_counter(prefix + text[:middle]) <= budget_tokens:
                        low = middle
                    else:
                        high = middle - 1
                text = text[:low]
            if not text:
                break
            selected_text = prefix + text
            selected_token_count = length_counter(selected_text) if truncated_last_block else candidate_tokens
            if selected_token_count > budget_tokens:
                raise ValueError("selected context exceeded the reference tokenizer budget")
        else:
            text = _truncate(block["text"], remaining)
        if not text:
            continue
        evidence.append({"document_id": did, "block_id": block["id"], "quote": text, "start_char": 0, "end_char": len(text)})
        remaining -= len(text.encode("utf-8"))
        if budget_tokens is not None and (truncated_last_block or selected_token_count >= budget_tokens):
            break
    return {"id": task["id"], "evidence": evidence, "status": "ok", "answer": None, "predicted_answerable": bool(evidence), "mode": "context_selection_only", "selected_bytes": sum(len(ev["quote"].encode("utf-8")) for ev in evidence), "selected_tokens": selected_token_count if budget_tokens is not None else None}


def _normalize(text: str) -> str:
    return " ".join(terms(text))


def answer_scores(answer: str, references: list[str]) -> dict:
    """Token F1 supports Chinese characters; punctuation/case are ignored."""
    scores = []
    for reference in references:
        pred, gold = terms(answer), terms(reference)
        overlap = sum((Counter(pred) & Counter(gold)).values())
        f1 = 2 * overlap / (len(pred) + len(gold)) if pred or gold else 1.0
        scores.append({"answer_em": float(_normalize(answer) == _normalize(reference)), "answer_f1": f1})
    return max(scores, key=lambda x: (x["answer_f1"], x["answer_em"]))


def _union_length(intervals: list[tuple[int, int]]) -> int:
    end = total = 0
    for start, stop in sorted(intervals):
        if stop > end:
            total += stop - max(start, end)
            end = stop
    return total


def _overlap_length(left: list[tuple[int, int]], right: list[tuple[int, int]]) -> int:
    return _union_length([(max(a, c), min(b, d)) for a, b in left for c, d in right if min(b, d) > max(a, c)])


def _align_quote_occurrences(gold_options: list[list[tuple[int, int]]], prediction_options: list[list[tuple[int, int]]]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Align ambiguous literal quotes to evidence, never duplicate one prediction.

    Offsets give a single option. Otherwise choose a complete matching occurrence
    with maximum span overlap. Coordinate refinement prevents an ambiguous quote
    from always being attached to its first mention. Identical ambiguous evidence
    entries are deduplicated, so repeated citations do not create extra coverage.
    """
    gold_options = list(dict.fromkeys(tuple(x) for x in gold_options))
    prediction_options = list(dict.fromkeys(tuple(x) for x in prediction_options))
    possible_gold = [span for options in gold_options for span in options]
    selected = [max(options, key=lambda span: (_overlap_length([span], possible_gold), -span[0])) for options in prediction_options]
    gold = [max(options, key=lambda span: (_overlap_length([span], selected), -span[0])) for options in gold_options]
    for _ in range(3):
        old = selected, gold
        revised = []
        for i, options in enumerate(prediction_options):
            others = revised + selected[i + 1:]
            revised.append(max(options, key=lambda span: (_overlap_length(others + [span], gold), -span[0])))
        selected = revised
        gold = [max(options, key=lambda span: (_overlap_length([span], selected), -span[0])) for options in gold_options]
        if old == (selected, gold):
            break
    return gold, selected


def score_prediction(task: dict, prediction: dict | None, documents: dict[str, dict], *, score_answers: bool = True) -> dict:
    row = {"id": task["id"], "task_type": task["task_type"], "annotation_status": task["annotation_status"], "language": task["language"], "split": task["split"], "status": "ok", "answer_scored": False, "evidence_scored": bool(task["evidence"]), "abstention_scored": False}
    if prediction is None:
        row.update(status="missing_prediction", error="No prediction with this ID")
        prediction = {}
    elif not isinstance(prediction, dict):
        row.update(status="invalid_prediction", error="prediction must be an object")
        prediction = {}
    elif not isinstance(prediction.get("status", "ok"), str) or prediction.get("status", "ok") not in {"ok", "success"}:
        row.update(status="failed_prediction", error=str(prediction.get("error", prediction.get("status"))))
    block_map = {(did, block["id"]): block for did in task["document_ids"] for block in documents[did]["blocks"]}
    selected_options: dict[tuple[str, str], list[list[tuple[int, int]]]] = defaultdict(list)
    evidence = prediction.get("evidence", [])
    if not isinstance(evidence, list):
        row.update(status="invalid_prediction", error="evidence must be a list")
        evidence = []
    answer = prediction.get("answer")
    if answer is not None and not isinstance(answer, str):
        row.update(status="invalid_prediction", error="answer must be a string or null")
    if "predicted_answerable" in prediction and not isinstance(prediction["predicted_answerable"], bool):
        row.update(status="invalid_prediction", error="predicted_answerable must be boolean")
    if "abstain" in prediction and not isinstance(prediction["abstain"], bool):
        row.update(status="invalid_prediction", error="abstain must be boolean")
    answerability = prediction.get("predicted_answerable")
    if isinstance(prediction.get("abstain"), bool):
        from_abstain = not prediction["abstain"]
        if isinstance(answerability, bool) and answerability != from_abstain:
            row.update(status="invalid_prediction", error="abstain contradicts predicted_answerable")
        elif answerability is None:
            answerability = from_abstain
    if answerability is False and (answer or evidence):
        row.update(status="invalid_prediction", error="abstention requires empty answer and evidence")
    if answerability is True and not evidence:
        row.update(status="invalid_prediction", error="answerable prediction requires supporting evidence")
    for ev in evidence:
        if not isinstance(ev, dict):
            row.update(status="invalid_prediction", error="evidence item must be an object")
            continue
        key = ev.get("document_id"), ev.get("block_id")
        if any(not isinstance(x, str) for x in key):
            row.update(status="invalid_prediction", error="predicted evidence IDs must be strings")
            continue
        block = block_map.get(key)
        if block is None:
            row.update(status="invalid_prediction", error=f"unknown/out-of-context predicted block {key}")
            continue
        quote = ev.get("quote", block["text"])
        if not isinstance(quote, str) or not quote or quote not in block["text"]:
            row.update(status="invalid_prediction", error=f"predicted quote not in block {key}")
            continue
        try:
            selected_options[key].append(quote_spans(block["text"], ev))
        except ValueError as exc:
            row.update(status="invalid_prediction", error=str(exc))
    gold_options: dict[tuple[str, str], list[list[tuple[int, int]]]] = defaultdict(list)
    for ev in task["evidence"]:
        key = ev["document_id"], ev["block_id"]
        gold_options[key].append(quote_spans(block_map[key]["text"], ev))
    gold, selected = {}, {}
    for key in set(gold_options) | set(selected_options):
        aligned_gold, aligned_predicted = _align_quote_occurrences(gold_options.get(key, []), selected_options.get(key, []))
        if aligned_gold:
            gold[key] = aligned_gold
        if aligned_predicted:
            selected[key] = aligned_predicted
    row["ambiguous_quote_entries"] = sum(len(options) > 1 for group in (*gold_options.values(), *selected_options.values()) for options in group)
    failed = row["status"] != "ok"
    if gold:
        hits = set()
        gold_chars = overlap_chars = predicted_chars = 0
        for key, intervals in gold.items():
            gold_chars += _union_length(intervals)
            intersections = [(max(a, c), min(b, d)) for a, b in intervals for c, d in selected.get(key, []) if min(b, d) > max(a, c)]
            overlap_chars += _union_length(intersections)
            if all(_union_length([(max(a, c), min(b, d)) for c, d in selected.get(key, []) if min(b, d) > max(a, c)]) >= b - a for a, b in intervals):
                hits.add(key)
        for intervals in selected.values():
            predicted_chars += _union_length(intervals)
        recall = len(hits) / len(gold) if not failed else 0.0
        precision = len(hits) / len(selected) if selected and not failed else 0.0
        row.update(evidence_recall=recall, evidence_precision=precision, evidence_complete_hit=float(len(hits) == len(gold) and not failed), evidence_span_recall=overlap_chars / gold_chars if not failed else 0.0, evidence_span_precision=overlap_chars / predicted_chars if predicted_chars and not failed else 0.0)
    if score_answers and task["task_type"] in ANSWER_TASKS:
        refs = task.get("reference_answers") or [task["reference_answer"]]
        if not all(isinstance(x, str) for x in refs):
            refs = [task["reference_answer"]]
        if isinstance(answer, str) and not failed:
            row.update(answer_scores(answer, refs), answer_scored=True)
        else:
            row.update(answer_em=0.0, answer_f1=0.0)
    predicted_answerable = prediction.get("predicted_answerable")
    if predicted_answerable is None and isinstance(prediction.get("abstain"), bool):
        predicted_answerable = not prediction["abstain"]
    if isinstance(predicted_answerable, bool):
        row.update(abstention_scored=not failed, answerability_accuracy=float(predicted_answerable == task["answerable"] and not failed))
        if not task["answerable"]:
            row["unanswerable_rejection_accuracy"] = float(not predicted_answerable and not failed)
        else:
            row["answerable_false_rejection"] = float(not predicted_answerable or failed)
    elif score_answers:
        row["answerability_accuracy"] = 0.0
        if not task["answerable"]:
            row["unanswerable_rejection_accuracy"] = 0.0
    row["prediction"] = prediction
    return row


METRICS = ("evidence_recall", "evidence_precision", "evidence_complete_hit", "evidence_span_recall", "evidence_span_precision", "answer_em", "answer_f1", "answerability_accuracy", "unanswerable_rejection_accuracy", "answerable_false_rejection")


def aggregate(rows: list[dict]) -> dict:
    result: dict[str, Any] = {"n": len(rows), "successful": sum(r["status"] == "ok" for r in rows), "failed_or_missing": sum(r["status"] != "ok" for r in rows)}
    result["prediction_coverage"] = result["successful"] / len(rows) if rows else 0.0
    for field in ("answer_scored", "evidence_scored", "abstention_scored"):
        count = sum(bool(r.get(field)) for r in rows)
        result[field + "_n"] = count
        result[field.replace("_scored", "_coverage")] = count / len(rows) if rows else 0.0
    for metric in METRICS:
        values = [r[metric] for r in rows if metric in r]
        result[metric] = sum(values) / len(values) if values else None
        result[metric + "_n"] = len(values)
    return result


def evaluate(documents: list[dict], tasks: list[dict], *, strategy: str = "bm25", budget_bytes: int = 16384, budget_tokens: int | None = None, predictions: list[dict] | None = None, length_counter: Callable[[str], int] | None = None, length_unit: str = "utf8_bytes") -> tuple[list[dict], dict]:
    if strategy not in {"bm25", "structure", "truncate"}:
        raise ValueError("strategy must be bm25, structure, or truncate")
    if budget_bytes < 1:
        raise ValueError("budget_bytes must be positive")
    if budget_tokens is not None and (budget_tokens < 1 or length_counter is None):
        raise ValueError("positive budget_tokens and a real tokenizer length counter are required together")
    validation = validate(documents, tasks)
    if not validation["valid"]:
        raise ValueError("; ".join(validation["errors"]))
    docs = {d["id"]: d for d in documents}
    if predictions is not None:
        pids = [p.get("id") for p in predictions]
        if len(pids) != len(set(pids)):
            raise ValueError("duplicate prediction IDs")
        unknown = set(pids) - {t["id"] for t in tasks}
        if unknown:
            raise ValueError("unknown prediction IDs: " + ", ".join(str(x) for x in unknown))
        pred_map = {p["id"]: p for p in predictions}
    else:
        pred_map = {}
    counter = length_counter or (lambda text: len(text.encode("utf-8")))
    rows = []
    for task in tasks:
        prediction = pred_map.get(task["id"]) if predictions is not None else select_context(task, docs, strategy, budget_bytes, budget_tokens=budget_tokens, length_counter=length_counter)
        row = score_prediction(task, prediction, docs, score_answers=predictions is not None)
        body = "\n".join(b["text"] for did in task["document_ids"] for b in docs[did]["blocks"])
        length = counter(body)
        bin_name = "<8k" if length < 8192 else "8k-32k" if length < 32768 else "32k-128k" if length < 131072 else ">=128k"
        domains = sorted({docs[did]["domain"] for did in task["document_ids"]})
        ev_docs = {ev["document_id"] for ev in task["evidence"]}
        if len(ev_docs) > 1:
            distance = "cross_document"
        elif len(task["evidence"]) < 2:
            distance = "single_or_no_evidence"
        else:
            did = next(iter(ev_docs))
            offsets, cursor = {}, 0
            block_texts = {}
            for block in docs[did]["blocks"]:
                offsets[block["id"]] = cursor
                block_texts[block["id"]] = block["text"]
                cursor += len(block["text"].encode("utf-8"))
            options = [quote_spans(block_texts[ev["block_id"]], ev) for ev in task["evidence"]]
            if any(len(spans) > 1 for spans in options):
                distance = "ambiguous_occurrence"
            else:
                points = [offsets[ev["block_id"]] + len(block_texts[ev["block_id"]][:spans[0][0]].encode("utf-8")) for ev, spans in zip(task["evidence"], options)]
                gap = max(points) - min(points)
                distance = "<8k_bytes" if gap < 8192 else "8k-32k_bytes" if gap < 32768 else ">=32k_bytes"
        row.update(domain="+".join(domains), context_length=length, length_unit=length_unit, length_bucket=bin_name, evidence_distance_bucket=distance)
        rows.append(row)
    groups = {}
    for key in ("annotation_status", "language", "domain", "length_bucket", "evidence_distance_bucket", "task_type", "split"):
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            grouped[row[key]].append(row)
        groups[key] = {name: aggregate(items) for name, items in sorted(grouped.items())}
    summary = {"schema_version": 1, "mode": "imported_predictions" if predictions is not None else "offline_context_selection_only", "strategy": strategy if predictions is None else None, "budget_bytes": budget_bytes if predictions is None and budget_tokens is None else None, "budget_tokens": budget_tokens if predictions is None else None, "length_unit": length_unit, "overall": aggregate(rows), "groups": groups, "warnings": validation["warnings"], "resolved_splits": validation["resolved_splits"], "limitations": ["Offline context selection does not generate or evaluate model answers.", "Controlled synthetic location checks are not human-verified reasoning gold.", "Evidence block hits require full gold quote coverage; span metrics use Unicode character offsets.", "Context budget excludes metadata and prompt formatting; token budgets include inter-block separators.", "Human-verified, AI-generated and unreviewed labels are reported separately.", "Inferred family splits are deterministic and may be imbalanced in small datasets."]}
    summary["limitations"].extend([
        "Candidates are restricted to task.document_ids: within-given-document/group evidence retrieval, not full-corpus document retrieval.",
        "The structure strategy adds title/section_path lexical weighting to BM25; it does not expand neighboring blocks.",
        "Controlled synthetic quote-location questions expose the exact quotations and do not establish reasoning capability.",
    ])
    return rows, summary
