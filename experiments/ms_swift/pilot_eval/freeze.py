"""Freeze the exploratory public suite before any model outputs are observed.

Records are selected by seeded hashes of IDs, never answers or model scores.
LongBench primary tasks require >=8K native chat input and total budget <=16K.
Exact context / shared substantial paragraph components are withheld together;
this bounded audit is not proof of absence of every semantic contamination.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import string
import zipfile

from cpt.core import digest, file_hash, new_output, read_jsonl, write_json, write_jsonl
from gemma_eval.chat import encode_chat, PROTOCOL
from gemma_eval.core import tokenizer_assets
from .acquire import RULER_REV, BOOLQ_REV, TASKS
from .ruler_lineage import annotate_ruler_qa

LB_REV = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
LB_CODE = "2e00731f8d0bff23dc4325161044d0ed8af94c1e"
LB_ZIP_SHA = "cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64"
PROMPT_SHA = "56d22ad4f382169c2b8a11ff4c982a4a1bea096c8152b0f0b85b64686b157c30"
MAXLEN_SHA = "72966b3c0933e214591637fb085798c5e687ebff4ddaab5d99bbc31120532022"
PRIMARY = ("hotpotqa", "multifieldqa_en", "narrativeqa", "gov_report")
SECONDARY = ("qasper", "multifieldqa_zh", "multi_news", "2wikimqa")
SEED = 17011


def normalized(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def fingerprint(text):
    return hashlib.sha256(normalized(text).encode()).hexdigest()


def rank(identifier):
    return hashlib.sha256(f"{SEED}:{identifier}".encode()).hexdigest()


def length_bucket(length):
    for cap in (2048, 4096, 8192, 16384, 32768):
        if length <= cap:
            return f"<={cap}"
    return ">32768"


def paragraphs(text):
    # Shared long complete paragraphs are strong source overlap evidence, unlike
    # short headers or common boilerplate. Exact context equality is also tested.
    return {fingerprint(p) for p in re.split(r"\n\s*\n", text) if len(normalized(p)) >= 300}


def source_components(rows):
    parent = list(range(len(rows)))
    owners = {}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for index, row in enumerate(rows):
        for key in paragraphs(row["context"]) | {fingerprint(row["context"])}:
            if key in owners:
                a, b = find(index), find(owners[key])
                parent[max(a, b)] = min(a, b)
            else:
                owners[key] = index
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[find(index)].append(row["id"])
    ids = {i: "longbench-context-component:" + digest(sorted(group))[:24] for i, group in groups.items()}
    for index, row in enumerate(rows):
        row["cluster_id"] = ids[find(index)]


def choose(rows, count):
    # One pilot record per source family; all siblings are held out of confirmation.
    chosen, used = [], set()
    for row in sorted(rows, key=lambda r: rank(r["id"])):
        if row["cluster_id"] not in used:
            chosen.append(row)
            used.add(row["cluster_id"])
            if len(chosen) == count:
                return chosen
    raise ValueError(f"need {count} independent families; available {len(chosen)}")


def decorate(row, tokenizer):
    row.update(evaluation_only=True, split_role="exploratory_evaluation_only_never_training",
               prompt_sha256=hashlib.sha256(row["prompt"].encode()).hexdigest())
    row["native_input_tokens"] = len(encode_chat(tokenizer, row["prompt"]))
    row["total_budget_tokens"] = row["native_input_tokens"] + row["max_new_tokens"]
    row["length_bucket"] = length_bucket(row["native_input_tokens"])
    return row


def freeze(*, cache, longbench_cache, model_assets, training_sources, output):
    cache, longbench_cache, model_assets = map(lambda p: Path(p).resolve(), (cache, longbench_cache, model_assets))
    out = new_output(output)
    tokenizer, tokenizer_metadata = tokenizer_assets(model_assets)
    source_file = longbench_cache / "raw/data.zip"
    prompt_file = longbench_cache / "official/LongBench/config/dataset2prompt.json"
    maxlen_file = longbench_cache / "official/LongBench/config/dataset2maxlen.json"
    for path, expected in ((source_file, LB_ZIP_SHA), (prompt_file, PROMPT_SHA), (maxlen_file, MAXLEN_SHA)):
        if file_hash(path) != expected:
            raise ValueError(f"pinned LongBench source hash mismatch: {path.name}")
    templates = json.loads(prompt_file.read_text(encoding="utf-8"))
    limits = json.loads(maxlen_file.read_text(encoding="utf-8"))
    all_lb = []
    with zipfile.ZipFile(source_file) as archive:
        for task in PRIMARY + SECONDARY:
            fields = {field for _, field, _, _ in string.Formatter().parse(templates[task]) if field is not None}
            if fields - {"context", "input"}:
                raise ValueError("official template would interpolate non-input fields")
            name = next(name for name in archive.namelist() if name in (f"data/{task}.jsonl", f"{task}.jsonl"))
            for index, line in enumerate(archive.read(name).decode("utf-8").splitlines()):
                raw = json.loads(line)
                row = {"id": f"longbench-v1:{task}:{raw['_id']}", "task": task, "benchmark": "longbench-v1",
                       "context": raw["context"], "answers": raw["answers"], "raw_record": raw,
                       "raw_record_sha256": digest(raw), "source_index": index, "official_id": raw["_id"],
                       "source_provenance": {"repo": "zai-org/LongBench", "revision": LB_REV, "archive_sha256": LB_ZIP_SHA},
                       "language": raw.get("language", "en"), "max_new_tokens": limits[task],
                       "prompt": templates[task].format(context=raw["context"], input=raw["input"]),
                       "score_kind": "longbench_v1", "metric_name": "rouge_score" if task in ("gov_report", "multi_news") else "qa_f1_zh_score" if task == "multifieldqa_zh" else "qa_f1_score",
                       "pilot_role": "primary" if task in PRIMARY else "secondary"}
                all_lb.append(decorate(row, tokenizer))
    source_components(all_lb)
    # Source-level input provenance is available before any inference. Exact full
    # documents and shared complete paragraphs are checked against admitted-candidate
    # sources; the report explicitly limits this audit's semantic coverage.
    sources = read_jsonl(training_sources)
    train_hashes = set()
    for source in sources:
        text = "\n\n".join(block.get("cleaned_text", block["text"]) for block in source["blocks"])
        train_hashes |= paragraphs(text) | {fingerprint(text)}
        train_hashes |= {fingerprint(b.get("cleaned_text", b["text"])) for b in source["blocks"] if len(normalized(b.get("cleaned_text", b["text"]))) >= 300}
    quarantined = {r["cluster_id"] for r in all_lb if (paragraphs(r["context"]) | {fingerprint(r["context"])}) & train_hashes}
    lb_selected = []
    eligibility = {}
    for task in PRIMARY + SECONDARY:
        already_selected = {r["cluster_id"] for r in lb_selected}
        candidates = [r for r in all_lb if r["task"] == task and r["cluster_id"] not in quarantined
                      and r["cluster_id"] not in already_selected
                      and r["total_budget_tokens"] <= 16384
                      and (task not in PRIMARY or r["native_input_tokens"] >= 8192)]
        eligibility[task] = {"eligible_records": len(candidates), "eligible_clusters": len({r["cluster_id"] for r in candidates})}
        lb_selected.extend(choose(candidates, 5 if task in PRIMARY else 10))
    pilot_clusters = {r["cluster_id"] for r in lb_selected}
    reserve = [{"id": r["id"], "task": r["task"], "cluster_id": r["cluster_id"],
                "raw_record_sha256": r["raw_record_sha256"], "native_input_tokens": r["native_input_tokens"],
                "total_budget_tokens": r["total_budget_tokens"], "eligible_primary_16k": 8192 <= r["native_input_tokens"] and r["total_budget_tokens"] <= 16384}
               for r in all_lb if r["cluster_id"] not in pilot_clusters | quarantined]
    selected = list(lb_selected)
    acquisition = json.loads((cache / "acquisition_manifest.json").read_text())
    ruler_assets = cache / f"RULER-{RULER_REV}/scripts/data/synthetic/json"
    for name, expected in (("english_words.json", acquisition["english_words_sha256"]),
                           ("PaulGrahamEssays.json", acquisition["essay_haystack_sha256"]),
                           ("squad.json", acquisition["squad"]["sha256"])):
        if file_hash(ruler_assets / name) != expected:
            raise ValueError(f"RULER acquired generator asset changed: {name}")
    ruler_manifest = json.loads((cache / "ruler_generation_manifest.json").read_text())
    if (ruler_manifest.get("revision") != RULER_REV
            or ruler_manifest.get("acquisition_manifest_sha256") != file_hash(cache / "acquisition_manifest.json")
            or acquisition["ruler"]["revision"] != RULER_REV):
        raise ValueError("RULER generation and acquisition locks disagree")
    input_lock_path = Path(ruler_manifest["input_lock_path"])
    if file_hash(input_lock_path) != ruler_manifest["input_lock_sha256"]:
        raise ValueError("RULER immutable generation input lock changed")
    grid = [(s["task"], s["length"]) for s in ruler_manifest["generation"]]
    expected_grid = {(task, length) for task in TASKS for length in (4096, 8192, 16384)}
    if len(grid) != len(expected_grid) or set(grid) != expected_grid:
        raise ValueError("RULER task/length grid differs from design")
    for spec in ruler_manifest["generation"]:
        path = Path(spec["path"])
        generated_rows = read_jsonl(path)
        if (file_hash(path) != spec["sha256"] or spec["seed"] != SEED
                or spec["input_lock_sha256"] != ruler_manifest["input_lock_sha256"]
                or spec["records"] != 10 or len(generated_rows) != 10):
            raise ValueError("RULER generation changed after its receipt")
        for index, raw in enumerate(generated_rows):
            task, target = spec["task"], spec["length"]
            row = {"id": f"ruler:{task}:{target}:{index}", "benchmark": "ruler", "task": task,
                   "cluster_id": f"ruler-seed17011:{task}:{index}", "answers": raw["outputs"],
                   "prompt": raw["input"] + raw.get("answer_prefix", ""), "max_new_tokens": spec["output_tokens"],
                   "target_context_tokens": target, "raw_record": raw, "raw_record_sha256": digest(raw),
                   "score_kind": "ruler_any" if task == "qa_1" else "ruler_all", "language": "en",
                   "source_provenance": {"repo": "NVIDIA/RULER", "revision": RULER_REV, "generation_sha256": spec["sha256"]},
                   "pilot_role": "diagnostic"}
            if task == "qa_1":
                row.update(annotate_ruler_qa(raw, cache / f"RULER-{RULER_REV}/scripts/data/synthetic/json/squad.json"))
            decorate(row, tokenizer)
            if row["total_budget_tokens"] > target:
                raise ValueError(f"RULER {row['id']} exceeds native chat budget; regenerate, never truncate")
            selected.append(row)
    boolq_rows = []
    if (file_hash(cache / "boolq_dev.jsonl") != acquisition["boolq"]["jsonl_sha256"]
            or file_hash(cache / "boolq_validation.parquet") != acquisition["boolq"]["sha256"]
            or acquisition["boolq"]["revision"] != BOOLQ_REV or acquisition["boolq"]["records"] != 3270):
        raise ValueError("BoolQ raw/derived acquisition locks disagree")
    for index, raw in enumerate(read_jsonl(cache / "boolq_dev.jsonl")):
        prompt = f"Read the passage and answer the question. Reply with exactly Yes or No.\n\nPassage: {raw['passage']}\n\nQuestion: {raw['question']}\nAnswer:"
        row = {"id": f"boolq:validation:{index}", "benchmark": "boolq", "task": "yes_no",
               "cluster_id": "boolq-passage:" + fingerprint(raw["passage"]), "answers": ["yes" if raw["answer"] else "no"],
               "prompt": prompt, "max_new_tokens": 8, "score_kind": "boolq_yes_no", "language": "en",
               "raw_record": raw, "raw_record_sha256": digest(raw), "pilot_role": "short_regression_control",
               "source_provenance": {"repo": "google/boolq", "revision": BOOLQ_REV, "split": "validation"}}
        decorate(row, tokenizer)
        if row["total_budget_tokens"] <= 2048 and not (paragraphs(raw["passage"]) | {fingerprint(raw["passage"])}) & train_hashes:
            boolq_rows.append(row)
    selected_boolq = choose(boolq_rows, 40)
    selected.extend(selected_boolq)
    boolq_clusters = {r["cluster_id"] for r in selected_boolq}
    reserve_boolq = [{"id": r["id"], "cluster_id": r["cluster_id"], "raw_record_sha256": r["raw_record_sha256"]}
                     for r in boolq_rows if r["cluster_id"] not in boolq_clusters]
    if len(selected) != 250 or len({r["id"] for r in selected}) != 250:
        raise ValueError("frozen pilot count is not 250")
    write_jsonl(out / "records.jsonl", selected)
    write_jsonl(out / "confirmation_reserve_ids.jsonl", reserve + reserve_boolq)
    for name in ("acquisition_manifest.json", "ruler_generation_manifest.json"):
        shutil.copyfile(cache / name, out / name)
    shutil.copyfile(input_lock_path, out / "ruler_generation.inputs.json")
    provenance = [
        {"benchmark": "longbench-v1", "dataset_repo": "zai-org/LongBench", "revision": LB_REV,
         "raw_sha256": LB_ZIP_SHA, "code_revision": LB_CODE, "prompt_sha256": PROMPT_SHA, "maxlen_sha256": MAXLEN_SHA,
         "score_source_sha256": file_hash(longbench_cache / "official/LongBench/metrics.py")},
        {"benchmark": "ruler", "repo": "NVIDIA/RULER", "revision": RULER_REV,
         "acquisition_sha256": file_hash(cache / "acquisition_manifest.json"),
         "generation_sha256": file_hash(cache / "ruler_generation_manifest.json"),
         "score_source_sha256": file_hash(cache / f"RULER-{RULER_REV}/scripts/eval/synthetic/constants.py")},
        {"benchmark": "boolq", "repo": "google/boolq", "revision": BOOLQ_REV,
         "raw_sha256": file_hash(cache / "boolq_validation.parquet"), "license": "CC-BY-SA-3.0",
         "scoring_protocol": "custom generative whole-answer yes/no; whitespace/terminal punctuation permitted; explanations score zero"}]
    manifest = {"schema_version": "gemma-eval-input-v1", "records_sha256": file_hash(out / "records.jsonl"),
                "records": len(selected), "selected_ids": [r["id"] for r in selected], "source_manifests": provenance,
                "selection_seed": SEED, "selection": {
                    "longbench": "fixed task order, SHA256(seed:id) order; one example per source component across tasks",
                    "boolq": "SHA256(seed:id) order; one example per identical passage component",
                    "ruler": "unchanged official generators; seed17011; ten examples per task/length; qa_1 first ten answerable SQuAD questions, clustered by article",
                    "answer_or_model_score_dependent_selection": False},
                "protocol": PROTOCOL, "tokenizer": tokenizer_metadata,
                "pilot_only": True, "official_full_run": False,
                "confirmation_reserve_sha256": file_hash(out / "confirmation_reserve_ids.jsonl"),
                "training_sources_sha256": file_hash(training_sources), "scorer_sha256": file_hash(Path(__file__).with_name("scoring.py"))}
    write_json(out / "eval_manifest.json", manifest)
    summary = {"schema_version": "public-pilot-freeze-v1", "records": len(selected),
               "by_benchmark": dict(Counter(r["benchmark"] for r in selected)),
               "by_task": dict(Counter(r["task"] for r in selected)),
               "by_length_bucket": dict(Counter(r["length_bucket"] for r in selected)),
               "native_input_range": [min(r["native_input_tokens"] for r in selected), max(r["native_input_tokens"] for r in selected)],
               "all_fit_16k_total_budget": all(r["total_budget_tokens"] <= 16384 for r in selected),
               "primary_longbench_input_minimum": min(r["native_input_tokens"] for r in lb_selected if r["pilot_role"] == "primary"),
               "longbench_eligibility": eligibility,
               "confirmation_primary_eligible_by_task": dict(Counter(r["task"] for r in reserve if r["eligible_primary_16k"])),
               "reserved_boolq_records": len(reserve_boolq), "quarantined_overlap_clusters": sorted(quarantined),
               "ruler_qa_source_articles": sorted({r["source_article"] for r in selected if r.get("source_article")}),
               "ruler_qa_independent_clusters": len({r["cluster_id"] for r in selected if r["benchmark"] == "ruler" and r["task"] == "qa_1"}),
               "overlap_audit_scope": "exact normalized contexts and complete paragraphs/blocks >=300 chars; not an exhaustive semantic or pretraining-contamination audit",
               "model_predictions_observed": 0, "records_sha256": manifest["records_sha256"],
               "source_manifest_sha256": file_hash(out / "eval_manifest.json"), "protocol": PROTOCOL["version"]}
    write_json(out / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "longbench-cache", "model-assets", "training-sources", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    print(json.dumps(freeze(**vars(parser.parse_args())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
