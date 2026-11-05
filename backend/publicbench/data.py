"""Fetch immutable LongBench snapshots without executing dataset/source scripts.

The official raw values remain in ``raw_record``; examples are evaluation-only.
Sampling (including optional length eligibility) is fixed before any inference.
"""
from __future__ import annotations

import hashlib
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterator
from zipfile import ZipFile

import requests

OFFICIAL_CODE_REVISION = "2e00731f8d0bff23dc4325161044d0ed8af94c1e"
OFFICIAL_CODE_REPO = "THUDM/LongBench"
DEFAULT_V1_TASKS = ("qasper", "hotpotqa", "multifieldqa_zh", "gov_report", "trec", "passage_retrieval_en")
V2_PLACEHOLDERS = ("$DOC$", "$Q$", "$C_A$", "$C_B$", "$C_C$", "$C_D$")
SNAPSHOTS = {
    "longbench-v1": {"repository": "zai-org/LongBench", "revision": "5e628be450b7e67fb7ae6e201bd6d8f7056f7672",
                     "filename": "data.zip", "sha256": "cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64",
                     "license": "not-unified-upstream-per-task", "card_license": None},
    "longbench-v2": {"repository": "zai-org/LongBench-v2", "revision": "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9",
                     "filename": "data.json", "sha256": "15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2",
                     "license": "Apache-2.0", "card_license": "apache-2.0"},
}
OFFICIAL_ASSETS = {
    "LICENSE": "bf7f05f274d65931bdf14f060b6855f1c359b8424eb74fd3091d25de89d700c3",
    "LongBench/README.md": "1e16429a4c8ef3da202855b9e1242943d3987b09b64fc7b8a1c7725a445ab8d5",
    "LongBench/metrics.py": "e22e2a2662e0f7e683137fa3541f64edb6a801e9138d16d2f3459a6ab9941323",
    "LongBench/eval.py": "1a3acfc25d9b053e9bb75c479f7e385d0cb9989f0f7115346b7d632655967721",
    "LongBench/config/dataset2prompt.json": "56d22ad4f382169c2b8a11ff4c982a4a1bea096c8152b0f0b85b64686b157c30",
    "LongBench/config/dataset2maxlen.json": "72966b3c0933e214591637fb085798c5e687ebff4ddaab5d99bbc31120532022",
    "pred.py": "ab63f77866a1c0dc770582bc3fe6b014c3b4be4667399b0ee267075780c6a138",
    "result.py": "b6675c1c0b72090c01e12c94d3b176c4991fe38ceed9a301d24d22b1997465bd",
    "prompts/0shot.txt": "68a162252bc9ff71d5d7abca3d69bb31aac3c35f832d657a2866f2018b8a6950",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _file_sha(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _snapshot_sources() -> dict[str, str]:
    return {p.name: _file_sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def _ensure(path: Path, url: str, expected_sha256: str | None, allow_download: bool) -> dict:
    """Only inert public bytes are fetched; no auth token or remote code is used."""
    if not path.exists():
        if not allow_download:
            raise ValueError(f"missing cached artifact (downloads disabled): {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".download-part")
        if temporary.exists():
            raise ValueError(f"another download or incomplete artifact exists: {temporary}")
        try:
            with requests.get(url, timeout=(15, 180), stream=True) as response:
                response.raise_for_status()
                with temporary.open("xb") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
            if expected_sha256 and _file_sha(temporary) != expected_sha256:
                raise ValueError(f"download hash mismatch: {url}")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    actual = _file_sha(path)
    if expected_sha256 and actual != expected_sha256:
        raise ValueError(f"cached artifact hash mismatch: {path}")
    return {"url": url, "sha256": actual, "bytes": path.stat().st_size, "local_path": str(path.resolve())}


def normalize_benchmark(name: str) -> str:
    aliases = {"v1": "longbench-v1", "longbench": "longbench-v1", "longbench-v1": "longbench-v1",
               "v2": "longbench-v2", "longbench-v2": "longbench-v2"}
    try:
        return aliases[name.lower()]
    except (KeyError, AttributeError) as exc:
        raise ValueError("benchmark must be v1/longbench-v1 or v2/longbench-v2") from exc


def _assets(cache_dir: Path, allow_download: bool) -> tuple[dict, dict, str, dict]:
    provenance = {}
    for relative, expected in OFFICIAL_ASSETS.items():
        url = f"https://raw.githubusercontent.com/{OFFICIAL_CODE_REPO}/{OFFICIAL_CODE_REVISION}/{relative}"
        provenance[relative] = _ensure(cache_dir / "official" / relative, url, expected, allow_download)
    prompts = json.loads((cache_dir / "official/LongBench/config/dataset2prompt.json").read_text(encoding="utf-8"))
    lengths = json.loads((cache_dir / "official/LongBench/config/dataset2maxlen.json").read_text(encoding="utf-8"))
    v2_prompt = (cache_dir / "official/prompts/0shot.txt").read_text(encoding="utf-8")
    return prompts, lengths, v2_prompt, provenance


def iter_json_array(path: Path) -> Iterator[dict]:
    """Stream the 465MB official v2 JSON array without loading it all at once."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8-sig") as stream:
        buffer, cursor, eof = "", 0, False

        def more() -> bool:
            nonlocal buffer, cursor, eof
            chunk = stream.read(1024 * 1024)
            buffer = buffer[cursor:] + chunk
            cursor = 0
            eof = not chunk
            return bool(chunk)

        def nonspace() -> str:
            nonlocal cursor
            while True:
                while cursor < len(buffer) and buffer[cursor].isspace():
                    cursor += 1
                if cursor < len(buffer):
                    return buffer[cursor]
                if not more():
                    return ""

        if nonspace() != "[":
            raise ValueError("LongBench v2 data.json must contain a JSON array")
        cursor += 1
        first = True
        while True:
            token = nonspace()
            if token == "]":
                cursor += 1
                if nonspace():
                    raise ValueError("unexpected bytes after benchmark JSON array")
                return
            if not first:
                if token != ",":
                    raise ValueError("expected comma between benchmark records")
                cursor += 1
                if nonspace() == "]":
                    raise ValueError("trailing comma in benchmark JSON array")
            while True:
                try:
                    row, end = decoder.raw_decode(buffer, cursor)
                    cursor = end
                    break
                except json.JSONDecodeError as exc:
                    if eof or not more():
                        raise ValueError("invalid/incomplete benchmark JSON array") from exc
            if not isinstance(row, dict):
                raise ValueError("benchmark records must be JSON objects")
            yield row
            first = False


def _v1_records(path: Path, tasks: list[str]) -> Iterator[tuple[str, int, dict]]:
    # Never extract an archive: traversal names, scripts and unrelated artifacts
    # are inert, and only explicitly selected task JSONL members are opened.
    with ZipFile(path) as archive:
        for task in tasks:
            members = [n for n in archive.namelist() if n == f"data/{task}.jsonl" or n == f"{task}.jsonl"]
            if len(members) != 1:
                raise ValueError(f"expected one official JSONL member for task {task}, found {len(members)}")
            info = archive.getinfo(members[0])
            if info.file_size > 1024 * 1024 * 1024:
                raise ValueError("benchmark JSONL member exceeds 1GB safety bound")
            with archive.open(members[0]) as stream:
                for index, line in enumerate(stream):
                    if line.strip():
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise ValueError(f"{task}:{index}: record is not an object")
                        yield task, index, row


def render_prompt(example: dict, context_override: str | None = None) -> str:
    """Render only whitelisted official input fields, never reference answers."""
    context = example["context"] if context_override is None else context_override
    if not isinstance(context, str):
        raise ValueError("context_override must be a string")
    template = example["prompt_template"]
    if example["benchmark"] == "longbench-v1":
        fields = {field for _, field, _, _ in string.Formatter().parse(template) if field is not None}
        if fields - {"context", "input"}:
            raise ValueError("official v1 template contains non-input/answer fields")
        return template.format(context=context, input=example["question"])
    if example["benchmark"] != "longbench-v2":
        raise ValueError("unknown benchmark in prompt record")
    # Preserve the official pred.py replacement order even when literal tokens
    # inside the source are subsequently rewritten. Raw records stay unchanged;
    # the potential second substitutions are explicitly audited below.
    prompt = template.replace("$DOC$", context.strip()).replace("$Q$", example["question"].strip())
    for choice in "ABCD":
        prompt = prompt.replace(f"$C_{choice}$", example[f"choice_{choice}"].strip())
    return prompt


def _example(raw: dict, benchmark: str, task: str, index: int, spec: dict,
             prompts: dict, lengths: dict, v2_prompt: str) -> dict:
    from .metrics import TASK_METRICS

    if not isinstance(raw.get("_id"), str) or not raw["_id"]:
        raise ValueError("official benchmark record requires its original nonempty _id")
    if not isinstance(raw.get("context"), str) or not raw["context"]:
        raise ValueError(f"{raw['_id']}: missing original context")
    if benchmark == "longbench-v1":
        if task not in prompts or task not in lengths or task not in TASK_METRICS:
            raise ValueError(f"no pinned official prompt/maxlen/metric for task {task}")
        if raw.get("dataset") != task or not isinstance(raw.get("input"), str):
            raise ValueError(f"{raw['_id']}: official v1 task/input mismatch")
        answers = raw.get("answers")
        if not isinstance(answers, list) or not answers or any(not isinstance(a, str) for a in answers):
            raise ValueError("official v1 answers must be a nonempty string list")
        question, domain, metric = raw["input"], task, TASK_METRICS[task]
        template, maximum = prompts[task], lengths[task]
        reported_unit = "characters_for_Chinese_words_for_English_official_input_context_answers_length"
        classes = raw.get("all_classes")
        if classes is not None and (not isinstance(classes, list) or any(not isinstance(c, str) for c in classes)):
            raise ValueError("official all_classes must be null or a string list")
    else:
        for field in ("domain", "sub_domain", "question", "choice_A", "choice_B", "choice_C", "choice_D"):
            if not isinstance(raw.get(field), str) or not raw[field]:
                raise ValueError(f"{raw['_id']}: missing official v2 {field}")
        if raw.get("answer") not in {"A", "B", "C", "D"}:
            raise ValueError("official v2 answer must be A/B/C/D")
        answers = [raw["answer"]]
        question, domain, metric = raw["question"], raw["domain"], "accuracy_official_v2_extraction"
        template, maximum, classes = v2_prompt, 128, None
        reported_unit = "official_length_category_short_medium_long_not_model_tokens"
    example = {"id": f"{benchmark}:{task}:{raw['_id']}", "official_id": raw["_id"], "benchmark": benchmark,
               "task": task, "domain": domain, "language": raw.get("language", "en"),
               "context": raw["context"], "question": question, "input": question,
               "answers": list(answers), "reference_answers": list(answers), "all_classes": classes,
               "prompt_template": template, "official_max_new_tokens": maximum, "metric_name": metric,
               "source_repository": spec["repository"], "source_revision": spec["revision"],
               "source_file": spec["filename"], "source_file_sha256": spec["sha256"], "source_index": index,
               "raw_record": raw, "raw_record_sha256": _sha(_canonical(raw)), "raw_record_hash_kind": "canonical_JSON_values",
               "benchmark_reported_length": raw.get("length"), "benchmark_reported_length_unit": reported_unit,
               "license": spec["license"], "license_scope": "dataset_card_or_upstream_task_not_official_code_license",
               "evaluation_only": True, "official_source_split": "test" if benchmark == "longbench-v1" else "train",
               "split_role": "evaluation_only_never_training", "official_full_run": False,
               "prompt_protocol": "official_task_template_input_fields_only", "official_code_revision": OFFICIAL_CODE_REVISION}
    if benchmark == "longbench-v2":
        example.update({field: raw[field] for field in ("sub_domain", "difficulty", "length", "choice_A", "choice_B", "choice_C", "choice_D")})
        counts = {token: raw["context"].count(token) for token in V2_PLACEHOLDERS}
        affected = sum(counts[token] for token in V2_PLACEHOLDERS[1:])
        example["prompt_rendering"] = {
            "method": "official_pred_py_strip_and_sequential_replace_DOC_Q_A_B_C_D",
            "source_context_placeholder_literal_counts": counts,
            "potential_secondary_context_substitutions": affected,
            "warning": "Official sequential replacements can rewrite these literal source-context placeholders; raw_record is unchanged." if affected else None}
    example["prompt"] = render_prompt(example)
    example["prompt_sha256"] = _sha(example["prompt"].encode("utf-8"))
    return example


def fetch_benchmark(benchmark: str, cache_dir: str | Path, *, tasks: list[str] | tuple[str, ...] | None = None,
                    n_per_task: int = 5, seed: int = 13, revision: str | None = None, v2_stratify: bool = True,
                    max_context_tokens: int | None = None, length_counter: Callable[[str], int] | None = None,
                    tokenizer_metadata: dict | None = None, allow_download: bool = True) -> tuple[list[dict], dict]:
    """Return a fixed source-ID sample and its provenance/eligibility manifest.

    For v2 ``n_per_task`` means per official domain when ``v2_stratify=True``.
    No failed/oversized model case is resampled later. The optional token filter
    measures source context only with the caller's reference tokenizer.
    """
    implementation = _snapshot_sources()
    benchmark = normalize_benchmark(benchmark)
    if type(n_per_task) is not int or n_per_task < 1:
        raise ValueError("n_per_task must be a positive integer")
    if max_context_tokens is not None and (max_context_tokens < 1 or length_counter is None):
        raise ValueError("max_context_tokens requires a positive value and an explicit local tokenizer counter")
    spec = SNAPSHOTS[benchmark]
    if revision is not None and revision != spec["revision"]:
        raise ValueError("this benchmark loader supports only its pinned, hash-verified snapshot revision")
    cache = Path(cache_dir)
    url = f"https://huggingface.co/datasets/{spec['repository']}/resolve/{spec['revision']}/{spec['filename']}"
    source = _ensure(cache / "raw" / spec["filename"], url, spec["sha256"], allow_download)
    source_sidecar = (cache / "raw" / spec["filename"]).with_name(spec["filename"] + ".source.json")
    if source_sidecar.exists():
        declared = json.loads(source_sidecar.read_text(encoding="utf-8"))
        required = {"repo": spec["repository"], "revision": spec["revision"],
                    "filename": spec["filename"], "sha256": source["sha256"]}
        if any(declared.get(key) != value for key, value in required.items()):
            raise ValueError(f"cached source sidecar does not match pinned repository/revision/hash: {source_sidecar}")
        if declared.get("source_url", url) != url or declared.get("bytes", source["bytes"]) != source["bytes"]:
            raise ValueError("cached source sidecar URL/size does not match the pinned artifact")
        source["sidecar_sha256"] = _file_sha(source_sidecar)
    readme_url = f"https://huggingface.co/datasets/{spec['repository']}/resolve/{spec['revision']}/README.md"
    card = _ensure(cache / "raw" / f"{benchmark}.README.md", readme_url, None, allow_download)
    prompts, lengths, v2_prompt, official_assets = _assets(cache, allow_download)
    chosen_tasks = list(tasks) if tasks is not None else list(DEFAULT_V1_TASKS) if benchmark == "longbench-v1" else None
    if chosen_tasks is not None and (not chosen_tasks or len(set(chosen_tasks)) != len(chosen_tasks)):
        raise ValueError("tasks/domain selection must be nonempty and contain no duplicates")
    if benchmark == "longbench-v1":
        unsupported = set(chosen_tasks or []) - set(prompts)
        if unsupported:
            raise ValueError("unknown official v1 tasks: " + ", ".join(sorted(unsupported)))
        iterator = _v1_records(cache / "raw" / spec["filename"], chosen_tasks or [])
    else:
        iterator = (("longbench-v2", i, raw) for i, raw in enumerate(iter_json_array(cache / "raw" / spec["filename"])))
    before, eligible, rejected = Counter(), Counter(), Counter()
    pools: dict[str, list[dict]] = defaultdict(list)
    seen = set()
    for task, index, raw in iterator:
        example = _example(raw, benchmark, task, index, spec, prompts, lengths, v2_prompt)
        group = task if benchmark == "longbench-v1" else example["domain"] if v2_stratify else "all"
        if benchmark == "longbench-v2" and chosen_tasks is not None and example["domain"] not in chosen_tasks:
            continue
        if example["id"] in seen:
            raise ValueError("duplicate official source ID: " + example["id"])
        seen.add(example["id"])
        before[group] += 1
        count = length_counter(example["context"]) if max_context_tokens is not None else None
        if count is not None:
            if type(count) is not int or count < 0:
                raise ValueError("reference tokenizer counter must return nonnegative integers")
            example["reference_context_tokens"] = count
        if count is not None and count > max_context_tokens:
            rejected[group] += 1
            continue
        eligible[group] += 1
        pools[group].append(example)
    if not before:
        raise ValueError("no official source records match the selection")
    if benchmark == "longbench-v2" and chosen_tasks is not None:
        absent = set(chosen_tasks) - {d["domain"] for group in pools.values() for d in group}
        if absent:
            raise ValueError("requested v2 domains have no eligible records: " + ", ".join(sorted(absent)))
    selected = []
    for group in sorted(before):
        if not pools[group]:
            continue
        ordered = sorted(pools[group], key=lambda e: (_sha(f"{seed}:{benchmark}:{group}:{e['official_id']}".encode()), e["official_id"]))
        selected.extend(ordered[:n_per_task])
    if not selected:
        raise ValueError("no source records pass pre-inference eligibility")
    manifest = {"schema_version": "publicbench-sample-v1", "benchmark": benchmark, "source": source,
                "source_repository": spec["repository"], "source_revision": spec["revision"], "source_card": card,
                "dataset_card_license": spec["card_license"], "license": spec["license"],
                "official_code_repository": OFFICIAL_CODE_REPO, "official_code_revision": OFFICIAL_CODE_REVISION,
                "official_code_license": "MIT", "official_assets": official_assets,
                "implementation_sha256": implementation, "seed": seed, "n_per_group": n_per_task,
                "grouping": "official_task" if benchmark == "longbench-v1" else "official_domain" if v2_stratify else "all",
                "requested_tasks_or_domains": chosen_tasks, "source_counts_before_filter": dict(sorted(before.items())),
                "eligible_counts": dict(sorted(eligible.items())), "length_rejected_counts": dict(sorted(rejected.items())),
                "max_reference_context_tokens": max_context_tokens, "reference_tokenizer": tokenizer_metadata,
                "selection_method": "fixed_seed_SHA256_rank_by_official_ID_after_preinference_eligibility",
                "selection_frozen_before_inference": True, "resampling_after_failure": False,
                "selected_examples": len(selected), "selected_ids": [e["id"] for e in selected],
                "selected_official_ids": [e["official_id"] for e in selected],
                "selected_records_sha256": _sha(b"".join(_canonical(e) + b"\n" for e in selected)),
                "prompt_rendering": "v1 official format(context,input); v2 official pred.py strip and sequential replace($DOC$,$Q$,$C_A$,$C_B$,$C_C$,$C_D$)",
                "placeholder_literal_audit": {
                    "source_context_counts_in_selected_examples": {token: sum(e.get("prompt_rendering", {}).get("source_context_placeholder_literal_counts", {}).get(token, 0) for e in selected) for token in V2_PLACEHOLDERS},
                    "potentially_affected_selected_ids": [e["id"] for e in selected if e.get("prompt_rendering", {}).get("potential_secondary_context_substitutions", 0)],
                    "note": "Official v2 sequential replacement behavior is intentionally preserved, not corrected; original records remain intact."},
                "remote_dataset_scripts_executed": False, "source_python_files_executed": False,
                "evaluation_only": True, "official_full_run": False,
                "protocol": "length-filtered_official-template_small-subset" if max_context_tokens is not None else "official-template_small-subset",
                "limitations": ["This fixed small subset is not a full official benchmark score.",
                                "LongBench v1 combines upstream tasks; repository MIT code license is not a blanket data license.",
                                "Source-provided lengths are not local/model input token counts.",
                                "Reference context eligibility excludes template, question, chat and output overhead.",
                                "No model calls, training, generated references or human-verification claims are made by fetching."]}
    return selected, manifest
