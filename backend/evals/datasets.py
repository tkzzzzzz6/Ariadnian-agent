"""测评数据：自动下载公开数据集 + 仓库内置的中文小数据集。

公开数据
  SciFact（Wadden et al., EMNLP 2020）的论断—证据对。用的是 COVID-Fact 仓库
  （Saakyan et al., ACL 2021）整理好的 RTE 格式版本：每行一条
  (证据句, 论断, entailment / not_entailment)，标注来自 SciFact 原始数据。
  SciFact 许可为 CC BY-NC 2.0，因此只在运行时下载到 evals/.cache/，不放进仓库。

内置数据（evals/data/）
  zh_claims.jsonl      60 条 SciFact 论断的中文译文，用于跨语言检索测评
  zh_citeworthy.jsonl  20 段中文研究报告段落、72 句，逐句标注“是否需要引用”
  fixture.jsonl        冒烟测试用的几条手写样例（不联网也能跑）
"""
from __future__ import annotations

import csv
import io
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .provenance import file_record, stable_id

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CACHE = HERE / ".cache"

_BASE = "https://raw.githubusercontent.com/asaakyan/covidfact/main"
SOURCES = {
    "scifact_dev": f"{_BASE}/RTE-scifact/dev.tsv",          # 带标签：用于判断与假引用测评
    "scifact_test": f"{_BASE}/RTE-scifact/test.tsv",        # 无标签：只取证据作检索干扰项
    "covidfact": f"{_BASE}/COVIDFACT_dataset.jsonl",        # 只取证据作检索干扰项
}


def _mirrors(url: str) -> list[str]:
    """GitHub raw 在国内有时连不上，再试 jsDelivr 的 GitHub 镜像"""
    out = [url]
    if url.startswith("https://raw.githubusercontent.com/"):
        owner, repo, branch, path = url.removeprefix("https://raw.githubusercontent.com/").split("/", 3)
        out.append(f"https://cdn.jsdelivr.net/gh/{owner}/{repo}@{branch}/{path}")
    return out


def _fetch(name: str) -> str:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / Path(SOURCES[name]).name.replace(".", f"_{name}.", 1)
    if not path.exists():
        errors = []
        for url in _mirrors(SOURCES[name]):
            try:
                r = httpx.get(url, timeout=60, follow_redirects=True)
                r.raise_for_status()
                path.write_text(r.text, encoding="utf-8")
                break
            except httpx.HTTPError as e:
                errors.append(f"{url}: {e}")
        else:
            raise SystemExit(
                "✗ 测评数据下载失败：\n  " + "\n  ".join(errors)
                + f"\n可以手动下载 {SOURCES[name]}\n保存为 {path}，再重新运行；或先用 --source fixture 跑通流程。"
            )
    return path.read_text(encoding="utf-8")


def _tsv(text: str) -> list[dict[str, str]]:
    rows = list(csv.reader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE))
    head = [h.strip().lower() for h in rows[0]]
    return [dict(zip(head, r)) for r in rows[1:] if len(r) >= 3]


@dataclass
class Pair:
    claim: str
    evidence: str
    supported: bool


@dataclass
class Bundle:
    """一次测评用到的全部数据。"""
    pairs: list[Pair]                                   # 带标签的论断—证据对
    distractors: list[str] = field(default_factory=list)  # 只作为检索干扰项的证据段落
    zh_claims: dict[str, str] = field(default_factory=dict)  # 英文论断 → 中文译文
    citeworthy: list[dict] = field(default_factory=list)
    source: str = "scifact"
    source_files: list[dict] = field(default_factory=list)
    reviewed_negatives: list[dict] = field(default_factory=list)

    @property
    def passages(self) -> list[str]:
        """检索语料：所有证据段落去重，顺序固定"""
        seen: dict[str, None] = {}
        for p in self.pairs:
            seen.setdefault(p.evidence)
        for d in self.distractors:
            seen.setdefault(d)
        return list(seen)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _cache_path(name: str) -> Path:
    return CACHE / Path(SOURCES[name]).name.replace(".", f"_{name}.", 1)


def load_reviewed_negatives(path: Path) -> list[dict]:
    """Load explicitly human-reviewed negative pairs, never promote draft labels.

    Unreviewed, ambiguous and supported rows are retained in the file but not
    scored. An asserted negative without the required review metadata is an error.
    Stable IDs bind the review to the exact claim + evidence content.
    """
    rows = _jsonl(path)
    out: dict[str, dict] = {}
    seen: dict[str, str | None] = {}
    for line, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"reviewed negatives line {line}: expected an object")
        label = row.get("label")
        if label not in (None, "", "not_supported", "supports", "ambiguous"):
            raise ValueError(f"reviewed negatives line {line}: invalid label")
        if label in (None, ""):
            continue
        claim, evidence = row.get("claim"), row.get("evidence")
        if not isinstance(claim, str) or not claim.strip() or not isinstance(evidence, str) or not evidence.strip():
            raise ValueError(f"reviewed negatives line {line}: claim and evidence are required")
        cid = stable_id("negative", claim, evidence)
        if row.get("candidate_id") != cid:
            raise ValueError(f"reviewed negatives line {line}: candidate_id does not match content")
        if cid in seen and seen[cid] != label:
            raise ValueError(f"reviewed negatives line {line}: conflicting labels for {cid}")
        seen[cid] = label
        if label != "not_supported":
            continue
        if (row.get("review_status") != "human_reviewed" or not isinstance(row.get("reviewer"), str)
                or not row["reviewer"].strip()):
            raise ValueError(f"reviewed negatives line {line}: not_supported requires human_reviewed and reviewer")
        out[cid] = {"candidate_id": cid, "claim": claim, "evidence": evidence, "label": label,
                    "review_status": "human_reviewed", "reviewer": str(row["reviewer"]),
                    "review_note": str(row.get("review_note") or ""),
                    "independent_review": row.get("independent_review") is True}
    return list(out.values())


def load(source: str = "scifact", reviewed_negatives: Path | None = None) -> Bundle:
    zh = {d["claim_en"]: d["claim_zh"] for d in _jsonl(DATA / "zh_claims.jsonl")}
    cw = _jsonl(DATA / "zh_citeworthy.jsonl")
    if source == "fixture":
        rows = _jsonl(DATA / "fixture.jsonl")
        pairs = [Pair(r["claim"], r["evidence"], r["supported"]) for r in rows]
        zh_fix = {r["claim"]: r["claim_zh"] for r in rows if r.get("claim_zh")}
        bundle = Bundle(pairs, [], zh_fix, cw[:3], source,
                        [file_record(DATA / "fixture.jsonl"), file_record(DATA / "zh_citeworthy.jsonl")])
        return _with_reviews(bundle, reviewed_negatives)

    seen: set[tuple[str, str]] = set()
    pairs: list[Pair] = []
    for r in _tsv(_fetch("scifact_dev")):
        ev, claim, label = r.get("sentence1", "").strip(), r.get("sentence2", "").strip(), r.get("entailment", "").strip()
        if not ev or not claim or label not in {"entailment", "not_entailment"} or (claim, ev) in seen:
            continue
        seen.add((claim, ev))
        pairs.append(Pair(claim, ev, label == "entailment"))

    distractors = [r.get("sentence1", "").strip() for r in _tsv(_fetch("scifact_test"))]
    for line in _fetch("covidfact").splitlines():
        if line.strip():
            distractors.append(" ".join(json.loads(line).get("evidence") or []).strip())
    distractors = [d for d in dict.fromkeys(distractors) if len(d) > 40]
    files = [file_record(_cache_path(name), origin=url) for name, url in SOURCES.items()]
    files += [file_record(DATA / name) for name in ("zh_claims.jsonl", "zh_citeworthy.jsonl")]
    return _with_reviews(Bundle(pairs, distractors, zh, cw, source, files), reviewed_negatives)


def _with_reviews(bundle: Bundle, path: Path | None) -> Bundle:
    if path is not None:
        bundle.reviewed_negatives = load_reviewed_negatives(path)
        positives = {(p.claim, p.evidence) for p in bundle.pairs if p.supported}
        if any((r["claim"], r["evidence"]) in positives for r in bundle.reviewed_negatives):
            raise ValueError("reviewed negative conflicts with a dataset positive")
        bundle.source_files.append(file_record(path, name="reviewed_negatives"))
        bundle.distractors = list(dict.fromkeys(bundle.distractors + [r["evidence"] for r in bundle.reviewed_negatives]))
    return bundle


def sample(items: list, limit: int | None, seed: int = 13) -> list:
    if not limit or limit >= len(items):
        return list(items)
    rng = random.Random(seed)
    return rng.sample(items, limit)
