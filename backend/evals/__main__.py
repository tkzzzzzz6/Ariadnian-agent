"""运行测评：  uv run python -m evals [--tasks judge,fake_cite,locate] [--limit 50] [--model deepseek/deepseek-chat] [--offline]

（仓库根目录下也可以用 npm run eval -- <同样的参数>）

- 使用网页「设置」或 .env 里配置的模型；--model 可临时把所有档位换成同一个模型做对比。
- 测评在一个临时数据目录里进行（复制一份设置），不会碰你的文献库和论文。
- 结果写入 evals/results/<时间>_<模型>.json，并重新生成 evals/results/README.md 汇总表。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
RESULTS = HERE / "results"


def _prepare_env(args: argparse.Namespace) -> Path:
    """在 import app 之前切到临时数据目录，并带上用户的设置（模型、Key）。"""
    real = Path(os.getenv("ARIA_DATA_DIR") or (BACKEND / "data"))
    settings: dict[str, str] = {}
    if (real / "settings.json").exists():
        settings = json.loads((real / "settings.json").read_text(encoding="utf-8"))
        if not isinstance(settings, dict):
            raise ValueError("settings.json must contain an object")
    if args.model:
        settings.update({"MODEL_ID": args.model, "FAST_MODEL_ID": args.model,
                         **{f"MODEL_{t}": args.model for t in ("COMPLETE", "WRITE", "JUDGE", "AGENT")}})
    # 网页里开着“离线演示模式”时，真实测评也要调用模型，所以显式关掉
    settings["LLM_FAKE"] = "1" if args.offline else "__none__"
    if args.offline:
        settings["EMBEDDING_PROVIDER"] = "none"  # offline must never call a configured embedding API
    tmp = Path(tempfile.mkdtemp(prefix="aria-eval-"))
    (tmp / "settings.json").write_text(json.dumps(settings, ensure_ascii=False), encoding="utf-8")
    os.environ["ARIA_DATA_DIR"] = str(tmp)
    if args.offline:
        os.environ["LLM_FAKE"] = "1"
    return tmp


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m evals", description="Ariadne 测评")
    ap.add_argument("--tasks", default="judge,fake_cite,retrieval,missing,locate,latency", help="逗号分隔")
    ap.add_argument("--limit", type=int, default=None, help="每个任务最多取多少条（默认全部）")
    ap.add_argument("--workers", type=int, default=4, help="并发调用模型的线程数")
    ap.add_argument("--model", default=None, help="临时把所有档位换成这个模型，如 deepseek/deepseek-chat")
    ap.add_argument("--offline", action="store_true", help="不调用模型：用离线规则作为基线")
    ap.add_argument("--source", default="scifact", choices=["scifact", "fixture"], help="fixture=内置几条样例，不联网")
    ap.add_argument("--label", default=None, help="结果文件名里的标签")
    ap.add_argument("--seed", type=int, default=13, help="抽样与负例候选生成的固定种子；locate 构造/排版固定为 7/11")
    ap.add_argument("--reviewed-negatives", type=Path, help="人工复核的负例 JSONL；未复核候选不会计入检出率")
    ap.add_argument("--results-dir", type=Path, default=RESULTS, help="结果目录，默认 evals/results")
    ap.add_argument("--no-save", action="store_true", help="只打印，不写结果文件")
    args = ap.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        ap.error("--limit 必须是正整数")
    if args.workers < 1:
        ap.error("--workers 必须是正整数")

    from . import report, tasks
    from .datasets import load
    from .provenance import PROTOCOL_VERSION, audit_requests, capture, digest, git_state, safe_text

    names = tasks.iter_tasks(x.strip() for x in args.tasks.split(",") if x.strip())

    try:
        tmp = _prepare_env(args)
    except (ValueError, OSError) as exc:
        print(f"✗ 无法读取或复制本地模型配置（{type(exc).__name__}）", file=sys.stderr)
        return 2
    sys.path.insert(0, str(BACKEND))
    from app.config import get_settings
    from app.db import init_db

    try:
        s = get_settings()
        if not s.llm_ready:
            print("✗ 没有配置模型 API Key。请先在网页「设置」里配置，或加 --offline 只跑离线基线。", file=sys.stderr)
            return 2
        init_db()
        models = {t: safe_text(tier.model) for t, tier in s.tiers.items()}
        mode = "offline" if s.llm_fake else "model"
        print(f"▸ 模式：{'离线基线（不调用模型）' if s.llm_fake else '真实模型'}  judge={models['judge']}  complete={models['complete']}", file=sys.stderr)
        t0 = time.perf_counter()
        try:
            bundle = load(args.source, args.reviewed_negatives)
        except (ValueError, OSError) as exc:
            print(f"✗ 数据或人工复核文件无效：{safe_text(str(exc))}", file=sys.stderr)
            return 2
        provenance = capture(s, bundle, args.seed, names, args.limit)
        provenance["workers"] = {"tasks": args.workers, "retrieval_keyword_expansion": 4}
        print(f"▸ 数据：{len(bundle.pairs)} 对论断—证据，{len(bundle.passages)} 段检索语料", file=sys.stderr)
        lib = None
        out: dict = {}
        with audit_requests() as requests:
            for name in names:
                print(f"▸ {name}", file=sys.stderr)
                ts = time.perf_counter()
                try:
                    if name in ("fake_cite", "retrieval") and lib is None:
                        lib = tasks.Library(bundle)
                    if name == "judge":
                        out[name] = tasks.run_judge(bundle, args.limit, args.workers, args.seed)
                    elif name == "fake_cite":
                        out[name] = tasks.run_fake_cite(bundle, lib, args.limit, args.workers, args.seed)
                    elif name == "retrieval":
                        out[name] = tasks.run_retrieval(bundle, lib, args.limit, s.llm_fake, args.seed)
                    elif name == "missing":
                        out[name] = tasks.run_missing(bundle, args.limit, args.workers, args.seed)
                    elif name == "locate":
                        out[name] = tasks.run_locate_audited(bundle, args.limit)
                    elif name == "latency":
                        out[name] = tasks.run_latency(bundle, args.limit, args.seed) if not s.llm_fake else {"skipped": "离线模式不测延迟"}
                except Exception as exc:
                    out[name] = {"status": "failure", "error_type": type(exc).__name__, "error": safe_text(str(exc))}
                out[name]["seconds"] = round(time.perf_counter() - ts, 1)
        provenance["model_requests"] = requests
        provenance["model_requests_sha256"] = digest(sorted((digest(r) for r in requests)))
        provenance["git_at_end"] = git_state()
        usage = tasks.llm_usage()

        result = {
            "protocol_version": PROTOCOL_VERSION, "provenance": provenance,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mode": mode, "models": models,
            "embeddings": f"{s.embedding_provider}/{safe_text(s.embedding_model)}" if s.embedding_provider != "none" else None,
            "source": args.source, "seed": args.seed, "limit": args.limit,
            "data": {"pairs": len(bundle.pairs), "corpus": len(bundle.passages),
                     "zh_claims": len(bundle.zh_claims), "citeworthy_paragraphs": len(bundle.citeworthy),
                     "reviewed_negatives": len(bundle.reviewed_negatives)},
            "tasks": out, "llm_usage": usage, "seconds": round(time.perf_counter() - t0, 1),
        }
        print(report.summarize(result))
        if not args.no_save:
            results_dir = args.results_dir.resolve()
            results_dir.mkdir(parents=True, exist_ok=True)
            label = args.label or ("offline" if s.llm_fake else models["judge"])
            slug = re.sub(r"[^\w.-]+", "-", safe_text(label)).strip("-")[:60]
            path = results_dir / f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S-%f}_{slug}.json"
            candidates = list(out.get("fake_cite", {}).get("negative_candidates", []))
            candidates += out.get("judge", {}).get("zh_negative_candidates", [])
            if candidates:
                candidate_path = path.with_suffix(".negative-candidates.jsonl")
                unique: dict[str, dict] = {}
                for candidate in candidates:
                    cid = candidate["candidate_id"]
                    previous = unique.get(cid, {})
                    preferred = previous if previous.get("review_status") == "human_reviewed" else candidate
                    kinds = sorted(set(previous.get("candidate_kinds", [])) | set(candidate.get("candidate_kinds", [])))
                    unique[cid] = {**preferred, "candidate_kinds": kinds}
                candidate_path.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in unique.values()), encoding="utf-8")
                result["negative_candidates_file"] = candidate_path.name
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            summary_path = report.write_summary(path)
            report.write_index(results_dir)
            print(f"\n✓ 本机逐条结果：{path}；可提交摘要：{summary_path}；汇总表：{results_dir / 'README.md'}", file=sys.stderr)
        return 1 if any(t.get("status") == "failure" for t in out.values()) else 0
    finally:
        # tmp was created by mkdtemp; guard the exact resolved cleanup target.
        if tmp.resolve().parent != Path(tempfile.gettempdir()).resolve() or not tmp.name.startswith("aria-eval-"):
            raise RuntimeError("Refusing cleanup outside the evaluation temporary directory")
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
