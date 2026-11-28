"""Chinese, self-contained visual record of a frozen paired long-context pilot.

CPU only. Raw corpora, answers and individual predictions never enter outputs.
Optional GPU/evaluation paths must be supplied explicitly; missing results are
reported as unexecuted, never as zero accuracy or invented curves.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
TASK_NAMES = {"crossdoc": "跨文档关联", "table_condition": "表格与条件约束",
              "evidence_local_qa": "局部证据问答", "multihop": "多证据推理候选"}
DOMAIN_NAMES = {"programming": "程序与技术文档", "machine_learning": "机器学习",
                "agriculture": "农业与环境", "life_sciences": "生命科学"}
ARM_NAMES = {"base": "Base", "short": "短文 SFT", "long": "长文 SFT"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def named_paths(values):
    result = {}
    for value in values or []:
        arm, separator, path = value.partition("=")
        if separator != "=" or arm not in ARM_NAMES or arm in result:
            raise ValueError("explicit arm paths use base=PATH, short=PATH or long=PATH, once per arm")
        result[arm] = Path(path).resolve(strict=True)
    return result


def validate_scores(summary, records_sha256, total):
    """Accept only the actual scorer schema bound to the frozen 250-record suite."""
    if (summary.get("schema_version") != "gemma-pilot-scoring-v1"
            or total != 250 or summary.get("total_frozen_records") != total
            or summary.get("inputs", {}).get("records", {}).get("sha256") != records_sha256
            or summary.get("overall_score") is not None):
        raise ValueError("score summary must bind the frozen250records and must not assert a pooled overall score")
    aliases = {"base": "base", "short": "short", "long": "long", "s_short": "short", "s_long": "long"}
    normalized, names = {}, {}
    for label, rows in summary.get("groups", {}).items():
        arm = aliases.get(label.lower())
        if arm is None or arm in normalized:
            raise ValueError("score labels must uniquely identify Base/short/long")
        names[label] = arm
        # scorer 同时输出长度分桶和 all_lengths 汇总；只累计互斥分桶。
        bucket_rows = [row for row in rows if row["length_bucket"] != "all_lengths"]
        if not bucket_rows or sum(row["records"] for row in bucket_rows) != total:
            raise ValueError("score task groups lost records from their denominator")
        seen = set()
        for row in rows:
            key = tuple(row[field] for field in ("benchmark", "task", "metric", "length_bucket"))
            value = row["score_percent"]
            if (key in seen or row.get("all_ids_in_denominator") is not True
                    or type(row["records"]) is not int or row["records"] <= 0
                    or sum(row["statuses"].values()) != row["records"]
                    or type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 100):
                raise ValueError("invalid score group or failed-call denominator")
            seen.add(key)
        aggregates = {(r['benchmark'], r['task'], r['metric']): r for r in rows if r['length_bucket'] == 'all_lengths'}
        tasks = {(r['benchmark'], r['task'], r['metric']) for r in bucket_rows}
        if aggregates:
            if set(aggregates) != tasks:
                raise ValueError('aggregate task denominator is incomplete')
            for task, aggregate in aggregates.items():
                buckets = [r for r in bucket_rows if (r['benchmark'], r['task'], r['metric']) == task]
                statuses = Counter()
                for bucket in buckets:
                    statuses.update(bucket['statuses'])
                if (aggregate['records'] != sum(r['records'] for r in buckets)
                        or Counter(aggregate['statuses']) != statuses):
                    raise ValueError('aggregate status denominator differs from its length buckets')
        normalized[arm] = rows
    if not normalized or set(names) != set(summary.get("labels", [])):
        raise ValueError("score labels/groups disagree")
    reference = {tuple(r[k] for k in ("benchmark", "task", "metric", "length_bucket")): r["records"] for r in next(iter(normalized.values()))}
    if any({tuple(r[k] for k in ("benchmark", "task", "metric", "length_bucket")): r["records"] for r in rows} != reference for rows in normalized.values()):
        raise ValueError("score groups are not paired across model arms")
    deltas = []
    for row in summary.get("comparisons", []):
        before, after = names.get(row.get("before")), names.get(row.get("after"))
        if {before, after} != {"short", "long"}:
            continue
        key = tuple(row[field] for field in ("benchmark", "task", "metric", "length_bucket"))
        if key not in reference or row.get("records") != reference[key]:
            raise ValueError("paired comparison does not match a frozen task denominator")
        delta, ci = row["delta_percentage_points"], row["ci95_percentage_points"]
        if not isinstance(delta, (int, float)) or not math.isfinite(delta):
            raise ValueError("invalid paired delta")
        if ci is not None and (not isinstance(ci, list) or len(ci) != 2 or any(not math.isfinite(v) for v in ci) or ci[0] > ci[1]):
            raise ValueError("invalid observed bootstrap interval")
        if ci is None and not row.get("ci_unavailable_reason"):
            raise ValueError("absent confidence interval requires an explicit reason")
        item = {**row, "before": "short", "after": "long"}
        if before == "long":
            item["delta_percentage_points"] = -delta
            item["ci95_percentage_points"] = [-ci[1], -ci[0]] if ci else None
        deltas.append(item)
    composites = summary.get("exploratory_composites", [])
    if not isinstance(composites, list):
        raise ValueError("exploratory composites must be a separate list")
    return {"groups": normalized, "short_to_long": deltas, "exploratory_composites": composites,
            "limitations": summary.get("limitations", []), "total": total}


TOKEN_COUNTS = r'''
import json, sys
from pathlib import Path
from tokenizers import Tokenizer
from cpt.assets import load_native_tokenizer
from cpt.swift_verify import _sft_expected_ids
from gemma_eval.chat import encode_chat
packet, assets = Path(sys.argv[1]), Path(sys.argv[2])
native = load_native_tokenizer(assets)
tokenizer = Tokenizer.from_file(str(assets / 'tokenizer.json'))
class Adapter:
    def encode(self, text, add_special_tokens=False):
        return tokenizer.encode(text, add_special_tokens=add_special_tokens).ids
def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
short, long = rows(packet / 'short.jsonl'), rows(packet / 'long.jsonl')
if len(short) != len(long): raise ValueError('paired count mismatch')
result = {'pairs': [], 'native_tokenizer': native.metadata,
          'scope': 'CPU native-tokenizer encoding via existing independent SFT expected-mask implementation; real SWIFT verification not performed'}
for left, right in zip(short, long):
    if left['id'] != right['id'] or left['split'] != right['split'] or left['messages'][-1] != right['messages'][-1]:
        raise ValueError('paired identity/answer mismatch')
    sp, sa = _sft_expected_ids(left['messages'], Adapter())
    lp, la = _sft_expected_ids(right['messages'], Adapter())
    if sa != la: raise ValueError('paired supervised native token IDs differ')
    result['pairs'].append({'split': left['split'], 'short': len(sp) + len(sa),
                           'long': len(lp) + len(la), 'supervised': len(sa),
                           'short_prompt': len(sp), 'long_prompt': len(lp)})
if len(sys.argv) > 3 and sys.argv[3]:
    result['public'] = [{'benchmark': row['benchmark'], 'task': row['task'],
                          'tokens': len(encode_chat(tokenizer, row['prompt'])),
                          'generation_reserve': row.get('official_max_new_tokens', row.get('max_new_tokens'))}
                         for row in rows(Path(sys.argv[3]))]
print(json.dumps(result, ensure_ascii=True))
'''


def collect(packet, *, tokenizer_python, model_assets, public_records=None,
            resource_evidence=None, training_arms=None, evaluation_summaries=None, score_summary=None):
    packet, model_assets = Path(packet).resolve(), Path(model_assets).resolve()
    sources = {}

    def bind(path, label=None):
        path = Path(path).resolve(strict=True)
        label = label or (path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path))
        sources[label] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        return path

    lock = read_json(bind(packet / "data.lock.json"))
    if lock.get("schema_version") != "paired-data-lock-v1":
        raise ValueError("unknown paired data lock")
    for name, expected in lock["artifacts"].items():
        path = (packet / name).resolve()
        if not path.is_relative_to(packet) or sha256(path) != expected:
            raise ValueError("frozen packet artifact changed: " + name)
        bind(path)
    summary, pairs = read_json(packet / "summary.json"), read_rows(packet / "pairs.jsonl")
    if summary.get("schema_version") != "paired-review-candidates-v1":
        raise ValueError("unknown candidate summary")
    short, long = read_rows(packet / "short.jsonl"), read_rows(packet / "long.jsonl")
    if len(pairs) != len(short) or len(short) != len(long):
        raise ValueError("candidate and SFT counts differ")
    for pair, left, right in zip(pairs, short, long):
        if (pair["id"] != left["id"] or left["id"] != right["id"]
                or left["messages"][-1] != right["messages"][-1]
                or left["split"] != right["split"] or pair["split"] != left["split"]):
            raise ValueError("paired identities, answers or splits differ")
    observed_counts = dict(Counter(pair["split"] for pair in pairs))
    if observed_counts != summary["counts"]:
        raise ValueError("summary count differs from frozen pairs")
    domains = dict(Counter(row["domain"] for row in read_rows(packet / "sources.jsonl")))
    if domains != summary["domains"]:
        raise ValueError("summary domain counts differ from source documents")
    for split in ("train", "dev"):
        counts = dict(Counter(row["direction"] for row in pairs if row["split"] == split))
        if counts != summary["directions"][split]:
            raise ValueError("task taxonomy counts disagree with frozen pairs")
    if sha256(model_assets / "model.lock.json") != lock["model_lock_sha256"]:
        raise ValueError("data and plotting tokenizer locks differ")
    bind(model_assets / "model.lock.json")
    bind(ROOT / "experiments/ms_swift/cpt/swift_verify.py")
    bind(ROOT / "experiments/ms_swift/gemma_eval/chat.py")
    public = None
    if public_records is not None:
        public_records = bind(public_records)
        public_manifest = read_json(bind(public_records.parent / "eval_manifest.json"))
        public_rows = read_rows(public_records)
        if (public_manifest.get("schema_version") != "gemma-eval-input-v1"
                or not public_manifest.get("source_manifests")
                or public_manifest.get("records_sha256") != sha256(public_records)
                or public_manifest.get("records") != len(public_rows)
                or public_manifest.get("selected_ids") != [r["id"] for r in public_rows]):
            raise ValueError("public evaluation records differ from their frozen manifest")
        public = {"records": len(public_rows), "benchmark_counts": dict(Counter(r["benchmark"] for r in public_rows)),
                  "evaluation_only": all(r.get("evaluation_only") is True for r in public_rows)}
        if not public["evaluation_only"]:
            raise ValueError("public suite must be evaluation only")
        if (public_records.parent / "summary.json").is_file():
            public["producer_summary"] = read_json(bind(public_records.parent / "summary.json"))
    result = subprocess.run([str(tokenizer_python), "-c", TOKEN_COUNTS, str(packet), str(model_assets),
                             str(public_records) if public_records else ""],
                            capture_output=True, text=True, encoding="utf-8", timeout=180, check=True)
    lengths = json.loads(result.stdout)
    if len(lengths["pairs"]) != len(pairs):
        raise ValueError("native count subprocess lost a pair")
    resource = []
    for kind, target_lengths in (("train", (4096, 8192, 16384)), ("infer", (4096, 8192, 16384, 32768))):
        for length in target_lengths:
            row = {"kind": kind, "length": length, "status": "not_run", "steps": []}
            if resource_evidence:
                base = Path(resource_evidence)
                result_path, stage_path = base / f"probe_{kind}_{length}_result.json", base / f"probe_{kind}_{length}.json"
                if result_path.is_file():
                    observed = read_json(bind(result_path))
                    if (observed.get("schema_version") != "paired-resource-probe-v1"
                            or observed.get("kind") != kind or observed.get("input_length") != length):
                        raise ValueError("resource result is not this paired engineering protocol")
                    row.update(status=observed["status"], steps=observed.get("steps", []),
                               wall_seconds=observed.get("total_wall_seconds"))
                if stage_path.is_file():
                    stage = read_json(bind(stage_path))
                    row["process_status"] = stage["status"]
                    if stage["status"] != "succeeded" and not (stage["status"] == "failed" and row["status"] == "oom"):
                        row["status"] = stage["status"]
                if row["status"] == "succeeded":
                    if len(row["steps"]) != (3 if kind == "train" else 1):
                        raise ValueError("resource success lacks all required steps")
                    if any(step["input_tokens"] != length for step in row["steps"]):
                        raise ValueError("resource actual input length differs from target")
            resource.append(row)
    training = {}
    for arm, path in (training_arms or {}).items():
        if arm == "base":
            raise ValueError("Base is not a training arm")
        receipt = read_json(bind(path / "launch_receipt.json"))
        row = {"status": receipt["status"], "returncode": receipt.get("returncode"), "steps": [], "loss": []}
        audit = path / "training_audit/optimizer_steps.jsonl"
        if audit.is_file():
            row["steps"] = read_rows(bind(audit))
        summary_path = path / "training_audit/summary.json"
        if summary_path.is_file():
            row["exposure"] = read_json(bind(summary_path))
        states = sorted((path / "checkpoints").rglob("trainer_state.json"))
        if len(states) > 1:
            raise ValueError("choose a plan with one retained final trainer state")
        metrics = path / "training_audit/metrics.jsonl"
        if metrics.is_file():
            row["loss"] = [{"step": item["step"], "loss": item["loss"]}
                           for item in read_rows(bind(metrics)) if "loss" in item]
        elif states:
            state = read_json(bind(states[0]))
            row["loss"] = [{"step": item["step"], "loss": item["loss"]}
                           for item in state["log_history"] if "loss" in item]
        training[arm] = row
    evaluation = {}
    for arm, path in (evaluation_summaries or {}).items():
        item = read_json(bind(path))
        if item.get("schema_version") != "gemma-bounded-summary-v1" or item.get("kind") != "chat":
            raise ValueError("evaluation requires a bounded unified-chat summary, not a prior smoke result")
        if item["total"] != item["successful"] + item["failed_or_skipped"]:
            raise ValueError("evaluation failure denominator is inconsistent")
        evaluation[arm] = {key: item[key] for key in ("status", "total", "successful", "failed_or_skipped", "statuses")}
    scores = None
    if score_summary is not None:
        if public_records is None:
            raise ValueError("score summary requires explicit frozen public records")
        scores = validate_scores(read_json(bind(score_summary)), sha256(public_records), public["records"])
    return {"summary": summary, "lengths": lengths, "public": public,
            "resource": resource, "training": training, "evaluation": evaluation, "scores": scores, "sources": sources}


def configure_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    candidate = Path("C:/Windows/Fonts/msyh.ttc")
    family = "DejaVu Sans"
    if candidate.is_file():
        font_manager.fontManager.addfont(str(candidate))
        family = font_manager.FontProperties(fname=str(candidate)).get_name()
    plt.rcParams.update({"font.family": family, "font.size": 10, "axes.titlesize": 13,
                        "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
                        "axes.edgecolor": "#cad5e2", "axes.labelcolor": "#334155", "text.color": "#142b45",
                        "xtick.color": "#53657a", "ytick.color": "#53657a", "axes.unicode_minus": False,
                        "svg.hashsalt": "ariadne-paired-pilot-v1", "savefig.facecolor": "white"})
    return plt


def plots(data, output):
    plt = configure_plot()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files = {}

    def finish(fig, name):
        for extension in ("png", "svg"):
            path = output / f"{name}.{extension}"
            fig.savefig(path, dpi=170, bbox_inches="tight", metadata={"Creator": "Ariadne paired pilot evidence renderer"})
            if extension == "svg":
                # Matplotlib path 行尾空格不影响图形；新产物固定 LF 后再计算发布指纹。
                svg = "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n"
                path.write_text(svg, encoding="utf-8", newline="\n")
            files[f"{name}.{extension}"] = path
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.7), sharey=True, gridspec_kw={"width_ratios": [3, 1]})
    fig.subplots_adjust(left=.07, right=.98, bottom=.24, top=.80, wspace=.15)
    fig.suptitle("同一答案、不同上下文：长短配对输入", x=.07, y=.96, ha="left", fontsize=21, fontweight="bold")
    template_status = ("本轮全部配对已通过真实 SWIFT 编码核验"
                       if data.get("actual_swift_template_verified") is True else "真实 SWIFT 编码仍待核验")
    fig.text(.07, .875, "Gemma 原生分词器 + 固定 SFT 格式的 CPU 预计算；" + template_status, color="#64748b", fontsize=11)
    for ax, split, title in zip(axes, ("train", "dev"), ("训练候选", "验证候选")):
        selected = sorted((r for r in data["lengths"]["pairs"] if r["split"] == split), key=lambda r: r["long"])
        x = list(range(1, len(selected) + 1))
        ax.vlines(x, [r["short"] for r in selected], [r["long"] for r in selected], colors="#d0dbe8", linewidth=1.4)
        ax.scatter(x, [r["short"] for r in selected], s=30, color="#2563b7", label="短文输入", zorder=3)
        ax.scatter(x, [r["long"] for r in selected], s=30, color="#12877b", label="长文输入", zorder=3)
        ax.set_title(f"{title}  ·  {len(selected)} 对", loc="left", pad=13)
        ax.set_xlabel("匿名配对序号（按长文长度排序）")
        ax.set_ylim(0, 8600)
        ax.set_yticks([0, 2048, 4096, 6144, 8192], ["0", "2K", "4K", "6K", "8K"])
        ax.grid(axis="y", color="#e8edf3", linewidth=.8)
        ax.set_axisbelow(True)
        if split == "train":
            ax.legend(frameon=False, loc="upper left", ncols=2)
            ax.set_ylabel("完整 SFT 序列 token 数（包含监督答案）")
    fig.text(.07, .08, "每条灰线连接同一题目的短、长输入；答案 token 完全相同。长度差异不等于模型能力提升。", color="#64748b", fontsize=10)
    finish(fig, "paired_lengths")

    summary = data["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.7), gridspec_kw={"width_ratios": [1.3, 1]})
    fig.subplots_adjust(left=.07, right=.97, bottom=.26, top=.79, wspace=.36)
    fig.suptitle("覆盖方向与来源结构", x=.07, y=.96, ha="left", fontsize=21, fontweight="bold")
    fig.text(.07, .875, f"{sum(summary['counts'].values())} 个候选配对来自 {summary['source_documents']} 份文档；任务标签是待人工验证的设计意图", color="#64748b", fontsize=11)
    tasks = list(TASK_NAMES)
    x = list(range(len(tasks)))
    for split, offset, color, label in (("train", -.19, "#2563b7", "训练候选"), ("dev", .19, "#e6a046", "验证候选")):
        values = [summary["directions"][split].get(task, 0) for task in tasks]
        bars = axes[0].bar([v + offset for v in x], values, .36, color=color, label=label)
        axes[0].bar_label(bars, padding=4, fontsize=10)
    axes[0].set_xticks(x, [TASK_NAMES[t].replace("与", "与\n") for t in tasks])
    axes[0].set_ylabel("配对样本数")
    axes[0].set_ylim(0, max(max(summary["directions"][s].values(), default=0) for s in ("train", "dev")) * 1.4 + 1)
    axes[0].legend(frameon=False, ncols=2)
    domains = sorted(summary["domains"], key=summary["domains"].get)
    bars = axes[1].barh([DOMAIN_NAMES.get(d, d) for d in domains], [summary["domains"][d] for d in domains], color="#12877b", height=.55)
    axes[1].bar_label(bars, padding=6)
    axes[1].set_xlim(0, max(summary["domains"].values()) * 1.3)
    axes[1].set_xlabel("去重来源文档数（不是样本数）")
    axes[1].set_title("来源领域", loc="left", pad=14)
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="y" if ax is axes[0] else "x", color="#e8edf3")
    fig.text(.07, .095, f"连接来源族：train {summary['connected_families']['train']} / dev {summary['connected_families']['dev']}；隔离单位为文件、章节或 DOI，非整本书或整个仓库。", color="#64748b", fontsize=10)
    fig.text(.07, .045, "覆盖限制：训练集跨文档候选集中于同一 CSV–Decimal 来源族；验证候选仅为英文。", color="#9a641b", fontsize=10)
    finish(fig, "task_source_mix")

    if data["public"]:
        rows = data["lengths"]["public"]
        groups = sorted(set(row["benchmark"] for row in rows))
        bins = [0, 2048, 4096, 8192, 16384, 32768, 65536, 131072, float("inf")]
        fig, ax = plt.subplots(figsize=(13.4, 5.7))
        fig.subplots_adjust(left=.07, right=.97, bottom=.22, top=.79)
        fig.suptitle("冻结公共测试集：原生 chat 输入长度", x=.07, y=.96, ha="left", fontsize=21, fontweight="bold")
        fig.text(.07, .875, f"共 {len(rows)} 题；只作评测，不进入 SFT。生成 token 预算另行保留。", color="#64748b", fontsize=11)
        positions = list(range(len(bins) - 1))
        bottom = [0] * len(positions)
        palette = ["#2563b7", "#12877b", "#e6a046", "#8464ba"]
        for index, group in enumerate(groups):
            counts = [sum(row["benchmark"] == group and lo <= row["tokens"] < hi for row in rows) for lo, hi in zip(bins, bins[1:])]
            ax.bar(positions, counts, bottom=bottom, color=palette[index % len(palette)], label=group, width=.7)
            bottom = [a + b for a, b in zip(bottom, counts)]
        ax.set_xticks(positions, ["<2K", "2–4K", "4–8K", "8–16K", "16–32K", "32–64K", "64–128K", "≥128K"])
        ax.set_ylabel("评测题数")
        ax.set_xlabel("完整输入 token 区间；图示采用左闭右开区间")
        ax.legend(frameon=False, ncols=min(4, len(groups)))
        ax.grid(axis="y", color="#e8edf3")
        ax.set_axisbelow(True)
        fig.text(.07, .07, "超出输入 + 生成预算的题目应计入 skipped_context_limit，不能从评测分母中删除。", color="#64748b", fontsize=10)
        finish(fig, "public_lengths")

    measured = [r for r in data["resource"] if r["steps"]]
    if measured:
        fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.3))
        fig.subplots_adjust(left=.07, right=.97, bottom=.24, top=.79, wspace=.24)
        fig.suptitle("A100 资源实测记录", x=.07, y=.96, ha="left", fontsize=21, fontweight="bold")
        fig.text(.07, .875, "仅绘制已完成 step 的真实观测；失败或超时保留在下方状态表", color="#64748b", fontsize=11)
        for kind, color, label, offset in (("train", "#2563b7", "工程训练负载", -.1), ("infer", "#12877b", "工程推理负载", .1)):
            selected = [r for r in measured if r["kind"] == kind]
            xs = [r["length"] / 1024 + offset for r in selected]
            axes[0].scatter(xs, [max(s["peak_allocated_bytes"] for s in r["steps"]) / 2**30 for r in selected], label=label, color=color, s=60)
            axes[1].scatter(xs, [sum(s["wall_seconds"] for s in r["steps"]) / len(r["steps"]) for r in selected], color=color, s=60)
        for ax in axes:
            ax.set_xticks([4, 8, 16, 32], ["4K", "8K", "16K", "32K"])
            ax.set_xlabel("实际输入 token 长度")
            ax.grid(axis="y", color="#e8edf3")
            ax.set_axisbelow(True)
        axes[0].set_ylabel("已记录 step 的最大 allocated 显存 / GiB")
        axes[0].legend(frameon=False)
        axes[1].set_ylabel("已记录 step 的平均耗时 / 秒")
        fig.text(.07, .07, "工程训练使用一次性合成 token + 32 个监督目标；它不构成真实 QA 训练效果，也不验证更长答案的资源开销。", color="#64748b", fontsize=9)
        finish(fig, "resource_measurements")

    if any(row["loss"] for row in data["training"].values()):
        fig, ax = plt.subplots(figsize=(13.4, 5.3))
        fig.subplots_adjust(left=.08, right=.97, bottom=.20, top=.79)
        fig.suptitle("SFT 训练日志：仅展示实际记录", x=.08, y=.96, ha="left", fontsize=21, fontweight="bold")
        for arm, color in (("short", "#2563b7"), ("long", "#12877b")):
            points = data["training"].get(arm, {}).get("loss", [])
            if points:
                ax.plot([p["step"] for p in points], [p["loss"] for p in points], marker="o", color=color, label=ARM_NAMES[arm])
        ax.set_xlabel("实际 optimizer step")
        ax.set_ylabel("SWIFT 记录的训练 loss")
        ax.legend(frameon=False)
        ax.grid(axis="y", color="#e8edf3")
        fig.text(.08, .065, "未平滑、未插补；loss 下降不能单独证明长上下文泛化提升。", color="#64748b")
        finish(fig, "training_loss")
    scores = data.get("scores")
    if scores:
        tasks = sorted({(r["benchmark"], r["task"], r["metric"]) for rows in scores["groups"].values() for r in rows})
        columns, row_count = min(3, len(tasks)), math.ceil(len(tasks) / 3)
        fig, grid = plt.subplots(row_count, columns, figsize=(15, 3.7 * row_count + 1.0), squeeze=False)
        fig.subplots_adjust(left=.07, right=.97, bottom=.075, top=.90, hspace=.80, wspace=.32)
        fig.suptitle("统一评测：各任务独立展示", x=.07, y=.985, ha="left", fontsize=21, fontweight="bold")
        fig.text(.07, .945, "每个分面只比较同一任务、同一指标；失败与跳过样本保留在分母中，不计算异质整体分数。", color="#64748b", fontsize=10)
        for ax, task in zip(grid.flat, tasks):
            benchmark, task_name, metric = task
            matching = {arm: [r for r in rows if (r["benchmark"], r["task"], r["metric"]) == task] for arm, rows in scores["groups"].items()}
            buckets = sorted({r["length_bucket"] for rows in matching.values() for r in rows}, key=lambda b: int("".join(c for c in b if c.isdigit()) or 0))
            for arm, offset, color in (("base", -.24, "#94a3b8"), ("short", 0, "#2563b7"), ("long", .24, "#12877b")):
                by_bucket = {r["length_bucket"]: r for r in matching.get(arm, [])}
                positions = [index + offset for index, bucket in enumerate(buckets) if bucket in by_bucket]
                values = [by_bucket[bucket]["score_percent"] for bucket in buckets if bucket in by_bucket]
                if values:
                    ax.bar(positions, values, .23, color=color, label=ARM_NAMES[arm])
            ax.set_title(f"{benchmark} / {task_name}\n{metric}", loc="left", fontsize=10, pad=9)
            ax.set_xticks(range(len(buckets)), buckets, rotation=25, ha="right", fontsize=8)
            ax.set_ylim(0, 105)
            ax.set_ylabel("指标分值 / %")
            ax.grid(axis="y", color="#e8edf3")
            ax.set_axisbelow(True)
        for ax in list(grid.flat)[len(tasks):]:
            ax.set_visible(False)
        grid.flat[0].legend(frameon=False, fontsize=8, ncols=3)
        finish(fig, "task_scores")
        deltas = scores["short_to_long"]
        if deltas:
            fig, ax = plt.subplots(figsize=(13.4, max(5.0, len(deltas) * .30 + 1.9)))
            fig.subplots_adjust(left=.38, right=.93, bottom=.12, top=.86)
            fig.suptitle("短文 → 长文 SFT：配对差值与来源簇区间", x=.055, y=.97, ha="left", fontsize=18, fontweight="bold")
            fig.text(.055, .915, "同任务、同指标内的百分点变化；区间不可用时只画点，并明确标注。", color="#64748b", fontsize=10)
            for index, row in enumerate(deltas):
                value, interval = row["delta_percentage_points"], row["ci95_percentage_points"]
                color = "#12877b" if interval is not None else "#e6a046"
                if interval is not None:
                    ax.hlines(index, interval[0], interval[1], color=color, linewidth=2)
                ax.scatter([value], [index], color=color, s=28, zorder=3)
            labels = [f"{r['task']} · {r['metric']} · {r['length_bucket']}" + ("  [CI 不可用]" if r["ci95_percentage_points"] is None else "") for r in deltas]
            ax.set_yticks(range(len(labels)), labels, fontsize=8)
            ax.invert_yaxis()
            ax.axvline(0, color="#94a3b8", linestyle="--", linewidth=1)
            ax.set_xlabel("长文分数 − 短文分数 / 百分点；误差线为配对来源簇 bootstrap 95% CI")
            ax.grid(axis="x", color="#e8edf3")
            ax.set_axisbelow(True)
            fig.text(.055, .025, "探索性区间未校正多重比较；不包含训练随机种子的方差。单一来源簇不能估计来源泛化不确定性。", color="#64748b", fontsize=9)
            finish(fig, "short_long_deltas")
    return files


def status_name(value):
    return {"not_run": "尚未执行", "succeeded": "成功", "failed": "失败", "timeout": "超时",
            "oom": "显存不足", "starting": "未完成", "running": "运行中"}.get(value, value)


def render_html(data, figures, output, blocker):
    esc = lambda value: html.escape(str(value))
    summary = data["summary"]

    def figure(name, caption):
        path = figures.get(name + ".png")
        if path is None:
            return ""
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'<figure><img src="data:image/png;base64,{encoded}" alt="{esc(caption)}"><figcaption>{esc(caption)}</figcaption></figure>'

    resource_rows = "".join(f"<tr><td>{'训练' if r['kind']=='train' else '推理'}</td><td>{r['length'] // 1024}K</td><td>{esc(status_name(r['status']))}</td><td>{len(r['steps']) if r['status']!='not_run' else '—'}</td></tr>" for r in data["resource"])
    evaluation_rows = ""
    for arm in ARM_NAMES:
        row = data["evaluation"].get(arm)
        score_groups = (data.get("scores") or {}).get("groups", {}).get(arm)
        if row is None and score_groups:
            buckets = [group for group in score_groups if group['length_bucket'] != 'all_lengths']
            successful = sum(group["statuses"].get("ok", 0) for group in buckets)
            total = sum(group["records"] for group in buckets)
            row = {"status": "已有绑定评分汇总", "total": total, "successful": successful,
                   "failed_or_skipped": total - successful}
        values = [ARM_NAMES[arm], row["status"] if row else "尚未执行",
                  row["total"] if row else "—", row["successful"] if row else "—",
                  row["failed_or_skipped"] if row else "—", "见分任务指标图" if score_groups else "尚无准确率结果"]
        evaluation_rows += "<tr>" + "".join(f"<td>{esc(value)}</td>" for value in values) + "</tr>"
    training_rows = ""
    for arm in ("short", "long"):
        row = data["training"].get(arm)
        exposure = row.get("exposure", {}) if row else {}
        values = [ARM_NAMES[arm], row["status"] if row else "尚未执行",
                  len(row["steps"]) if row else "—", exposure.get("input_tokens", "—"), exposure.get("supervised_tokens", "—"),
                  exposure.get("best_step", "—"), exposure.get("selected_input_tokens", "—"), exposure.get("selected_supervised_tokens", "—")]
        training_rows += "<tr>" + "".join(f"<td>{esc(value)}</td>" for value in values) + "</tr>"
    source_rows = "".join(f"<tr><td>{esc(name)}</td><td><code>{item['sha256']}</code></td></tr>" for name, item in data["sources"].items())
    native = data["lengths"]["pairs"]
    short_range = f"{min(r['short'] for r in native):,}–{max(r['short'] for r in native):,}"
    long_range = f"{min(r['long'] for r in native):,}–{max(r['long'] for r in native):,}"
    public_description = (f"已冻结 {data['public']['records']} 题；下图为本地原生 tokenizer 计算的完整 chat 输入长度。"
                          if data["public"] else "公共测试集正在准备；本次报告未读取冻结测试集，不展示推测分布。")
    public_cautions = []
    producer = (data.get("public") or {}).get("producer_summary", {})
    if producer.get("ruler_qa_independent_clusters", 2) < 2:
        articles = "、".join(producer.get("ruler_qa_source_articles", []))
        public_cautions.append(f"RULER qa_1 来自同一来源文章 {articles}，仅 {producer['ruler_qa_independent_clusters']} 个独立来源簇；来源簇 bootstrap 置信区间不可估计。")
    narrative = producer.get("confirmation_primary_eligible_by_task", {}).get("narrativeqa")
    if narrative is not None and narrative < 50:
        public_cautions.append(f"正式确认实验保留集中 NarrativeQA 可用 {narrative} 条，低于规划的 50 条；应补充独立来源或调整方案后再启动确认实验。")
    caution_html = "<ul>" + "".join(f"<li>{esc(item)}</li>" for item in public_cautions) + "</ul>" if public_cautions else ""
    scores = data.get("scores")
    score_figures = figure("task_scores", "真实绑定评分汇总的分任务比较；F1、ROUGE、召回等指标分别展示，不解释为统一准确率。")
    score_figures += figure("short_long_deltas", "区间缺失保留为不可用；不使用零宽误差线冒充置信区间。")
    if scores:
        score_figures += f"<p class='muted'>评分输入绑定本轮 {scores['total']} 条冻结记录。检测到 {len(scores['exploratory_composites'])} 项探索性混合指标汇总；本页不将它们画作整体准确率或正式成功标准。</p>"
    report = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gemma 长上下文配对实验记录</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#eef3f8;color:#142b45;font-family:"Microsoft YaHei",system-ui,sans-serif;line-height:1.7}}main{{max-width:1180px;margin:auto;padding:48px 32px}}header{{padding-bottom:25px;border-bottom:1px solid #d3dee9}}.eyebrow{{color:#12877b;font-size:13px;font-weight:700;letter-spacing:2px}}h1{{font-size:35px;line-height:1.3;margin:14px 0}}h2{{font-size:23px;margin:36px 0 12px}}p{{margin:8px 0 15px}}.muted,figcaption{{color:#60748a;font-size:14px}}.banner{{background:#fff6e8;border-left:4px solid #e0a144;padding:15px 20px;margin-top:24px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:24px 0}}.card,section{{background:white;border:1px solid #dbe4ee;border-radius:10px;padding:23px}}.number{{font-size:28px;font-weight:750;color:#2563b7}}.card small{{display:block;color:#64748b}}section{{margin:23px 0}}section h2{{margin-top:0}}figure{{margin:20px 0 5px}}img{{width:100%;height:auto;display:block}}figcaption{{padding:8px 5px}}table{{width:100%;border-collapse:collapse;margin:15px 0;font-size:14px}}th,td{{padding:10px;border-bottom:1px solid #e2e8f0;text-align:left;vertical-align:top}}th{{background:#f4f7fa}}code{{font-family:ui-monospace,monospace;font-size:11px;word-break:break-all}}.scroll{{overflow:auto}}.pill{{display:inline-block;background:#e8f2fa;padding:3px 10px;border-radius:20px;font-size:12px}}ul{{padding-left:22px}}footer{{padding:25px 0;color:#60748a;font-size:12px}}@media(max-width:760px){{main{{padding:24px 14px}}h1{{font-size:27px}}.cards{{grid-template-columns:repeat(2,1fr)}}section{{padding:16px}}}}
</style><main><header><div class="eyebrow">ARIADNE · GEMMA 4 E2B · PAIRED LONG CONTEXT</div>
<h1>长上下文能力：配对实验可视化记录</h1><p>固定任务、答案与来源，比较短文 SFT 与长文 SFT，再使用统一公共测试集检验差异。</p>
<span class="pill">数据候选与工程准备记录</span> <span class="pill">本页不宣称能力提升</span>
<div class="banner"><strong>当前状态</strong><br>{esc(blocker)}<br>历史 10 条 smoke 只证明旧工程链路；本页不复用其 loss、显存或正确率作为本轮结果。</div></header>
<div class="cards"><div class="card"><div class="number">{summary['counts']['train']} + {summary['counts']['dev']}</div><small>训练候选 / 验证候选（配对）</small></div><div class="card"><div class="number">{summary['source_documents']}</div><small>来源文档；{sum(summary['connected_families'].values())} 个连接来源族</small></div><div class="card"><div class="number">{short_range}</div><small>短文完整 SFT token 预计算</small></div><div class="card"><div class="number">{long_range}</div><small>长文完整 SFT token 预计算</small></div></div>
<section><h2>1. 配对数据与长度</h2><p>每一对保留相同问题、答案、来源和 split。下面直接比较完整 SFT 序列预计算值，而非配置中的最大长度。真实训练前仍须由锁定版本的 SWIFT 对全部记录核验。</p>
{figure('paired_lengths','匿名配对点；使用原生 tokenizer 与既有 SFT expected-mask 实现计算，尚不等于实际训练已消费的输入。')}
<p class="muted">人工批准配对：{summary['approved_pairs']}；training_ready：{str(summary['training_ready']).lower()}。本轮为 {summary['counts']['train']} / {summary['counts']['dev']} 的预实验，原提案为 128 / 32；没有通过复制样本补齐规模。</p></section>
<section><h2>2. 多方向覆盖与来源边界</h2>{figure('task_source_mix','任务计数单位为配对；领域计数单位为去重来源文档，两者不可混用。')}
<ul><li>“多证据推理候选”是设计标签，不能仅凭证据距离或机械检查宣称推理正确。</li><li>train / dev 按连接来源族隔离；文件、章节或 DOI 隔离不等于整库、整书隔离。</li><li>训练跨文档候选集中于同一个 CSV–Decimal 来源族；dev 仅英文，泛化覆盖有限。</li></ul></section>
<section><h2>3. 冻结公共评测</h2><p>{esc(public_description)}</p>{figure('public_lengths','公共原始任务 + 受控长距离诊断 + 短文回归项分别保留来源、预算及评测用途。')}
{caution_html}
<p class="muted">测试集不用于训练、数据筛选或 checkpoint 选择；正式对比应保留所有失败和超长记录的分母。</p></section>
<section><h2>4. A100 资源验证</h2>{figure('resource_measurements','真实资源观测：allocator peak 与整卡采样属于不同口径；此图仅展示前者。')}
<table><thead><tr><th>负载</th><th>实际输入长度</th><th>状态</th><th>已记录 step</th></tr></thead><tbody>{resource_rows}</tbody></table>
<p class="muted">工程训练目标：4K / 8K / 16K 各 3 次更新；推理：4K / 8K / 16K / 32K 各生成 32 token。缺失结果记为“尚未执行”，失败、OOM 和超时不记为通过。</p></section>
<section><h2>5. 两臂训练与实际 token 消费</h2>{figure('training_loss','仅来自实际 trainer_state 的未平滑日志；训练 loss 不能替代 held-out 能力测评。')}
<table><thead><tr><th>模型</th><th>状态</th><th>已记录更新</th><th>实际输入 token</th><th>实际监督 token</th><th>best step</th><th>选中模型输入 tokens</th><th>选中模型监督 tokens</th></tr></thead><tbody>{training_rows}</tbody></table>
<p class="muted">完整训练预算与选中模型的有效曝光分别报告；best 可能早于最终更新，两臂选中模型不一定经历相同步数。</p>
<p class="muted">批准后按实际 {summary['counts']['train']} 个训练配对执行 2 epochs、batch 1、accumulation 8，即每臂 {(summary['counts']['train'] + 7) // 8 * 2} 个 optimizer steps；两臂相同答案与更新数，实际输入 token 量分别审计。</p></section>
<section><h2>6. 统一模型评测记录</h2><table><thead><tr><th>模型</th><th>状态</th><th>总分母</th><th>调用成功</th><th>失败 / 跳过</th><th>能力指标</th></tr></thead><tbody>{evaluation_rows}</tbody></table>
{score_figures}<p class="muted">调用成功率与回答正确率是两个指标。未提供绑定评分证据时不绘制能力分数；本页不会混合各任务指标构造整体准确率。</p></section>
<section><h2>7. 数据与代码来源指纹</h2><p class="muted">此页面及 PNG / SVG 自包含聚合信息，不包含题目、答案、来源全文或个体预测。可用下列 SHA-256 对照本地忽略目录中的原始证据。</p>
<details><summary>展开来源文件与 SHA-256</summary><div class="scroll"><table><thead><tr><th>来源</th><th>SHA-256</th></tr></thead><tbody>{source_rows}</tbody></table></div></details></section>
<footer>生成时间：{datetime.now(timezone.utc).isoformat()} · 渲染器 plot_paired_pilot.py · 无外部脚本、字体服务或图像依赖。</footer></main></html>'''
    output = Path(output)
    output.write_text(report, encoding="utf-8")
    return output


def build_report(packet, output, **kwargs):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    blocker = kwargs.pop("blocker", "新配对数据仍待人工审核；尚无本轮 GPU 结果输入。")
    data = collect(packet, **kwargs)
    figures = plots(data, output / "figures")
    report = render_html(data, figures, output / "report.html", blocker)
    pairs = data["lengths"]["pairs"]
    manifest = {"schema_version": "paired-visual-report-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "renderer_sha256": sha256(__file__), "source_hashes": data["sources"],
                "outputs": {path.relative_to(output).as_posix(): sha256(path) for path in [report, *figures.values()]},
                "counts": data["summary"]["counts"], "approved_pairs": data["summary"]["approved_pairs"],
                "native_full_sft_precomputed_ranges": {arm: [min(r[arm] for r in pairs), max(r[arm] for r in pairs)] for arm in ("short", "long")},
                "actual_swift_template_verified": False, "quality_improvement_measured": False,
                "resource_status_counts": dict(Counter(row["status"] for row in data["resource"])),
                "training_arms_with_evidence": sorted(data["training"]), "evaluation_arms_with_evidence": sorted(data["evaluation"]),
                "public_eval_records": data["public"]["records"] if data["public"] else None,
                "bound_score_summary_supplied": data.get("scores") is not None,
                "blocker": blocker, "record_content_exported": False,
                "semantics": "Missing results are not_run, never zero accuracy. CPU token precomputation is not real SWIFT execution."}
    save(output / "visualization_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--packet", type=Path, default=ROOT / "experiments/ms_swift/.cache/paired_pilot_20261108_v1/review_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/results/paired_pilot_20261108")
    parser.add_argument("--tokenizer-python", type=Path, default=ROOT / "experiments/ms_swift/.venv/Scripts/python.exe")
    parser.add_argument("--model-assets", type=Path, default=ROOT / "experiments/ms_swift/.cache/model_assets/gemma4-e2b")
    parser.add_argument("--public-records", type=Path)
    parser.add_argument("--resource-evidence", type=Path)
    parser.add_argument("--training-arm", action="append", default=[])
    parser.add_argument("--evaluation-summary", action="append", default=[])
    parser.add_argument("--score-summary", type=Path)
    parser.add_argument("--blocker", default="新配对数据仍待人工审核；尚无本轮 GPU 结果输入。")
    args = vars(parser.parse_args())
    args["training_arms"] = named_paths(args.pop("training_arm"))
    args["evaluation_summaries"] = named_paths(args.pop("evaluation_summary"))
    print(json.dumps(build_report(**args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
