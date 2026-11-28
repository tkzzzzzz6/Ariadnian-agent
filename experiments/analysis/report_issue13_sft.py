"""Issue13 实测报告：从已封存的 Colab 证据生成官方评分和聚合图表。

在仓库根目录通过 experiments/ms_swift 独立环境运行；私有输入必须显式提供。
不修改原始审核包、远端计划或逐题预测，不根据公开分数重新选择 checkpoint。
"""

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path

import plot_paired_pilot as report
from pilot_eval.scoring import score_files, yes_no

parser = argparse.ArgumentParser()
parser.add_argument("--artifact-root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--packet", type=Path, required=True)
parser.add_argument("--approval-receipt", type=Path, required=True)
parser.add_argument("--model-assets", type=Path, required=True)
parser.add_argument("--tokenizer-python", type=Path, default=Path(sys.executable))
parser.add_argument("--metric-root", type=Path, required=True)
parser.add_argument("--suite-summary", type=Path, required=True)
args = parser.parse_args()
repo = Path(__file__).resolve().parents[2]
root = args.artifact_root.resolve()
out = args.output.resolve()
out.mkdir(exist_ok=False, parents=True)
read = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_lf(path, text):
    # 新生成的可版本化文本固定 LF，使 GitHub 与本地指纹相同。
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


proof = read(root / "evidence/final_validation_summary.json")
assert proof["paired_sft_completed"] and proof["complete_best_and_drive_verified"]
packet = args.packet.resolve()
approval_path = args.approval_receipt.resolve()
approval = read(approval_path)
assert (
    approval["reviewer_type"] == "human"
    and "internal_sft_pilot" in approval["approved_scope"]
)
assert approval["review_packet_sha256"] == sha(packet / "review_packet.lock.json")
assert all(sha(packet / n) == h for n, h in approval["approved_inputs"].items())
records = root / "inputs/eval_records.jsonl"
assert (
    sha(records) == "f22fb98c38a65f68abd8b2838de345ed7c0cda467ddb6c3acf73d9cfceef974e"
)
base = root / "experiments/ms_swift/.cache/paired"


def evaluation_folder(arm):
    path = (root / proof["evaluation"][arm]["relative_run_directory"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("evaluation directory escaped the verified artifact root")
    return path


predictions = {
    a: evaluation_folder(a) / "predictions.jsonl" for a in ["base", "short", "long"]
}
for arm, path in predictions.items():
    assert sha(path) == proof["evaluation"][arm]["predictions_sha256"]
retry_prefix = None
if proof.get("prior_short_timeout"):
    prior_file = base / "evaluation/short_cachefix2/run/predictions.jsonl"
    assert sha(prior_file) == proof["prior_short_timeout"]["predictions_sha256"]
    before = report.read_rows(prior_file)
    after = report.read_rows(predictions["short"])
    count = proof["prior_short_timeout"]["child_completed_records"]
    assert [row["id"] for row in before] == [row["id"] for row in after]
    retry_prefix = {
        "previous_completed_records": count,
        "identical_prediction_and_status": sum(
            left["prediction"] == right["prediction"]
            and left["status"] == right["status"]
            for left, right in zip(before[:count], after[:count])
        ),
        "identical_generation_config_hash": sum(
            left.get("generation_config_sha256")
            == right.get("generation_config_sha256")
            for left, right in zip(before[:count], after[:count])
        ),
        "original_predictions_sha256": sha(prior_file),
        "note": "Post-run reproducibility diagnostic only; never used to select an answer or checkpoint.",
    }
metric_root = args.metric_root.resolve()
scored = score_files(
    records,
    predictions,
    out / "scoring",
    longbench_metrics=metric_root / "longbench_metrics.py",
    longbench_metrics_sha256="e22e2a2662e0f7e683137fa3541f64edb6a801e9138d16d2f3459a6ab9941323",
    ruler_metrics=metric_root / "ruler_constants.py",
    ruler_metrics_sha256="6740467c17b8dc06b6b30f4f97e54ce8de81db0dd879f1538d0b6b5727f4bd5f",
)
assert (
    scored["scoring_implementation_sha256"]
    == read(records.parent / "eval_manifest.json")["scorer_sha256"]
)
score_summary = out / "scoring/summary.json"
write_lf(score_summary, score_summary.read_text(encoding="utf-8"))
boolq_formats = {}
frozen_rows = report.read_rows(records)
for arm, path in predictions.items():
    counts = {
        key: 0
        for key in [
            "total",
            "empty_or_failed",
            "invalid_format",
            "valid_wrong",
            "valid_correct",
        ]
    }
    for example, prediction in zip(frozen_rows, report.read_rows(path)):
        assert example["id"] == prediction["id"]
        if example["benchmark"] != "boolq":
            continue
        counts["total"] += 1
        if (
            prediction["status"] not in {"ok", "success"}
            or not prediction["prediction"].strip()
        ):
            counts["empty_or_failed"] += 1
        else:
            parsed = yes_no(prediction["prediction"])
            category = (
                "invalid_format"
                if parsed is None
                else "valid_correct"
                if parsed == example["answers"][0]
                else "valid_wrong"
            )
            counts[category] += 1
    assert (
        counts["total"] == 40
        and sum(v for k, v in counts.items() if k != "total") == 40
    )
    group = [
        r
        for r in scored["groups"][arm]
        if r["benchmark"] == "boolq" and r["length_bucket"] == "all_lengths"
    ]
    assert (
        len(group) == 1
        and abs(group[0]["score_percent"] - counts["valid_correct"] / 40 * 100) < 0.001
    )
    boolq_formats[arm] = counts
data = report.collect(
    packet,
    tokenizer_python=args.tokenizer_python,
    model_assets=args.model_assets,
    public_records=records,
    resource_evidence=root / "evidence",
    training_arms={a: base / a / "plan" for a in ["short", "long"]},
    evaluation_summaries={
        a: evaluation_folder(a) / "summary.json" for a in ["base", "short", "long"]
    },
    score_summary=out / "scoring/summary.json",
)
suite_path = args.suite_summary.resolve()
suite = read(suite_path)
assert suite["schema_version"] == "public-pilot-freeze-v1"
assert suite["records_sha256"] == sha(records)
assert suite["source_manifest_sha256"] == sha(records.parent / "eval_manifest.json")
data["public"]["producer_summary"] = suite
data["actual_swift_template_verified"] = proof["actual_template_verified"]
data["sources"][str(suite_path)] = {
    "sha256": sha(suite_path),
    "bytes": suite_path.stat().st_size,
}
for source in [
    approval_path,
    root / "evidence/final_validation_summary.json",
    Path(__file__),
    repo / "experiments/analysis/plot_paired_pilot.py",
    repo / "experiments/ms_swift/pilot_eval/scoring.py",
    metric_root / "longbench_metrics.py",
    metric_root / "ruler_constants.py",
]:
    data["sources"][str(source)] = {
        "sha256": sha(source),
        "bytes": source.stat().st_size,
    }
figures = report.plots(data, out / "figures")
plt = report.configure_plot()
fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.4))
fig.subplots_adjust(left=0.08, right=0.97, top=0.83, bottom=0.20, wspace=0.25)
series = {}
for arm, color in [("short", "#2563b7"), ("long", "#12877b")]:
    audit = base / arm / "plan/training_audit"
    metrics = [
        json.loads(x)
        for x in (audit / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    dev = [
        json.loads(x)
        for x in (audit / "dev_evaluations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    summary = read(audit / "summary.json")
    assert summary == proof["arms"][arm]["summary"]
    train = [x for x in metrics if "loss" in x]
    best = [x for x in dev if x["step"] == summary["best_step"]]
    assert len(best) == 1
    axes[0].plot(
        [x["step"] for x in train],
        [x["loss"] for x in train],
        "-o",
        label=arm,
        color=color,
    )
    axes[1].plot(
        [x["step"] for x in dev],
        [x["eval_loss"] for x in dev],
        "-o",
        label=arm,
        color=color,
    )
    axes[1].scatter(
        [best[0]["step"]],
        [best[0]["eval_loss"]],
        marker="*",
        s=180,
        color=color,
        zorder=5,
    )
    series[arm] = {"metrics": train, "dev": dev, "summary": summary}
    for name in ["metrics.jsonl", "dev_evaluations.jsonl"]:
        source = audit / name
        data["sources"][str(source)] = {
            "sha256": sha(source),
            "bytes": source.stat().st_size,
        }
for ax, title in zip(
    axes, ["训练 loss：全部已执行更新", "验证 loss：星号为保存的 best"]
):
    ax.set(title=title, xlabel="optimizer step", ylabel="loss")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    ax.set_xticks([2, 4, 6, 8])
fig.suptitle("Gemma 4 E2B · 实际配对 SFT", fontsize=20)
fig.text(
    0.08,
    0.05,
    "仅 seed17 / 32 train + 8 dev；loss 与 best 选优不是长文能力提升的独立证明。",
    fontsize=10,
    color="#60748a",
)
for ext in ["png", "svg"]:
    fig.savefig(out / f"figures/train_dev_best.{ext}", dpi=180, facecolor="white")
plt.close(fig)
blocker = "40组已批准内部 SFT；真实资源、模板、两臂训练和统一公开评测已执行。这里只报告单 seed 小规模预实验，不据此宣称正式能力提升。初次250条缓存加载失败完整保留，不混作模型答题成绩。"
page = report.render_html(data, figures, out / "report.html", blocker)
encoded = base64.b64encode((out / "figures/train_dev_best.png").read_bytes()).decode(
    "ascii"
)
extra = (
    '<section><h2>实际 train / dev / best 记录</h2><p>总更新与选中模型曝光分别见上表。人工批准仅用于内部 SFT，不代表公开数据集发布许可。</p><img alt="实际训练与验证loss以及best" src="data:image/png;base64,'
    + encoded
    + '"></section>'
)
format_rows = "".join(
    "<tr><td>"
    + report.ARM_NAMES[arm]
    + "</td>"
    + "".join(
        f"<td>{counts[key]}</td>"
        for key in ["empty_or_failed", "invalid_format", "valid_wrong", "valid_correct"]
    )
    + "</tr>"
    for arm, counts in boolq_formats.items()
)
extra += (
    "<section><h2>BoolQ 输出格式诊断（事后）</h2><p>每臂固定40题，继续使用原整段严格Yes/No计分。"
    "此分解只帮助区分格式、空输出与有效格式下的答题错误，不改变原分数。</p>"
    "<div class='scroll'><table><thead><tr><th>模型</th><th>空输出或失败</th><th>非空但格式不符</th><th>有效格式答错</th><th>有效格式答对</th></tr></thead><tbody>"
    + format_rows
    + "</tbody></table></div></section>"
)
if proof.get("prior_short_timeout"):
    prior = proof["prior_short_timeout"]
    extra += (
        "<section><h2>独立保留的工程超时尝试</h2><p>"
        f"首次 short 测评在 {prior['max_seconds']} 秒工程时限中完成 {prior['child_completed_records']} / {prior['total']} 题，"
        f"其余 {prior['failed_or_skipped']} 题保留中断或未尝试状态。原预测、摘要、清理和完整分母均未改写。"
        "后续使用相同 best、相同250题及完全一致的请求指纹重新执行，单臂上限5400秒；原六小时总上限不变。"
        "本页三臂质量比较使用明确绑定的最终完整运行；Base的6条空输出继续计零，未重试。</p></section>"
    )
    extra += (
        "<section><h2>重试的一致性审计</h2><p>"
        f"原尝试已生成的 {retry_prefix['previous_completed_records']} 条中，"
        f"{retry_prefix['identical_prediction_and_status']} 条的文本与状态逐字一致；"
        f"{retry_prefix['identical_generation_config_hash']} 条的生成参数哈希一致。"
        "这是事后复现诊断，不用于选择答案或模型。</p></section>"
    )
text = page.read_text(encoding="utf-8")
# 保留候选包当时的 pending 原件；只在本轮报告中明确展示后续人工准入与实测状态。
pending = f"人工批准配对：{data['summary']['approved_pairs']}；training_ready：{str(data['summary']['training_ready']).lower()}。"
assert text.count(pending) == 1
text = text.replace(
    pending,
    "本轮有效人工批准：40 对，内部 SFT；原始候选包生成时为 0 对批准、training_ready=false，历史状态未改写。",
)
text = text.replace("数据候选与工程准备记录", "实际训练与公开评测记录")
text = text.replace("训练候选 / 验证候选（配对）", "训练 / dev 配对（已批准内部SFT）")
text = text.replace(
    "真实训练前仍须由锁定版本的 SWIFT 对全部记录核验。",
    "本轮已由锁定版本的 SWIFT 核验全部 40 对，实际标签与长度检查通过。",
)
text = text.replace("批准后按实际 32 个训练配对执行", "本轮已按实际 32 个训练配对执行")
provenance_heading = "<section><h2>7. 数据与代码来源指纹</h2>"
assert text.count(provenance_heading) == 1
text = text.replace(provenance_heading, extra + provenance_heading)
write_lf(page, text)
for path in (out / "figures").glob("*.svg"):
    write_lf(path, "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n")
manifest = {
    "schema_version": "paired-gpu-visual-report-v1",
    "candidate_summary_approved_pairs_at_generation": data["summary"]["approved_pairs"],
    "effective_internal_training_admission": {
        "pairs": approval["pairs"],
        "scope": approval["approved_scope"],
        "receipt_sha256": sha(approval_path),
        "public_dataset_publication_approved": False,
    },
    "actual_swift_template_verified": True,
    "actual_sft_and_reload_verified": True,
    "public_metric_comparison_measured": True,
    "formal_improvement_demonstrated": False,
    "initial_base_load_failures": 250,
    "prior_short_timeout": proof.get("prior_short_timeout"),
    "retry_prefix_reproducibility": retry_prefix,
    "boolq_format_diagnostics": boolq_formats,
    "primary_comparison_records_per_arm": 250,
    "training": {a: s["summary"] for a, s in series.items()},
    "source_hashes": data["sources"],
    "final_runtime_proof_sha256": sha(root / "evidence/final_validation_summary.json"),
    "outputs": {
        p.relative_to(out).as_posix(): sha(p)
        for p in out.rglob("*")
        if p.is_file() and p.suffix in [".png", ".svg", ".html"]
    },
    "notes": [
        "Original candidate packet and serialized remote plans remain unchanged.",
        "Effective human admission and original generation state are separate.",
        "Selected best exposure may differ even with equal full training budgets.",
        "Single-source cluster confidence intervals remain unavailable; no pooled cross-benchmark accuracy.",
    ],
}
write_lf(
    out / "visualization_manifest.json",
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
)
print(
    json.dumps(
        {
            "report": str(page),
            "training": manifest["training"],
            "score_labels": scored["labels"],
        },
        ensure_ascii=False,
    )
)
