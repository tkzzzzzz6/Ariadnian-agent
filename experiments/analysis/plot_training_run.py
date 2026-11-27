"""Render source-verified Gemma SFT smoke evidence without loading model weights.

Run from the repository root with an existing matplotlib installation:
    python experiments/analysis/plot_training_run.py

The source JSONL/checkpoint files remain ignored. Outputs contain only aggregate
training/resource measurements, never prompts, answers, or individual predictions.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import platform


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path, root):
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(Path(path).resolve())


def collect(summary_path, artifacts, repo):
    summary = read_json(summary_path)
    if (summary.get("schema_version") != "colab-gemma-sft-smoke-result-v1"
            or summary.get("training", {}).get("global_step") != 20
            or summary.get("data", {}).get("train_records") != 8
            or summary.get("data", {}).get("dev_records") != 2
            or summary.get("training", {}).get("actual_encoded_length_range") != [95, 119]
            or summary.get("training", {}).get("max_length") != 4096
            or summary.get("verification", {}).get("checkpoint_reload_verified") is not True
            or summary.get("lifecycle", {}).get("owned_session_stopped") is not True):
        raise ValueError("this report layout is bound to the completed 2026-11-08 short SFT smoke; use a new layout for other protocols")
    evidence = summary["evidence_sha256"]
    sources = {"summary": {"path": relative(summary_path, repo), "sha256": sha256(summary_path)}}

    def source(suffix):
        matches = [name for name in evidence if name.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f"expected one evidence file ending in {suffix}")
        name = matches[0]
        path = (artifacts / name).resolve()
        if not path.is_relative_to(artifacts.resolve()) or sha256(path) != evidence[name]:
            raise ValueError("evidence hash/path mismatch: " + name)
        sources[suffix] = {"path": relative(path, repo), "sha256": evidence[name],
                           "verified_against_existing_summary": True}
        return path

    trainer = read_json(source("trainer_state.json"))
    logging_path = source("logging.jsonl")
    runtime_rows = [json.loads(line) for line in logging_path.read_text().splitlines() if line.strip()]
    profile_path = source("resource_summary.json")
    profile = read_json(profile_path)
    template = read_json(source("template_verification.json"))
    gpu_path = profile_path.parent / "gpu_samples.jsonl"
    gpu_rows = [json.loads(line) for line in gpu_path.read_text().splitlines() if line.strip()]
    # This file was not separately hashed in the original summary; bind its
    # actual bytes here and cross-check its counts and peak against that summary.
    sources["gpu_samples.jsonl"] = {"path": relative(gpu_path, repo), "sha256": sha256(gpu_path),
                                   "verified_against_existing_summary": False,
                                   "cross_checked": "row count, failure count and board-memory maximum"}
    steps = [{"step": row["step"], "loss": row["loss"], "teacher_forced_token_accuracy": row["token_acc"]}
             for row in trainer["log_history"] if "loss" in row]
    dev = [{"step": row["step"], "loss": row["eval_loss"],
            "teacher_forced_token_accuracy": row["eval_token_acc"]}
           for row in trainer["log_history"] if "eval_loss" in row]
    if [row["step"] for row in steps] != list(range(1, trainer["global_step"] + 1)):
        raise ValueError("missing, duplicate or unordered training steps")
    start = datetime.fromisoformat(profile["created_utc"])
    samples = []
    for row in gpu_rows:
        if row.get("error") or len(row.get("gpus", [])) != 1:
            used = None
        else:
            used = row["gpus"][0]["memory.used"] / 1024
        samples.append({"elapsed_seconds": (datetime.fromisoformat(row["time_utc"]) - start).total_seconds(),
                        "board_used_gib": used, "error": row.get("error")})
    measured = profile["resource_measurement"]
    failed = sum(row["board_used_gib"] is None for row in samples)
    peak = max(row["board_used_gib"] for row in samples if row["board_used_gib"] is not None)
    expected_boards = list(measured["boards"].values())
    if (len(expected_boards) != 1 or len(samples) != measured["samples"]
            or failed != measured["failed_samples"]
            or abs(peak * 1024 - expected_boards[0]["sampled_peak_board_used_mib"]) > 1e-6):
        raise ValueError("GPU series disagrees with original resource summary")
    end = runtime_rows[-1]
    timing = next(row for row in end["log_history"] if "train_runtime" in row)
    datasets = template["datasets"]
    lengths = {"minimum": min(row["min_input_length"] for row in datasets.values()),
               "maximum": max(row["max_input_length"] for row in datasets.values()),
               "configured_cap": template["max_length"]}
    result = {
        "schema_version": "training-visual-series-v1",
        "scope": "short synthetic SFT engineering smoke; no long-context capability result",
        "training_steps": steps, "observed_dev_points": dev, "gpu_samples": samples,
        "gpu_failed_samples": failed, "sampled_peak_board_used_gib": peak,
        "swift_reported_memory_gib": end["memory"], "actual_input_lengths": lengths,
        "train_records": datasets["train"]["records"], "dev_records": datasets["dev"]["records"],
        "trainer_runtime_seconds": timing["train_runtime"], "profile_wall_seconds": profile["wall_seconds"],
        "checkpoint_reload_verified": summary["verification"]["checkpoint_reload_verified"],
        "owned_session_stopped": summary["lifecycle"]["owned_session_stopped"],
        "source_hashes": sources,
        "plot_semantics": {
            "loss": "unsmoothed per-optimizer-step logs; log y axis; dev shown only at observed steps",
            "accuracy": "teacher-forced token accuracy, not free-generation answer accuracy",
            "memory": "discrete whole-board samples; distinct from the separately reported SWIFT memory metric",
            "length": "full SWIFT template input min/max, not declared context capacity",
            "no_inference": "loss decline and this tiny dev set cannot establish generalization or long-context gains"
        }
    }
    return result


def plot(series, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    colors = {"ink": "#172b4d", "blue": "#2563b7", "orange": "#c46219", "green": "#167b72",
              "muted": "#64748b", "gray": "#e2e8f0"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 14,
                         "axes.labelsize": 11, "axes.titleweight": "bold", "svg.hashsalt": "ariadne-sft-v1",
                         "axes.edgecolor": "#cbd5e1", "axes.labelcolor": colors["ink"],
                         "xtick.color": colors["muted"], "ytick.color": colors["muted"],
                         "text.color": colors["ink"], "savefig.facecolor": "white"})
    fig, axs = plt.subplots(2, 2, figsize=(14, 9.2))
    fig.subplots_adjust(left=.075, right=.965, bottom=.13, top=.80, hspace=.53, wspace=.23)
    fig.text(.075, .948, "GEMMA 4 E2B  /  COLAB A100", fontsize=12, weight="bold", color=colors["green"])
    fig.text(.075, .904, "SFT engineering smoke: measured run", fontsize=24, weight="bold")
    fig.text(.075, .868, "2026-11-08  |  BF16 + LoRA r16  |  8 train + 2 dev  |  20 optimizer steps",
             fontsize=12, color=colors["muted"])
    fig.text(.965, .943, "SHORT INPUTS ONLY", ha="right", fontsize=10, weight="bold", color=colors["orange"],
             bbox={"facecolor": "#fff5eb", "edgecolor": "none", "pad": 8})

    for ax in axs.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e8edf3", linewidth=.8)
        ax.set_axisbelow(True)

    steps = [row["step"] for row in series["training_steps"]]
    losses = [row["loss"] for row in series["training_steps"]]
    ax = axs[0, 0]
    ax.set_title("A   Loss falls on the small repeated dataset", loc="left", pad=14)
    ax.plot(steps, losses, color=colors["blue"], lw=2.2, marker="o", ms=3.5, label="Train: 20 observed steps")
    for point in series["observed_dev_points"]:
        ax.scatter(point["step"], point["loss"], color=colors["orange"], marker="D", s=60, zorder=5,
                   label=f"Dev: observed only at step {point['step']}")
    ax.set_yscale("log")
    ax.set(xlabel="Optimizer step", ylabel="Cross-entropy loss (log scale)", xlim=(.4, 21))
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.annotate(f"{losses[-1]:.6f}", (steps[-1], losses[-1]), xytext=(-58, 4), textcoords="offset points",
                fontsize=10, color=colors["blue"])

    ax = axs[0, 1]
    ax.set_title("B   Teacher-forced token accuracy", loc="left", pad=14)
    ax.plot(steps, [row["teacher_forced_token_accuracy"] for row in series["training_steps"]],
            color=colors["blue"], marker="o", ms=3.5, lw=2.2, label="Train tokens")
    for point in series["observed_dev_points"]:
        ax.scatter(point["step"], point["teacher_forced_token_accuracy"], marker="D", s=60,
                   color=colors["orange"], zorder=5, label=f"Dev tokens at step {point['step']}")
    ax.set(xlabel="Optimizer step", ylabel="Token accuracy", ylim=(0, 1.08), xlim=(.4, 21))
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.text(.03, .13, "Not free-generation QA accuracy", transform=ax.transAxes, fontsize=10,
            color=colors["muted"])

    ax = axs[1, 0]
    ax.set_title("C   GPU memory during the profiled process", loc="left", pad=14)
    times = [row["elapsed_seconds"] for row in series["gpu_samples"]]
    memory = [row["board_used_gib"] if row["board_used_gib"] is not None else np.nan for row in series["gpu_samples"]]
    ax.plot(times, memory, color=colors["green"], lw=1.8, marker=".", ms=3.2, label="Whole-board sampled usage")
    ax.fill_between(times, memory, 0, color=colors["green"], alpha=.07)
    ax.axhline(series["swift_reported_memory_gib"], color=colors["orange"], lw=1.5, ls="--",
               label=f"SWIFT-reported metric: {series['swift_reported_memory_gib']:.2f} GiB")
    ax.set(xlabel="Seconds from profile creation (includes startup)", ylabel="Memory (GiB)",
           xlim=(0, max(times) + 3), ylim=(0, 12.3))
    ax.text(.97, .93, f"Sampled maximum: {series['sampled_peak_board_used_gib']:.2f} GiB",
            transform=ax.transAxes, ha="right", color=colors["green"], fontsize=10, weight="bold")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.text(.03, .92, f"{len(times)} samples / {series['gpu_failed_samples']} failed", transform=ax.transAxes,
            fontsize=9, color=colors["muted"])

    ax = axs[1, 1]
    ax.set_title("D   What context length was actually tested?", loc="left", pad=14)
    lengths = series["actual_input_lengths"]
    ax.barh([1, 0], [lengths["maximum"], lengths["configured_cap"]], height=.40,
            color=[colors["blue"], colors["gray"]], zorder=3)
    ax.set_yticks([])
    ax.set(xlabel="Full template input tokens", xlim=(0, lengths["configured_cap"] * 1.1), ylim=(-.65, 1.75))
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#e8edf3")
    ax.text(lengths["maximum"] + 80, 1, f"Actual: {lengths['maximum']}  (range {lengths['minimum']}–{lengths['maximum']})",
            va="center", fontsize=11, color=colors["blue"], weight="bold")
    ax.text(80, 0, "Configured cap", va="center", fontsize=11)
    ax.text(lengths["configured_cap"] - 65, 0, f"{lengths['configured_cap']:,}", va="center", ha="right", fontsize=11)
    ax.text(.03, .91, "No 4K training sequence was measured", transform=ax.transAxes,
            color=colors["orange"], fontsize=10)

    fig.text(.075, .058, "Verified: training completed → adapter saved → independent reload + finite forward pass → artifacts downloaded",
             fontsize=10.5, color=colors["green"])
    fig.text(.075, .029, "Evidence boundary: 2 short synthetic dev cases; no pre-training baseline or public benchmark. Sampled board memory is not an allocator peak.",
             fontsize=9.5, color=colors["muted"])
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / "sft_overview.png", output / "sft_overview.svg"]
    fig.savefig(paths[0], dpi=180)
    fig.savefig(paths[1], metadata={"Date": None, "Title": "Gemma 4 E2B SFT engineering smoke measurements"})
    plt.close(fig)
    return paths, matplotlib.__version__


def render_html(series, svg_path, output):
    svg = svg_path.read_text(encoding="utf-8")
    svg = svg[svg.index("<svg"):]
    rows = "".join(f"<tr><td>{r['step']}</td><td>{r['loss']:.8f}</td><td>{r['teacher_forced_token_accuracy']:.2%}</td></tr>"
                   for r in series["training_steps"])
    provenance = "".join(f"<tr><td>{html.escape(name)}</td><td><code>{value['sha256']}</code></td></tr>"
                         for name, value in series["source_hashes"].items())
    dev = series["observed_dev_points"][-1]
    page = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gemma 4 E2B · Colab SFT 可视化记录</title>
<style>
:root {{color-scheme:light; --ink:#172b4d; --muted:#64748b; --line:#e2e8f0; --teal:#167b72}}
* {{box-sizing:border-box}} body {{margin:0;background:#f3f6fa;color:var(--ink);font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif}}
main {{max-width:1180px;margin:32px auto;padding:0 24px 48px}} h1 {{font-size:32px;line-height:1.3;margin:8px 0 16px}} h2 {{font-size:22px;margin:0 0 14px}}
.eyebrow {{letter-spacing:.08em;color:var(--teal);font-size:13px;font-weight:700}} .muted {{color:var(--muted)}}
.lead {{max-width:920px}} .notice {{border-left:4px solid #c46219;background:#fff5eb;padding:14px 18px;margin:20px 0}}
.cards {{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}} .card,.panel {{background:white;border:1px solid var(--line);border-radius:12px;padding:18px}}
.card span {{display:block;font-size:13px;color:var(--muted)}} .card strong {{display:block;font-size:26px;font-weight:650;margin:3px 0}}
.panel {{margin:18px 0;padding:24px}} figure {{margin:0}} figure svg {{display:block;width:100%;height:auto}} figcaption {{font-size:14px;color:var(--muted);margin-top:12px}}
.columns {{display:grid;grid-template-columns:1fr 1fr;gap:24px}} ul {{padding-left:22px;margin:0}} li {{margin:7px 0}} table {{border-collapse:collapse;width:100%;font-size:14px}} th,td {{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left}} th {{background:#f8fafc}} code {{font:12px ui-monospace,monospace;overflow-wrap:anywhere}} summary {{cursor:pointer;font-weight:650;margin:6px 0 12px}} button {{float:right;background:var(--ink);color:white;border:0;border-radius:8px;padding:9px 14px;cursor:pointer}} a {{color:#2563b7}}
@media(max-width:760px) {{main {{padding:0 14px}} .cards {{grid-template-columns:repeat(2,1fr)}} .columns {{grid-template-columns:1fr}} .panel {{padding:15px}} h1 {{font-size:26px}}}}
@media print {{body {{background:white}} main {{margin:0;max-width:none}} button {{display:none}} .panel {{break-inside:avoid}} details {{display:none}}}}
</style></head><body><main>
<button onclick="window.print()">打印 / 保存 PDF</button>
<div class="eyebrow">ARIADNE EXPERIMENT RECORD · 2026-11-08</div>
<h1>Gemma 4 E2B：SFT 链路已经跑通</h1>
<p class="lead">这份记录展示 Colab A100 40GB 上真实发生的训练与资源采样。数据来自经用户审核批准的 8 条训练、2 条验证样本；所有原始凭据和检查点保持本地，页面只嵌入聚合指标。</p>
<div class="notice"><strong>本次结论范围：</strong>完成训练、保存、独立重载和产物下载。实际输入只有 95–119 token，不能据此声称长文能力提升。4K 是配置上限，2 条 dev 的低 loss 也不构成泛化证据。</div>
<div class="cards"><div class="card"><span>Optimizer steps</span><strong>20 / 20</strong><span>8 条训练样本重复 20 epochs</span></div>
<div class="card"><span>Trainer 训练时间</span><strong>{series['trainer_runtime_seconds']:.2f} s</strong><span>profile 全流程 {series['profile_wall_seconds']:.2f} s</span></div>
<div class="card"><span>整卡显存采样最大值</span><strong>{series['sampled_peak_board_used_gib']:.2f} GiB</strong><span>{len(series['gpu_samples'])} 次采样 / {series['gpu_failed_samples']} 次失败</span></div>
<div class="card"><span>保存与独立重载</span><strong>通过</strong><span>前向 logits 有限；A100 已释放</span></div></div>
<section class="panel"><figure role="img" aria-label="四张图：训练 loss、teacher-forced token accuracy、显存采样曲线、实际输入长度与配置上限。">{svg}<figcaption>训练点均为原始逐步日志，没有平滑。dev 只在第 20 步评估一次，图中只显示一个点；没有补出 dev 曲线或训练前基线。显存横轴从 profile 创建时刻起算，含加载与启动检查。</figcaption></figure></section>
<section class="panel columns"><div><h2>如何读这些数字</h2><ul>
<li>训练 loss 从 {series['training_steps'][0]['loss']:.4f} 下降到 {series['training_steps'][-1]['loss']:.6f}，说明小数据上的优化在运行。</li>
<li>dev loss 为 {dev['loss']:.6f}，只来自 2 条相似的合成样本。</li>
<li>100% 是已给定正确前缀的 token 命中率，不能视为自由生成回答准确率。</li>
<li>SWIFT 日志显存 {series['swift_reported_memory_gib']:.4f} GiB 与整卡采样是两种定义；离散采样可能漏掉尖峰。</li></ul></div>
<div><h2>下一阶段必须补的证据</h2><ul>
<li>冻结训练前基线与独立公开测试集，保持 prompt、输出预算和评分协议一致。</li>
<li>真实 4K / 8K / 16K 输入及跨段、跨文档任务；不能只把配置长度改大。</li>
<li>短上下文训练对照、长上下文训练对照、多随机种子及置信区间。</li>
<li>长文提升与短任务退化同时报告，保留失败和超限分母。</li></ul></div></section>
<section class="panel"><details><summary>查看 20 步聚合指标</summary><table><thead><tr><th>Step</th><th>Train loss</th><th>Teacher-forced token accuracy</th></tr></thead><tbody>{rows}</tbody></table></details>
<details><summary>查看来源哈希与复现方式</summary><p>脚本：<code>python experiments/analysis/plot_training_run.py</code>。源数据存在本机 Git 忽略目录，图表重建前核验原 summary 中的证据哈希；GPU 逐次采样额外绑定其文件哈希，并核对采样数量、失败数量和峰值。</p><table><thead><tr><th>来源</th><th>SHA-256</th></tr></thead><tbody>{provenance}</tbody></table></details></section>
<p class="muted">本页自包含：没有 CDN、远端脚本或网络请求。SVG 图、正文和指标均已嵌入，可离线打开、复制或打印。正式实验方案请参见仓库中的后续研究设计文档。</p>
</main></body></html>'''
    output.write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[2]
    parser.add_argument("--summary", type=Path, default=repo / "experiments/results/colab_sft_20261108/summary.json")
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = (args.output or args.summary.parent).resolve()
    source_summary = read_json(args.summary)
    artifacts = (args.artifacts or (repo / source_summary["artifacts"]["archive"]).with_suffix("")).resolve()
    series = collect(args.summary.resolve(), artifacts, repo)
    output.mkdir(parents=True, exist_ok=True)
    figures, mpl_version = plot(series, output / "figures")
    series_path = output / "visualization_series_summary.json"
    series_path.write_text(json.dumps(series, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path = output / "report.html"
    render_html(series, figures[1], report_path)
    outputs = [*figures, series_path, report_path]
    manifest = {
        "schema_version": "training-visualization-manifest-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator": {"path": relative(Path(__file__), repo), "sha256": sha256(Path(__file__)),
                      "python": platform.python_version(), "matplotlib": mpl_version},
        "sources": series["source_hashes"],
        "outputs": {relative(path, output): {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in outputs},
        "checks": {"training_steps": len(series["training_steps"]), "observed_dev_points": len(series["observed_dev_points"]),
                   "gpu_samples": len(series["gpu_samples"]), "failed_gpu_samples": series["gpu_failed_samples"],
                   "all_bound_source_hashes_match": True, "raw_dataset_text_exported": False},
        "limitations": list(series["plot_semantics"].values())
    }
    (output / "visualization_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "figures": [str(path) for path in figures], "checks": manifest["checks"]}, indent=2))


if __name__ == "__main__":
    main()
