"""Create a pending, original synthetic SFT engineering review packet.

This never approves records, fabricates human reviews, or starts training.
Run from experiments/ms_swift with the independent CPU preparation environment.
"""
from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

from cpt.core import digest, file_hash, new_output, write_json, write_jsonl
from cpt.sft import prepare_sft


LICENSE_CANDIDATE = "LicenseRef-Ariadne-Original-AI-Smoke"
EXAMPLES = [
    ("train", "en", "Amber mixing log",
     "Fictional engineering log AMBER-01. The mixing vessel is named Pine. The operator selected a stirring speed of 180 rpm and a duration of 7 minutes. The backup vessel Cedar was not used. This invented record is for a software training test only.",
     "Which vessel was used, and how long was the stirring duration?", "Pine; 7 minutes."),
    ("train", "zh", "青禾采样日志",
     "虚构工程日志 QH-02。青禾小组在测试架上放置了 12 个蓝色样本盒和 5 个灰色空盒。采样轮次为第 3 轮，保存目录为 trial_qh03。本段为软件训练测试新编的虚构文本，不是真实实验结果。",
     "蓝色样本盒有多少个，采样轮次是什么？", "12 个；第 3 轮。"),
    ("train", "en", "River sensor check",
     "Fictional engineering log RIVER-03. Sensor R8 was checked at a reference temperature of 24 degrees Celsius. The recorded offset was minus 0.4 degrees Celsius. Sensor R9 was listed as a spare and was not checked. These values were invented for an engineering smoke test.",
     "Which sensor was checked and what offset was recorded?", "R8; minus 0.4 degrees Celsius."),
    ("train", "zh", "松塔运行记录",
     "虚构工程日志 ST-04。松塔程序的本次输入文件为 seed_a.jsonl，输出文件为 parsed_a.jsonl。记录中写明已处理 18 行，其中 2 行被标为格式错误。以上内容是新编的软件流程测试材料。",
     "输入文件叫什么，有多少行被标为格式错误？", "seed_a.jsonl；2 行。"),
    ("train", "en", "Quartz storage note",
     "Fictional engineering log QUARTZ-05. The sample rack labelled East held 9 sealed tubes. The rack labelled West held 4 empty tubes. Only the sealed tubes were placed in drawer D6. The log and all quantities are original invented test material.",
     "How many sealed tubes were placed in which drawer?", "9 sealed tubes; drawer D6."),
    ("train", "zh", "星桥计时记录",
     "虚构工程日志 XQ-06。星桥计时器对两段程序进行了演示记录：解析阶段为 14 毫秒，写入阶段为 23 毫秒。日志标签为 demo_xq，重试次数记为 0。数字仅用于工程冒烟测试，不代表实际性能。",
     "写入阶段记为多少毫秒，重试次数是多少？", "23 毫秒；0 次。"),
    ("train", "en", "Meadow packet log",
     "Fictional engineering log MEADOW-07. A demonstration queue accepted 16 packets under route M4. Its retry interval was set to 6 seconds. Route M5 was mentioned in the notes but remained disabled. This is newly authored synthetic software-test text.",
     "Which route was enabled and what retry interval was set?", "M4; 6 seconds."),
    ("train", "zh", "岚石文件检查",
     "虚构工程日志 LS-08。岚石检查器读取了编号为 config_k 的配置，校验模式为 strict。日志将 11 个条目标为通过，将 3 个条目标为待复核。本文是为软件训练流程新编的虚构记录。",
     "校验模式是什么，有多少条目待复核？", "strict；3 个条目。"),
    ("dev", "en", "Harbor display log",
     "Fictional engineering log HARBOR-09. The display controller used theme Violet at a refresh interval of 40 milliseconds. Theme Gold was available but not selected. The operator saved the demo as harbor_preview. This invented record is reserved for the smoke run's development split.",
     "Which theme was selected, and what was the refresh interval?", "Violet; 40 milliseconds."),
    ("dev", "zh", "云帆阀门记录",
     "虚构工程日志 YF-10。云帆演示装置打开了编号 V3 的阀门，持续时间为 9 秒。备用阀门 V4 保持关闭，记录被写入 valve_demo。本文为新编虚构文本，仅保留在工程冒烟运行的 dev 划分。",
     "哪个阀门被打开，持续了多少秒？", "V3；9 秒。"),
]


def generate(output: Path, model_assets: Path):
    out = new_output(output)
    sources, records, display = [], [], []
    for index, (split, language, title, context, question, answer) in enumerate(EXAMPLES, 1):
        sid = f"colab-smoke-source-{index:02d}"
        source = {
            "id": sid, "family_id": sid, "title": title, "language": language,
            "domain": "fictional_engineering_smoke", "kind": "synthetic_report", "split": split,
            "license": LICENSE_CANDIDATE, "source_url": f"urn:ariadne:colab-smoke:20261108:{index:02d}",
            "revision": "original-ai-authored-20261108-v1", "review_status": "pending_human",
            "metadata": {"authorship": "new AI-authored fictional text", "human_verified": False,
                         "license_status": "candidate_pending_human", "public_license_granted": False,
                         "purpose": "engineering_smoke_only", "real_experiment_result": False},
            "blocks": [{"id": "p1", "type": "paragraph", "text": context,
                        "section_path": [title], "page": None}],
        }
        prefix = "仅根据以下虚构日志回答问题，简短作答。" if language == "zh" else "Answer briefly using only the fictional log below."
        prompt = f"{prefix}\n\n{context}\n\n{question}"
        record = {"id": f"colab-smoke-qa-{index:02d}", "split": split, "source_document_ids": [sid],
                  "license": LICENSE_CANDIDATE, "kind": "reviewed_synthetic_qa",
                  "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]}
        sources.append(source)
        records.append(record)
        display.append((source, record, context, question, answer))
    write_jsonl(out / "sources.jsonl", sources)
    write_jsonl(out / "sft_candidates.jsonl", records)
    source_reviews = [{"document_id": d["id"], "document_sha256": digest(d), "split": d["split"],
                       "license": LICENSE_CANDIDATE, "decision": "pending", "reviewer_type": "", "reviewer": "",
                       "reason": "", "license_decision": "pending", "license_evidence": []} for d in sources]
    reviews = [{"record_id": r["id"], "record_sha256": digest(r),
                "source_sha256": {d["id"]: digest(d)}, "decision": "pending", "reviewer_type": "", "reviewer": "",
                "reason": "", "license_decision": "pending", "license_evidence": []} for r, d in zip(records, sources)]
    write_jsonl(out / "source_reviews.pending.jsonl", source_reviews)
    write_jsonl(out / "answer_reviews.pending.jsonl", reviews)
    rights = """# Original synthetic text / rights review candidate

All ten fictional logs, questions and answers were newly authored by the assistant for this user's engineering smoke test on 2026-11-08. No external document, benchmark question or application-user data was copied into this packet. This provenance statement is a candidate supplied by the assistant, not an independent copyright or license determination.

`LicenseRef-Ariadne-Original-AI-Smoke` is an internal identifier for this candidate material. It is NOT an SPDX public license, NOT CC0, and NOT a statement that a human has granted rights or approved training. A human must decide whether the project's intended internal training use is authorized, and record the actual evidence and decision separately. No public redistribution permission is asserted.

The SFT input `kind=reviewed_synthetic_qa` is the existing validator's supported synthetic-QA category; it does NOT assert that review has happened. Actual source and answer review decisions are all pending, training_ready is false, and the admission gate remains closed.

This tiny packet tests encoding, assistant-only loss, optimizer execution, checkpoint writing and reload. Its development set is only a smoke diagnostic. It does not measure generalization, long-context ability, CPT gain or real experimental performance.
"""
    (out / "RIGHTS_CANDIDATE.md").write_text(rights, encoding="utf-8")
    parts = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
             "<title>Gemma Colab SFT：10 条待审核工程样本</title>",
             "<style>body{font:16px/1.65 system-ui,sans-serif;max-width:1050px;margin:32px auto;padding:0 20px;color:#20252b;background:#f8fafc}article{background:white;border:1px solid #cbd5e1;border-radius:10px;padding:20px;margin:20px 0}h1{font-size:28px}h2{font-size:21px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5f9;padding:12px}code{overflow-wrap:anywhere}.pending{color:#9a3412;font-weight:700}.notice{border-left:5px solid #f59e0b;padding:12px;background:#fffbeb}</style>",
             "<h1>Gemma Colab SFT：10 条待审核工程样本</h1>",
             "<p>8 条 train + 2 条 dev；5 条中文 + 5 条英文。每条为新编虚构日志和基于原文的抽取问答。所有内容如下完整展示。</p>",
             "<p class='notice'><strong>当前全部待人工审核，0 条批准。</strong>本包只用于工程冒烟验证，不是正式长文数据、公开基准或人工金标，也不能支持模型能力提升结论。未复用任何现有测评题。</p>",
             f"<p>候选权利标识：<code>{LICENSE_CANDIDATE}</code>。这是内部候选标识，不是公共许可，也未获真人批准。请同时审阅 <a href='RIGHTS_CANDIDATE.md'>完整权利候选说明</a>。</p>",
             "<p>审核内容：原文及答案是否准确对应、是否同意仅用于本次内部工程 SFT、每条来源与答案是否通过。可逐条提出修改/拒绝；任何实际批准需另存真实审核人、理由与许可证据，不覆盖下面的 pending 文件。</p>",
             "<p><a href='sources.jsonl'>来源记录</a> · <a href='sft_candidates.jsonl'>SFT 输入</a> · <a href='source_reviews.pending.jsonl'>来源待审凭据</a> · <a href='answer_reviews.pending.jsonl'>答案待审凭据</a> · <a href='README.md'>使用说明</a></p>"]
    for d, r, context, question, answer in display:
        parts.extend([f"<article id='{escape(r['id'])}'><h2>{escape(r['id'])} — {escape(d['title'])}</h2>",
                      f"<p><strong>{d['split']} / {d['language']}</strong> · <span class='pending'>来源 pending / 答案 pending / 许可 pending</span></p>",
                      f"<h3>完整原文</h3><pre>{escape(context)}</pre><h3>问题</h3><pre>{escape(question)}</pre><h3>候选答案</h3><pre>{escape(answer)}</pre>",
                      f"<p>实际 user 提示前缀：{escape('仅根据以下虚构日志回答问题，简短作答。' if d['language'] == 'zh' else 'Answer briefly using only the fictional log below.')}</p>",
                      f"<p>来源 SHA256：<code>{digest(d)}</code><br>答案记录 SHA256：<code>{digest(r)}</code></p></article>"])
    parts.append("</html>")
    (out / "index.html").write_text("\n".join(parts), encoding="utf-8")
    manifest = prepare_sft(out / "sft_candidates.jsonl", out / "sources.jsonl", model_assets, out / "preview",
                           reviews_path=out / "answer_reviews.pending.jsonl", source_reviews_path=out / "source_reviews.pending.jsonl",
                           allowed_licenses=(), sequence_length=4096, mode="preview")
    readme = f"""# Colab SFT engineering review packet — 2026-11-08

Open `index.html` for all ten complete source/question/answer examples. 8 train + 2 dev; 5 English + 5 Chinese. The material is original AI-authored fictional text, all human and license decisions remain pending. Review `RIGHTS_CANDIDATE.md` before any rights decision. No public license is claimed.

`preview/` was produced by the project's existing native-tokenizer SFT validator. It has `{manifest['approved_records']}` approved records and `training_ready=false`. It is not executable training data. The accepted schema category `reviewed_synthetic_qa` does not grant review or override receipts.

Run from `experiments/ms_swift/`. On Colab, copy source JSONL and actual completed receipts and generate NEW exports/plans at the remote paths; do not patch absolute paths in this Windows preview. Download the pinned model assets there first:

```bash
python -m cpt fetch-tokenizer --output .cache/model_assets/gemma4-e2b
```

After real human review ONLY, separately save `answer_reviews.approved.jsonl` and `source_reviews.approved.jsonl`. Every approved row must retain the exact hashes and contain `decision=approved`, actual `reviewer_type` (`human` or `ai_assisted_human`), nonempty actual `reviewer` and `reason`, `license_decision=approved`, and nonempty actual `license_evidence`. Do not invent identities or convert pending automatically. If content is edited, regenerate hashes and obtain a fresh review.

```bash
python -m cpt sft-prepare \\
  --input .cache/colab_sft_review_20261108_v1/sft_candidates.jsonl \\
  --source-corpus .cache/colab_sft_review_20261108_v1/sources.jsonl \\
  --reviews .cache/colab_sft_review_20261108_v1/answer_reviews.approved.jsonl \\
  --source-reviews .cache/colab_sft_review_20261108_v1/source_reviews.approved.jsonl \\
  --allow-license {LICENSE_CANDIDATE} \\
  --model-assets .cache/model_assets/gemma4-e2b --sequence-length 4096 \\
  --mode training --output .cache/colab_sft_training_20261108_v1
python -m cpt plan \\
  --prepared .cache/colab_sft_training_20261108_v1 \\
  --recipe configs/gemma4_e2b_sft_lora_4k_smoke.json \\
  --output .cache/plans/colab_sft_20261108_v1
python -m cpt.profile train \\
  --plan .cache/plans/colab_sft_20261108_v1/launch_plan.json \\
  --output .cache/profiles/colab_sft_20261108_v1 --max-seconds 900 --interval 2 --execute
```

The recipe's 4096 maximum is a ceiling; these ten records are short and do not constitute a 4K training test. Twenty optimizer steps reuse eight short train examples. Any loss change is an engineering diagnostic, not a capability claim. The development examples are not a public benchmark. Save actual runtime environment, logs, outcomes, and adapter reload results separately.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    summary = {"schema_version": "colab-sft-review-packet-v1", "created_date": "2026-11-08", "authorship": "AI-authored original fictional text",
               "purpose": "engineering_smoke_only", "split_counts": {"train": 8, "dev": 2}, "language_counts": {"en": 5, "zh": 5},
               "source_review_counts": {"pending": 10, "approved": 0}, "answer_review_counts": {"pending": 10, "approved": 0},
               "license_candidate": LICENSE_CANDIDATE, "license_verified": False, "training_ready": False,
               "preview_counts": manifest["counts"], "approved_records": manifest["approved_records"],
               "files": {str(p.relative_to(out)).replace("\\", "/"): file_hash(p) for p in sorted(out.rglob("*")) if p.is_file()}}
    write_json(out / "packet_manifest.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-assets", type=Path, required=True)
    args = parser.parse_args()
    result = generate(args.output, args.model_assets)
    print({k: v for k, v in result.items() if k != "files"})
