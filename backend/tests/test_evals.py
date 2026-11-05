"""评测协议、复现记录与离线流程的回归测试；不联网、不需要 API Key。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent

# 子进程里 mock 掉 litellm.completion，走“真实模型”分支（关键词扩展、延迟统计、调用日志）
_MOCK = r'''
import json, sys, types, litellm
def completion(**kw):
    system = kw["messages"][0]["content"]
    user = kw["messages"][-1]["content"]
    if "判断给定的文献片段" in system:
        text = json.dumps({"verdict": "supports" if "neural" in user.lower() or "retriev" in user.lower() else "unrelated", "reason": "mock"})
    elif "英文学术检索关键词" in system:
        text = "retrieval augmented generation dense passage residual transformer attention"
    elif "审稿人" in system:
        first = user.split("段落（[引用] 表示该处已有引用）：\n", 1)[1].split("。", 1)[0] + "。"
        text = json.dumps({"missing": [{"sentence": first, "reason": "mock"}], "rewrite": None}, ensure_ascii=False)
    else:
        text = "这是续写。"
    msg = types.SimpleNamespace(content=text)
    usage = types.SimpleNamespace(prompt_tokens=100, completion_tokens=10)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=usage)
litellm.completion = completion
from evals.__main__ import main
sys.exit(main(sys.argv[1:]))
'''


def _run(args: list[str], code: str | None = None, tmp_path: Path | None = None) -> tuple[int, str]:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    if tmp_path is not None:
        env["ARIA_DATA_DIR"] = str(tmp_path)
    env.pop("LLM_FAKE", None)
    cmd = [sys.executable, "-c", code, *args] if code else [sys.executable, "-m", "evals", *args]
    p = subprocess.run(cmd, cwd=BACKEND, env=env, capture_output=True, text=True, encoding="utf-8", timeout=300)
    return p.returncode, p.stdout + p.stderr


def test_eval_offline_fixture(tmp_path):
    code, out = _run(["--offline", "--source", "fixture", "--no-save"], tmp_path=tmp_path)
    assert code == 0, out
    assert "引用判断  n=6" in out
    assert "假引用检出" in out and "检索（语料 6 段）" in out and "缺引用检测" in out


def test_eval_model_path_with_mocked_litellm(tmp_path):
    (tmp_path / "settings.json").write_text(json.dumps({"LLM_API_KEY": "sk-test"}), encoding="utf-8")
    code, out = _run(["--source", "fixture", "--model", "openai/mock", "--no-save", "--workers", "2"], code=_MOCK, tmp_path=tmp_path)
    assert code == 0, out
    assert "真实模型" in out
    assert "zh_bm25_expand" in out          # 有模型时才会跑跨语言扩展
    assert "行内续写  P50" in out
    assert "模型调用" in out


def test_metrics():
    sys.path.insert(0, str(BACKEND))
    from evals.metrics import pct, prf, ranking_metrics

    assert prf(2, 1, 1) == {"precision": 0.667, "recall": 0.667, "f1": 0.667, "tp": 2, "fp": 1, "fn": 1}
    m = ranking_metrics([(["a", "b"], {"b"}), (["c"], {"x"})])
    assert m["recall@1"] == 0.0 and m["recall@5"] == 0.5 and m["mrr@10"] == 0.25
    assert pct([10, 20, 30, 40], 50) == 20 and pct([10, 20, 30, 40], 95) == 40


def test_retrieval_distinguishes_partial_and_complete_evidence():
    from evals.metrics import ranking_metrics

    # Query 1 finds only half its gold; query 2 finds all; query 3 has no gold
    # and cannot silently inflate either hit or complete-coverage metrics.
    m = ranking_metrics([(["a", "a", "irrelevant"], {"a", "b"}), (["c"], {"c"}), ([], set())])
    assert m["hit@5"] == 1.0
    assert m["recall@5"] == 0.75
    assert m["complete@5"] == 0.5
    assert m["n_scored"] == 2 and m["n_unscorable"] == 1


def test_unknown_and_invalid_cannot_score_as_correct_negatives():
    from evals.tasks import _judge_metrics

    m = _judge_metrics([True, False, False, False, True, False],
                       ["supports", "unrelated", "unknown", "invalid", "partial",
                        {"verdict": "unknown", "status": "failure"}])
    assert m["n_correct"] == 2 and m["accuracy"] == 0.333
    assert m["valid_accuracy"] == 0.667 and m["coverage"] == 0.5
    assert m["n_abstained"] == 1 and m["n_failed"] == 2
    abstentions = _judge_metrics([False, False], ["unknown", "unknown"])
    assert abstentions["accuracy"] == 0.0 and abstentions["valid_accuracy"] is None


def test_call_failures_are_traceable_and_scrubbed(monkeypatch):
    from app import tracing
    from evals.tasks import _verdict_call

    monkeypatch.setattr(tracing, "_secrets", lambda: {"private-test-key"})
    def fail():
        raise RuntimeError("private-test-key https://user:pass@host.test/v1?token=private-query Authorization=Bearer exposed")
    outcome = _verdict_call(fail, "failure-case")
    assert outcome["status"] == "failure" and outcome["error_type"] == "RuntimeError"
    assert all(secret not in outcome["error"] for secret in ("private-test-key", "private-query", "user:pass", "exposed"))
    assert _verdict_call(lambda: {"verdict": []}, "invalid-case")["status"] == "failure"


def test_review_requires_human_metadata_and_matching_content(tmp_path):
    from evals.datasets import load_reviewed_negatives
    from evals.provenance import stable_id

    review = {"candidate_id": stable_id("negative", "claim", "evidence"), "claim": "claim", "evidence": "evidence",
              "label": "not_supported", "review_status": "unreviewed", "reviewer": ""}
    path = tmp_path / "review.jsonl"
    path.write_text(json.dumps(review) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="human_reviewed"):
        load_reviewed_negatives(path)
    review.update(review_status="human_reviewed", reviewer="reviewer A")
    path.write_text(json.dumps(review) + "\n", encoding="utf-8")
    assert len(load_reviewed_negatives(path)) == 1
    review["evidence"] += " altered"
    path.write_text(json.dumps(review) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match content"):
        load_reviewed_negatives(path)


def test_saved_manifest_candidates_and_reviewed_round_trip(tmp_path):
    settings = {"LLM_API_KEY": "sk-eval-private-test", "LLM_API_BASE": "https://user:password@example.test/v1?token=private-query"}
    (tmp_path / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    results = tmp_path / "results"
    args = ["--offline", "--source", "fixture", "--tasks", "judge,fake_cite,retrieval", "--seed", "17",
            "--results-dir", str(results)]
    code, out = _run(args, tmp_path=tmp_path)
    assert code == 0, out
    first_path = next(p for p in results.glob("*.json") if not p.name.endswith(".summary.json"))
    text = first_path.read_text(encoding="utf-8")
    assert "sk-eval-private-test" not in text and "private-query" not in text and "user:password" not in text
    result = json.loads(text)
    assert result["protocol_version"] == "ariadne-eval-v2"
    provenance = result["provenance"]
    assert provenance["git"]["commit"] and isinstance(provenance["git"]["dirty"], bool)
    assert provenance["seed"] == 17 and provenance["dataset_files"] and provenance["prompt_sources_sha256"]
    assert provenance["model_requests"] and all(len(r["messages_sha256"]) == 64 for r in provenance["model_requests"])
    assert result["tasks"]["judge"]["coverage"] == 1.0  # paper stub must have an id
    fake = result["tasks"]["fake_cite"]
    assert fake["headline"]["n_reviewed"] == 0 and fake["headline"]["detect_reviewed"] is None
    assert "detect_random" not in fake["headline"]
    assert "not gold performance" in fake["candidate_diagnostics"]["label_scope"]
    candidate_file = results / result["negative_candidates_file"]
    candidates = [json.loads(line) for line in candidate_file.read_text(encoding="utf-8").splitlines()]
    assert candidates and all(c["label"] is None and c["review_status"] == "unreviewed" for c in candidates)
    # Use a full English candidate so its evidence belongs to the same run corpus.
    candidate = fake["negative_candidates"][0]
    candidate.update(label="not_supported", review_status="human_reviewed", reviewer="test reviewer")
    reviewed_path = tmp_path / "reviewed.jsonl"
    reviewed_path.write_text(json.dumps(candidate) + "\n", encoding="utf-8")
    code, out = _run(args + ["--reviewed-negatives", str(reviewed_path)], tmp_path=tmp_path)
    assert code == 0, out
    second = json.loads(max((p for p in results.glob("*.json") if not p.name.endswith(".summary.json")), key=lambda p: p.name).read_text(encoding="utf-8"))
    assert second["tasks"]["fake_cite"]["headline"]["n_reviewed"] == 1
    assert second["provenance"]["bundle_sha256"] != provenance["bundle_sha256"]
    assert first_path.read_text(encoding="utf-8") == text  # previous JSON is never rewritten


def test_raw_invalid_verdict_is_a_failure_before_application_coercion(tmp_path):
    (tmp_path / "settings.json").write_text(json.dumps({"LLM_API_KEY": "sk-test"}), encoding="utf-8")
    code = r'''
import sys, types, litellm
def completion(**kw):
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='{"verdict":"invalid-value"}'))],
                                 usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=5))
litellm.completion = completion
from evals.__main__ import main
sys.exit(main(sys.argv[1:]))
'''
    results = tmp_path / "invalid-results"
    status, out = _run(["--source", "fixture", "--model", "openai/mock", "--tasks", "judge", "--results-dir", str(results)],
                       code=code, tmp_path=tmp_path)
    assert status == 0, out  # run completes and records all sample failures
    result = json.loads(next(p for p in results.glob("*.json") if not p.name.endswith(".summary.json")).read_text(encoding="utf-8"))
    judge = result["tasks"]["judge"]
    assert judge["accuracy"] == 0.0 and judge["n_failed"] == judge["n"] == 6
    assert judge["coverage"] == 0.0 and judge["valid_accuracy"] is None
    assert all(c["error_type"] == "InvalidOutput" for c in judge["cases"])


def test_legacy_index_is_separated_and_never_rewrites_historical_json(tmp_path):
    from evals.report import write_index

    historical = BACKEND / "evals" / "results" / "20261015-0554_offline.json"
    original = historical.read_bytes()
    (tmp_path / historical.name).write_bytes(original)
    write_index(tmp_path)
    index = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "v1 历史结果（旧口径）" in index and "旧 Hit@5" in index and "unknown" in index
    assert (tmp_path / historical.name).read_bytes() == original


def test_public_summary_omits_nested_content_and_local_paths_and_index_prefers_it(tmp_path):
    from evals.report import write_index, write_summary

    private = "PRIVATE_CASE_TEXT_DO_NOT_PUBLISH"
    path = tmp_path / "20261105-000000-000000_test.json"
    result = {"protocol_version": "ariadne-eval-v2", "mode": "offline", "models": {"judge": "mock"},
              "created": "2026-11-05T00:00:00+00:00", "seconds": 1.2, "data": {"pairs": 2},
              "tasks": {"judge": {"n": 2, "accuracy": 0.5, "coverage": 1.0,
                                  "cases": [{"claim": private, "evidence": private}],
                                  "nested": {"raw_output": private, "unknown_free_text": private, "count": 3}},
                        "fake_cite": {"negative_candidates": [{"claim": private}], "headline": {"n_reviewed": 0}}},
              "llm_usage": {"judge": {"calls": 2, "prompt_tokens": 20, "completion_tokens": 5}},
              "provenance": {"git": {"commit": "abc123", "dirty": True, "status": [private]},
                             "config": {"endpoint": private}, "sources": [{"name": private}],
                             "dataset_files": [{"name": private}], "bundle_sha256": "a" * 64,
                             "model_requests": [{"messages": private, "error": private}]}}
    path.write_text(json.dumps(result), encoding="utf-8")
    original = path.read_bytes()
    summary_path = write_summary(path)
    text = summary_path.read_text(encoding="utf-8")
    assert private not in text and "negative_candidates" not in text and '"cases"' not in text
    summary = json.loads(text)
    assert summary["tasks"]["judge"]["accuracy"] == 0.5
    assert summary["tasks"]["judge"]["nested"] == {"count": 3}
    assert summary["provenance"]["model_request_count"] == 1
    assert summary["provenance"]["git"] == {"commit": "abc123", "dirty": True}
    assert len(summary["full_result_sha256"]) == 64 and path.read_bytes() == original
    write_index(tmp_path)
    index = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert index.count(f"]({summary_path.name})") == 1 and f"]({path.name})" not in index
    # A fresh checkout only has the public summary; the same index still works.
    path.unlink()
    write_index(tmp_path)
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == index
