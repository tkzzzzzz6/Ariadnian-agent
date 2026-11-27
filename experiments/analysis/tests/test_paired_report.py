"""Mechanically validate rendered artifacts and absent/failed result semantics."""
import importlib.util
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


MODULE = Path(__file__).resolve().parents[1] / "plot_paired_pilot.py"
spec = importlib.util.spec_from_file_location("paired_report", MODULE)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.packet = self.root / "packet"
        self.packet.mkdir()
        self.assets = self.root / "assets"
        self.assets.mkdir()
        report.save(self.assets / "model.lock.json", {"fixture": True})
        self.summary = {"schema_version": "paired-review-candidates-v1", "counts": {"train": 1, "dev": 1},
            "directions": {"train": {"crossdoc": 1}, "dev": {"evidence_local_qa": 1}},
            "domains": {"programming": 1, "agriculture": 1}, "source_documents": 2,
            "connected_families": {"train": 1, "dev": 1}, "approved_pairs": 0, "training_ready": False}
        report.save(self.packet / "summary.json", self.summary)
        pairs = [{"id": "a", "split": "train", "direction": "crossdoc"},
                 {"id": "b", "split": "dev", "direction": "evidence_local_qa"}]
        records = [{"id": row["id"], "split": row["split"],
                    "messages": [{"role": "user", "content": "PRIVATE_QUESTION"},
                                 {"role": "assistant", "content": "PRIVATE_ANSWER"}]} for row in pairs]
        for name, rows in (("pairs.jsonl", pairs), ("short.jsonl", records), ("long.jsonl", records),
                           ("sources.jsonl", [{"domain": "programming"}, {"domain": "agriculture"}])):
            (self.packet / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
        lock = {"schema_version": "paired-data-lock-v1", "model_lock_sha256": report.sha256(self.assets / "model.lock.json"),
                "artifacts": {name: report.sha256(self.packet / name) for name in
                              ("pairs.jsonl", "short.jsonl", "long.jsonl", "sources.jsonl", "summary.json")}}
        report.save(self.packet / "data.lock.json", lock)
        self.native = {"pairs": [{"split": "train", "short": 500, "long": 4500, "supervised": 80},
                                  {"split": "dev", "short": 600, "long": 5500, "supervised": 90}]}

    def tearDown(self):
        self.temporary.cleanup()

    def collect(self, **kwargs):
        result = subprocess.CompletedProcess([], 0, stdout=json.dumps(self.native), stderr="")
        with patch.object(report.subprocess, "run", return_value=result):
            return report.collect(self.packet, tokenizer_python="explicit-test-python", model_assets=self.assets, **kwargs)

    def test_changed_packet_rejected_before_tokenization(self):
        (self.packet / "short.jsonl").write_text("modified")
        with patch.object(report.subprocess, "run") as process:
            with self.assertRaisesRegex(ValueError, "frozen packet artifact changed"):
                report.collect(self.packet, tokenizer_python="unused", model_assets=self.assets)
            process.assert_not_called()

    def test_missing_gpu_results_are_not_zero_accuracy(self):
        data = self.collect()
        self.assertTrue(all(row["status"] == "not_run" for row in data["resource"]))
        path = report.render_html(data, {}, self.root / "report.html", "OAuth invalid_grant；待人工审核")
        text = path.read_text(encoding="utf-8")
        self.assertIn("尚未执行", text)
        self.assertIn("尚无准确率结果", text)
        self.assertIn("invalid_grant", text)
        self.assertNotIn("PRIVATE_QUESTION", text)

    def test_best_exposure_is_separate_from_completed_training(self):
        data = self.collect()
        data['training']['short'] = {'status': 'process_exited', 'steps': [{}] * 8,
            'exposure': {'total_completed_steps': 8, 'best_step': 2, 'input_tokens': 999,
                         'supervised_tokens': 80, 'selected_input_tokens': 222, 'selected_supervised_tokens': 20}}
        path = report.render_html(data, {}, self.root / 'best.html', 'fixture only')
        text = path.read_text(encoding='utf-8')
        self.assertIn('best step', text)
        self.assertIn('选中模型输入 tokens', text)
        self.assertIn('<td>2</td><td>222</td>', text)
        self.assertNotIn("PRIVATE_ANSWER", text)
        self.assertNotIn(">0%</td>", text)

    def test_oom_status_and_failed_evaluation_denominator_preserved(self):
        evidence = self.root / "evidence"
        evidence.mkdir()
        report.save(evidence / "probe_train_4096_result.json", {"schema_version": "paired-resource-probe-v1",
                    "kind": "train", "input_length": 4096, "status": "oom", "steps": []})
        report.save(evidence / "probe_train_4096.json", {"status": "failed"})
        score_path = self.root / "eval.json"
        report.save(score_path, {"schema_version": "gemma-bounded-summary-v1", "kind": "chat",
            "status": "completed_with_failures", "total": 2, "successful": 1, "failed_or_skipped": 1,
            "statuses": {"ok": 1, "failed_runtime": 1}})
        data = self.collect(resource_evidence=evidence, evaluation_summaries={"base": score_path})
        self.assertEqual(data["resource"][0]["status"], "oom")
        self.assertEqual(data["evaluation"]["base"]["total"], 2)
        self.assertEqual(data["evaluation"]["base"]["failed_or_skipped"], 1)

    def test_mechanical_render_has_png_svg_and_self_contained_images(self):
        data = self.collect()
        figures = report.plots(data, self.root / "figures")
        self.assertEqual(set(figures), {"paired_lengths.png", "paired_lengths.svg", "task_source_mix.png", "task_source_mix.svg"})
        for name, path in figures.items():
            self.assertGreater(path.stat().st_size, 1000)
            if name.endswith(".png"):
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            else:
                self.assertIn("<svg", path.read_text(encoding="utf-8"))
        output = report.render_html(data, figures, self.root / "report.html", "awaiting review")
        text = output.read_text(encoding="utf-8")
        self.assertEqual(text.count('src="data:image/png;base64,'), 2)
        self.assertNotIn('src="http', text)
        self.assertNotIn("PRIVATE_ANSWER", text)

    def score_fixture(self):
        groups = {}
        for arm, value in (("Base", 20), ("short", 30), ("long", 40)):
            groups[arm] = [{"benchmark": "ruler", "task": "qa_1", "metric": "qa_f1_score",
                            "length_bucket": "<=8192", "records": 250, "clusters": 1,
                            "statuses": {"ok": 240, "failed_runtime": 10},
                            "score_percent": value, "all_ids_in_denominator": True}]
        return {"schema_version": "gemma-pilot-scoring-v1", "total_frozen_records": 250,
                "inputs": {"records": {"sha256": "a" * 64}}, "overall_score": None,
                "labels": list(groups), "groups": groups,
                "comparisons": [{"before": "short", "after": "long", "benchmark": "ruler",
                    "task": "qa_1", "metric": "qa_f1_score", "length_bucket": "<=8192", "records": 250,
                    "delta_percentage_points": 10, "ci95_percentage_points": None,
                    "ci_unavailable_reason": "fewer_than_two_source_clusters"}],
                "exploratory_composites": [], "limitations": ["test fixture only"]}

    def test_scores_require_exact_frozen_hash_and_full_failure_denominator(self):
        score = self.score_fixture()
        with self.assertRaisesRegex(ValueError, "frozen250"):
            report.validate_scores(score, "b" * 64, 250)
        changed = copy.deepcopy(score)
        changed["groups"]["short"][0]["records"] = 240
        with self.assertRaisesRegex(ValueError, "denominator"):
            report.validate_scores(changed, "a" * 64, 250)
        normalized = report.validate_scores(score, "a" * 64, 250)
        self.assertEqual(normalized["groups"]["base"][0]["records"], 250)
        self.assertIsNone(normalized["short_to_long"][0]["ci95_percentage_points"])

    def test_bound_score_facets_and_missing_ci_render_only_with_explicit_fixture(self):
        data = self.collect()
        data["scores"] = report.validate_scores(self.score_fixture(), "a" * 64, 250)
        figures = report.plots(data, self.root / "score_figures")
        self.assertIn("task_scores.png", figures)
        self.assertIn("short_long_deltas.svg", figures)
        html_path = report.render_html(data, figures, self.root / "scored.html", "fixture only")
        text = html_path.read_text(encoding="utf-8")
        self.assertIn("已有绑定评分汇总", text)
        self.assertIn("见分任务指标图", text)
        self.assertNotIn("尚无准确率结果", text)

    def test_scorer_all_lengths_rows_do_not_double_count_denominator(self):
        score = self.score_fixture()
        for rows in score['groups'].values():
            aggregate = copy.deepcopy(rows[0])
            aggregate['length_bucket'] = 'all_lengths'
            rows.append(aggregate)
        normalized = report.validate_scores(score, 'a' * 64, 250)
        self.assertEqual(len(normalized['groups']['short']), 2)
        data = self.collect()
        data['scores'] = normalized
        page = report.render_html(data, {}, self.root / 'lengths.html', 'fixture only')
        text = page.read_text(encoding='utf-8')
        self.assertIn('<td>250</td><td>240</td><td>10</td>', text)
        self.assertNotIn('<td>500</td>', text)
        changed = copy.deepcopy(score)
        changed['groups']['long'][1]['statuses'] = {'ok': 250}
        with self.assertRaisesRegex(ValueError, 'aggregate.*denominator'):
            report.validate_scores(changed, 'a' * 64, 250)


if __name__ == "__main__":
    unittest.main()
