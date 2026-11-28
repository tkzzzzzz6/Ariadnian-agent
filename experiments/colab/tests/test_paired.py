"""CPU checks of audit semantics, transport isolation and bounded failure evidence."""
import importlib.util
import itertools
import json
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile


HERE = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


bundle_base = load("bundle")
remote = load("remote_job")
audit = load("training_audit")
bundle = load("paired_bundle")
job = load("paired_job")


class PairedTests(unittest.TestCase):
    def test_evaluation_cache_uses_hub_below_hf_home(self):
        root = Path('/content/ariadne_paired_cache_fixture')
        with patch.object(job, 'command') as command:
            job.evaluate(root, 'base')
        argv = command.call_args_list[0].args[2]
        self.assertEqual(Path(argv[argv.index('--cache-dir') + 1]), root / 'hf_cache/hub')

    def test_whole_suite_has_bounded_adapter_decode_budget(self):
        root = Path('/content/ariadne_paired_budget_fixture')
        with patch.object(job, 'command') as command:
            job.evaluate(root, 'base')
        launch = command.call_args_list[1]
        argv = launch.args[2]
        self.assertEqual(argv[argv.index('--max-seconds') + 1], '5400')
        self.assertEqual(launch.kwargs['timeout'], 5460)

    def test_terminated_group_leader_does_not_skip_descendant_kill(self):
        proc = unittest.mock.Mock(pid=12345)
        proc.wait.return_value = 0
        with patch.object(job.os, 'killpg', create=True) as kill, patch.object(job.signal, 'SIGKILL', 9, create=True):
            job.terminate_group(proc)
        self.assertEqual([c.args[1] for c in kill.call_args_list], [job.signal.SIGTERM, 9])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def paired_rows(self):
        def row(identifier, context):
            return {"id": identifier, "split": "train", "source_document_ids": ["source-a", "source-b"],
                    "messages": [{"role": "user", "content": "Read the evidence.\n\n<source_excerpts>\n"
                                  + context + "\n</source_excerpts>\n\nQuestion: Which sample remained stable?"},
                                 {"role": "assistant", "content": "Sample A."}]}
        return {"short": [row("pair-1", "Short complete evidence."), row("pair-2", "Other short evidence.")],
                "long": [row("pair-1", "Long source context including complete evidence."),
                         row("pair-2", "Other long source context including the same evidence.")]}

    def test_paired_contract_allows_source_length_difference_without_rewriting(self):
        records = self.paired_rows()
        before = deepcopy(records)
        job.paired_record_contract(records)
        self.assertEqual(records, before)
        self.assertNotEqual(records["short"][0]["messages"][0], records["long"][0]["messages"][0])

    def test_same_answers_with_different_questions_are_not_a_valid_pair(self):
        records = self.paired_rows()
        records["long"][0]["messages"][0]["content"] = records["long"][0]["messages"][0]["content"].replace(
            "Which sample remained stable?", "Which sample failed the control test?")
        self.assertEqual(records["short"][0]["messages"][-1], records["long"][0]["messages"][-1])
        with self.assertRaisesRegex(ValueError, "same question"):
            job.paired_record_contract(records)

    def test_pair_identity_cannot_be_reordered_duplicated_or_blank(self):
        for kind in ("reordered", "duplicate", "empty", "missing"):
            with self.subTest(kind=kind):
                records = self.paired_rows()
                if kind == "reordered":
                    records["long"].reverse()
                elif kind == "duplicate":
                    records["long"][1]["id"] = records["long"][0]["id"]
                elif kind == "empty":
                    records["long"][0]["id"] = " "
                else:
                    del records["long"][0]["id"]
                with self.assertRaisesRegex(ValueError, "IDs"):
                    job.paired_record_contract(records)

    def test_pair_source_order_split_and_answer_must_match(self):
        for field in ("source_document_ids", "split", "answer"):
            with self.subTest(field=field):
                records = self.paired_rows()
                row = records["long"][0]
                if field == "source_document_ids":
                    row[field].reverse()
                elif field == "split":
                    row[field] = "dev"
                else:
                    row["messages"][-1]["content"] += " Additional detail."
                with self.assertRaisesRegex(ValueError, "ordered sources and exact answers"):
                    job.paired_record_contract(records)

    def test_pair_rejects_system_roles_and_missing_or_empty_question_suffix(self):
        for kind in ("system", "wrong_role", "missing_question", "empty_question", "second_question_marker"):
            with self.subTest(kind=kind):
                records = self.paired_rows()
                messages = records["long"][0]["messages"]
                if kind == "system":
                    messages.insert(0, {"role": "system", "content": "Additional instruction."})
                elif kind == "wrong_role":
                    messages[0]["role"] = "system"
                elif kind == "missing_question":
                    messages[0]["content"] = "Question: Which sample remained stable?"
                elif kind == "empty_question":
                    messages[0]["content"] = "context\n</source_excerpts>\n\nQuestion: "
                else:
                    messages[0]["content"] += "\n\nQuestion: Which sample remained stable?"
                with self.assertRaises(ValueError):
                    job.paired_record_contract(records)

    def test_audit_counts_actual_causal_labels_not_padded_or_prompt_tokens(self):
        observed = audit.describe_batch([[2, 9, 10, 11, 0], [2, 8, 12, 0, 0]],
                                       [[-100, -100, 10, 11, -100], [2, -100, 12, -100, -100]],
                                       [[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])
        self.assertEqual(observed["input_tokens"], 7)
        self.assertEqual(observed["supervised_tokens"], 3)
        self.assertEqual(observed["causal_supervised_token_ids_sha256"], audit.token_hash([[10, 11], [12]]))
        with self.assertRaisesRegex(ValueError, "padding"):
            audit.describe_batch([[2, 0]], [[-100, 0]], [[1, 0]])

    def test_equal_answers_keep_label_hash_under_different_prompt_lengths(self):
        short = audit.describe_batch([[2, 5, 9, 10]], [[-100, -100, 9, 10]])
        long = audit.describe_batch([[2, 5, 6, 7, 9, 10]], [[-100, -100, -100, -100, 9, 10]])
        self.assertNotEqual(short["input_token_ids_sha256"], long["input_token_ids_sha256"])
        self.assertEqual(short["causal_supervised_token_ids_sha256"], long["causal_supervised_token_ids_sha256"])

    def test_summary_rejects_missing_step_or_skipped_optimizer_update(self):
        row = {"optimizer_step": 1, "sequences": 8, "optimizer_step_skipped": True}
        (self.root / "optimizer_steps.jsonl").write_text(json.dumps(row) + "\n")
        with self.assertRaisesRegex(ValueError, "steps differ"):
            audit.summarize(self.root, expected_steps=2, expected_sequences=16)
        with self.assertRaisesRegex(ValueError, "skipped"):
            audit.summarize(self.root, expected_steps=1, expected_sequences=8)

    def test_actual_pair_count_controls_two_epoch_budget(self):
        small = job.paired_recipe(32)["parameters"]
        self.assertEqual(small["max_steps"], 8)
        self.assertEqual(small["num_train_epochs"], 2)
        self.assertEqual(small["gradient_accumulation_steps"], 8)
        self.assertEqual(job.paired_recipe(128)["parameters"]["max_steps"], 32)
        self.assertEqual(job.paired_recipe(33)["parameters"]["max_steps"], 10)
        with self.assertRaises(ValueError):
            job.paired_recipe(0)

    def test_summary_records_measured_exposure_not_declared_length(self):
        row = {"optimizer_step": 1, "sequences": 2, "optimizer_step_skipped": False,
               "input_tokens": 4567, "supervised_tokens": 37,
               "ordered_microbatch_label_hashes": ["a", "b"], "peak_allocated_bytes": 123,
               "peak_reserved_bytes": 456, "wall_seconds": 7.25}
        (self.root / "optimizer_steps.jsonl").write_text(json.dumps(row) + "\n")
        result = audit.summarize(self.root, expected_steps=1, expected_sequences=2)
        self.assertEqual(result["input_tokens"], 4567)
        self.assertEqual(result["supervised_tokens"], 37)
        self.assertEqual(result["observed_optimizer_wall_seconds"], 7.25)

    def test_resource_bundle_excludes_unsupplied_corpora_and_secrets(self):
        names = ["experiments/common/pyproject.toml", "experiments/common/README.md",
                 "experiments/ms_swift/pyproject.toml", "experiments/ms_swift/requirements-train.txt",
                 "experiments/ms_swift/gemma_eval/model_weights.lock.json",
                 "experiments/colab/remote_job.py", "experiments/colab/paired_job.py",
                 "experiments/colab/training_audit.py", "experiments/colab/best_checkpoint.py",
                 "experiments/colab/tmux_runner.py",
                 "experiments/colab/evaluation_outcome.py",
                 "experiments/common/ariadne_data/__init__.py",
                 "experiments/ms_swift/cpt/__init__.py", "experiments/ms_swift/gemma_eval/__init__.py",
                 "backend/.env", "experiments/ms_swift/.cache/reviews.jsonl"]
        names += ["experiments/ms_swift/.cache/model_assets/gemma4-e2b/" + name for name in bundle_base.ASSET_NAMES]
        for name in names:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
        output = self.root / "resource.zip"
        result = bundle.build_bundle(self.root, output, resource_only=True)
        self.assertFalse(result["records_approved"])
        with zipfile.ZipFile(output) as archive:
            self.assertFalse(any(name.startswith("inputs/") or "backend" in name for name in archive.namelist()))
        destination = self.root / "unpacked"
        remote.unpack(output, result["sha256"], destination)
        job.verify_bundle(destination)
        (destination / "experiments/colab/paired_job.py").write_text("changed")
        with self.assertRaisesRegex(ValueError, "transport input changed"):
            job.verify_bundle(destination)

    def test_failed_process_writes_receipt_and_preserves_raw_log(self):
        with self.assertRaisesRegex(RuntimeError, "exited 3"):
            job.command(self.root, "deliberate_failure", [sys.executable, "-c", "print('retained'); raise SystemExit(3)"], timeout=10)
        receipt = json.loads((self.root / "evidence/deliberate_failure.json").read_text())
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["returncode"], 3)
        self.assertIn("retained", (self.root / "evidence/deliberate_failure.log").read_text())

    def test_timeout_only_signals_owned_process_group_and_keeps_evidence(self):
        class Process:
            pid = 13579
            returncode = None
            def poll(self): return self.returncode
            def wait(self, timeout): return self.returncode
        proc = Process()
        def kill(pid, sig):
            self.assertEqual(pid, 13579)
            proc.returncode = -15
        with patch.object(job.subprocess, "Popen", return_value=proc), \
             patch.object(job.os, "killpg", side_effect=kill, create=True) as killer, \
             patch.object(job.signal, "SIGKILL", 9, create=True), \
             patch.object(job.time, "monotonic", side_effect=itertools.chain([0], itertools.repeat(2))):
            with self.assertRaises(subprocess.TimeoutExpired):
                job.command(self.root, "timeout", ["owned-worker"], timeout=1)
        self.assertEqual(killer.call_count, 2)
        receipt = json.loads((self.root / "evidence/timeout.json").read_text())
        self.assertEqual(receipt["status"], "timeout")
        self.assertEqual(receipt["pid"], 13579)


if __name__ == "__main__":
    unittest.main()
