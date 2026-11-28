"""CPU tests for transfer boundaries and traversal/integrity rejection."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile


HERE = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


bundle = module("bundle")
remote = module("remote_job")


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def zip(self, name="inputs/records.jsonl", *, content=b"{}\n", declared_hash=None, duplicate=False):
        archive = self.root / "bundle.zip"
        manifest = {"schema_version": "colab-sft-bundle-v1", "files": {
            name: {"bytes": len(content), "sha256": declared_hash or hashlib.sha256(content).hexdigest()}}}
        with zipfile.ZipFile(archive, "w") as stream:
            stream.writestr(name, content)
            if duplicate:
                stream.writestr(name, content)
            stream.writestr("bundle_manifest.json", json.dumps(manifest))
        return archive

    def test_unpack_intact_bundle(self):
        archive = self.zip()
        destination = self.root / "new"
        remote.unpack(archive, remote.sha256(archive), destination)
        self.assertEqual((destination / "inputs/records.jsonl").read_bytes(), b"{}\n")

    def test_outer_hash_rejected_before_creating_destination(self):
        archive = self.zip()
        destination = self.root / "new"
        with self.assertRaisesRegex(ValueError, "SHA256"):
            remote.unpack(archive, "0" * 64, destination)
        self.assertFalse(destination.exists())

    def test_inner_hash_rejected_before_writing(self):
        archive = self.zip(declared_hash="0" * 64)
        destination = self.root / "new"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            remote.unpack(archive, remote.sha256(archive), destination)
        self.assertFalse(destination.exists())

    def test_unsafe_paths_and_aliases(self):
        for name in ("../escape", "/absolute", "x/../../escape", "x\\escape", "C:/escape", "x//alias"):
            with self.subTest(name=name):
                archive = self.zip(name)
                destination = self.root / "new"
                # Windows zipfile normalizes backslashes before writing; then
                # the manifest mismatch rejects it even before the path check.
                with self.assertRaisesRegex(ValueError, "unsafe|unlisted"):
                    remote.unpack(archive, remote.sha256(archive), destination)
                self.assertFalse(destination.exists())

    def test_refuses_overwriting_existing_run(self):
        archive = self.zip()
        with self.assertRaisesRegex(ValueError, "already exists"):
            remote.unpack(archive, remote.sha256(archive), self.root)

    def test_slug_has_no_path_injection(self):
        for slug in ("../root", "x/y", "", "a" * 65, "x;rm", "汉字"):
            with self.subTest(slug=slug), self.assertRaises(ValueError):
                remote.run_root(slug)

    def test_bundle_does_not_walk_unrelated_data(self):
        files = ["experiments/common/pyproject.toml", "experiments/common/README.md",
                 "experiments/ms_swift/pyproject.toml", "experiments/ms_swift/requirements-train.txt",
                 "experiments/ms_swift/gemma_eval/model_weights.lock.json",
                 "experiments/ms_swift/configs/gemma4_e2b_sft_lora_4k_smoke.json",
                 "experiments/colab/remote_job.py", "experiments/common/ariadne_data/__init__.py",
                 "experiments/ms_swift/cpt/__init__.py", "experiments/ms_swift/gemma_eval/__init__.py",
                 "backend/.env", "experiments/ms_swift/.cache/hf_hub/model.safetensors",
                 "experiments/ms_swift/.cache/plans/old.json", "records.jsonl", "sources.jsonl"]
        files += ["experiments/ms_swift/.cache/model_assets/gemma4-e2b/" + name for name in bundle.ASSET_NAMES]
        for name in files:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
        output = self.root / "transport.zip"
        bundle.build_bundle(self.root, output, records=self.root / "records.jsonl", sources=self.root / "sources.jsonl")
        with zipfile.ZipFile(output) as archive:
            names = archive.namelist()
        self.assertIn("inputs/records.jsonl", names)
        self.assertNotIn("backend/.env", names)
        self.assertFalse(any("hf_hub" in name or "/plans/" in name or "safetensors" in name for name in names))
        self.assertNotIn("inputs/reviews.jsonl", names)


if __name__ == "__main__":
    unittest.main()
