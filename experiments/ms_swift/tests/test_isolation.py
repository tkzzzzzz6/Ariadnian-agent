"""The training-preparation package must work without the application tree."""
import json
from pathlib import Path
import subprocess
import sys


def test_installed_experiments_work_outside_repo_without_backend_path(tmp_path):
    script = """
import importlib.metadata, json, sys
from pathlib import Path
import cpt.core
import ariadne_data.documents
assert 'longdoc.pipeline' not in sys.modules
assert 'fastapi' not in sys.modules
assert 'litellm' not in sys.modules
assert 'torch' not in sys.modules
assert not any(Path(p).name == 'backend' for p in sys.path if p)
print(json.dumps({'cpt': cpt.core.__file__, 'common': ariadne_data.documents.__file__,
                  'dependencies': importlib.metadata.requires('ariadne-ms-swift-experiments')}))
"""
    result = subprocess.run([sys.executable, "-I", "-c", script], cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    installed = json.loads(result.stdout)
    assert "backend" not in Path(installed["cpt"]).parts
    assert "backend" not in Path(installed["common"]).parts
    assert any("ariadne-data-core" in dep for dep in installed["dependencies"])
    assert not any(dep.startswith(("fastapi", "litellm", "torch", "ms-swift"))
                   for dep in installed["dependencies"])
