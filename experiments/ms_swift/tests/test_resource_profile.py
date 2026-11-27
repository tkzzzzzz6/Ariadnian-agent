import json
from unittest.mock import Mock

import pytest

from cpt import profile


def test_gpu_csv_missing_and_nonfinite_values_are_not_zero():
    rows = profile.parse_gpu_csv('0, GPU-a, "GPU, Test", 600.0, 8192, 3000, 5192, [N/A], NaN\n')
    assert rows[0]["name"] == "GPU, Test"
    assert rows[0]["memory.free"] == 5192
    assert rows[0]["utilization.gpu"] is None
    assert rows[0]["power.draw"] is None


def test_changed_driver_csv_fails_instead_of_mislabeling_counters():
    with pytest.raises(ValueError, match="column count"):
        profile.parse_gpu_csv("0, 1024, 42")


def test_missing_driver_is_preserved(monkeypatch):
    monkeypatch.setattr(profile.subprocess, "run", Mock(side_effect=FileNotFoundError()))
    sample = profile.gpu_sample()
    assert sample["error"] == "FileNotFoundError"
    assert profile.summarize_samples([sample])["failed_samples"] == 1


def test_sampled_board_peak_not_claimed_as_training_peak():
    first = profile.parse_gpu_csv("0,GPU-a,Test,600,8192,2000,6192,10,50")
    second = profile.parse_gpu_csv("0,GPU-a,Test,600,8192,5000,3192,99,120")
    report = profile.summarize_samples([{"gpus": first, "error": None}, {"gpus": second, "error": None}])
    assert report["boards"]["GPU-a"]["sampled_peak_board_used_mib"] == 5000
    assert report["boards"]["GPU-a"]["minimum_board_free_mib"] == 3192
    assert report["training_peak_allocated_bytes"] is None
    assert report["effective_loss_tokens_per_second"] is None


def test_preview_plan_never_launches_even_with_execute(tmp_path, monkeypatch):
    from cpt import launch
    plan = tmp_path / "plan.json"
    plan.write_text('{}', encoding='utf-8')
    monkeypatch.setattr(launch, "launch", Mock(return_value={"executed": False, "training_ready": False,
                                                            "blockers": ["preview_or_unapproved_data"]}))
    monkeypatch.setattr(profile, "inventory", lambda: {})
    child = Mock(side_effect=AssertionError("must not execute"))
    monkeypatch.setattr(profile.subprocess, "Popen", child)
    result = profile.profile_training(plan, tmp_path / "out", execute=True)
    assert result["status"] == "blocked_by_data_or_plan"
    assert result["executed"] is False
    assert not child.called
    assert json.loads((tmp_path / "out/resource_summary.json").read_text())["training_completed"] is False


def test_profile_does_not_overwrite_previous_evidence(tmp_path, monkeypatch):
    from cpt import launch
    plan = tmp_path / "plan.json"
    plan.write_text('{}', encoding='utf-8')
    monkeypatch.setattr(launch, "launch", Mock(return_value={"training_ready": False}))
    out = tmp_path / "out"
    out.mkdir()
    existing = out / "resource_summary.json"
    existing.write_text('old', encoding='utf-8')
    with pytest.raises(ValueError, match="new or empty"):
        profile.profile_training(plan, out)
    assert existing.read_text() == 'old'


@pytest.mark.parametrize('kwargs', [{"max_seconds": 0}, {"max_seconds": True}, {"interval": float('nan')}, {"interval": 0}])
def test_unbounded_or_invalid_monitoring_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        profile.profile_training(tmp_path / 'missing', tmp_path / 'out', **kwargs)


def test_timeout_kills_owned_children_after_wrapper_has_exited(monkeypatch):
    # Reproduce the wrapper-exited / trainer-still-running case without spawning
    # or signalling real processes on Windows CI.
    signals = []
    alive = [True]
    monkeypatch.setattr(profile.signal, 'SIGKILL', 9, raising=False)

    def killpg(pgid, signal_number):
        assert pgid == 123456
        signals.append(signal_number)
        if signal_number == 9:
            alive[0] = False
        elif signal_number == 0 and not alive[0]:
            raise ProcessLookupError()

    monkeypatch.setattr(profile.os, 'killpg', killpg, raising=False)
    proc = Mock(pid=123456)
    proc.poll.return_value = 0
    result = profile._stop_owned_process(proc, grace_seconds=0)
    assert result == {"forced": True, "group_alive_after_cleanup": False}
    assert int(profile.signal.SIGTERM) in signals and 9 in signals
    proc.wait.assert_called_once()
