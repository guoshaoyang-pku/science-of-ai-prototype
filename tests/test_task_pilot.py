from dataclasses import replace
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aiq_kb.task_pilot import (
    Config, FrozenTeacher, continuation_pairs, distribution_metrics,
    fit_baselines, generate, output_path, run, verify,
)


def tiny():
    return Config(vocab=7, dim=8, layers=1, heads=2, ff_dim=16,
                  max_positions=12, train_sequences=3, audit_sequences=2,
                  prefix=3, continuation=3, short_history=2)


def test_continuation_targets_and_causal_teacher():
    cfg = tiny()
    tokens = np.array([[1, 2, 3, 4, 5, 6]])
    previous, targets = continuation_pairs(tokens, cfg)
    np.testing.assert_array_equal(previous, [[3, 4, 5]])
    np.testing.assert_array_equal(targets, [[4, 5, 6]])
    teacher = FrozenTeacher(cfg)
    original = torch.tensor(tokens)
    changed = original.clone()
    changed[:, 4:] = 0
    with torch.inference_mode():
        np.testing.assert_allclose(teacher(original)[:, :4], teacher(changed)[:, :4], atol=1e-6)
    assert not teacher.training
    assert all(not p.requires_grad for p in teacher.parameters())


def test_generation_uses_context_before_target_and_preserves_short_positions():
    cfg = tiny()
    teacher = FrozenTeacher(cfg)
    data = generate(teacher, cfg, 2, cfg.audit_seed, audit=True)
    with torch.inference_mode():
        for step in range(cfg.continuation):
            context = torch.tensor(data["tokens"][:, :cfg.prefix + step])
            expected = teacher(context)[:, -1].double().softmax(-1).numpy()
            np.testing.assert_allclose(data["teacher_probs"][:, step], expected, rtol=0, atol=0)
            start = context.shape[1] - cfg.short_history
            short = teacher(context[:, start:], position_offset=start)[:, -1].double().softmax(-1)
            np.testing.assert_allclose(data["short_probs"][:, step], short, rtol=0, atol=0)


def test_expected_ce_minus_entropy_equals_forward_kl():
    teacher = np.array([[[0.1, 0.2, 0.7], [0.4, 0.4, 0.2]]])
    baseline = np.array([0.2, 0.3, 0.5])
    metrics = distribution_metrics(teacher, baseline, np.array([[2, 0]]))
    assert metrics["expected_ce_nats"] - metrics["teacher_entropy_nats"] == pytest.approx(metrics["forward_kl_nats"], abs=1e-14)
    assert metrics["sampled_ce_nats"] != pytest.approx(metrics["expected_ce_nats"])


def test_baselines_only_use_training_continuation():
    cfg = tiny()
    train = np.array([[0, 1, 2, 3, 4, 5]])
    first = fit_baselines(train, cfg)
    changed_prefix = train.copy()
    changed_prefix[:, :cfg.prefix - 1] = 6
    second = fit_baselines(changed_prefix, cfg)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    expected = np.ones(cfg.vocab)
    expected[[3, 4, 5]] += 1
    np.testing.assert_array_equal(first[0], expected / expected.sum())
    assert first[1][2, 3] == 2 / 8


def test_saved_artifact_recomputation_and_no_audit_leakage(tmp_path):
    cfg = tiny()
    first_path = tmp_path / "runs" / "first"
    first = run(cfg, first_path)
    assert verify(first_path) == first
    second_path = tmp_path / "runs" / "second"
    run(replace(cfg, audit_seed=999), second_path)
    with np.load(first_path / "arrays.npz") as a, np.load(second_path / "arrays.npz") as b:
        for key in ("train_tokens", "unigram_probs", "bigram_probs"):
            np.testing.assert_array_equal(a[key], b[key])
        assert not np.array_equal(a["audit_tokens"], b["audit_tokens"])
    assert first["audit_continuation_tokens"] == cfg.audit_sequences * cfg.continuation
    with pytest.raises(FileExistsError):
        run(cfg, first_path)
    metadata = json.loads((first_path / "summary.json").read_text())
    metadata["teacher_count"] = 2
    (first_path / "summary.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="differs"):
        verify(first_path)


def test_output_constraints(tmp_path, monkeypatch):
    repo = tmp_path / "repository"
    repo.mkdir()
    monkeypatch.delenv("AIQ_KB_DATA_ROOT", raising=False)
    with pytest.raises(ValueError, match="Set AIQ"):
        output_path("test")
    for name in ("../escape", "/absolute", "a/b", "..", ""):
        with pytest.raises(ValueError):
            output_path(name, tmp_path / "external", repo=repo)
    for root in (repo, repo / "artifacts"):
        with pytest.raises(ValueError, match="outside"):
            output_path("test", root, repo=repo)
    external = tmp_path / "external"
    external.mkdir()
    (external / "runs").symlink_to(repo, target_is_directory=True)
    with pytest.raises(ValueError, match="outside"):
        output_path("test", external, repo=repo)
    with pytest.raises(ValueError, match="external DATA_ROOT"):
        run(tiny(), tmp_path / "wrong" / "test")


def test_failure_preserves_started_metadata(tmp_path, monkeypatch):
    import aiq_kb.task_pilot as pilot

    destination = tmp_path / "runs" / "failure"

    def fail_generation(*args, **kwargs):
        metadata = json.loads((destination / "config.json").read_text())
        assert metadata["status"] == "running"
        assert metadata["started_utc"]
        assert metadata["device"] == "cpu"
        assert metadata["torch_threads"] == 1
        assert metadata["provenance"] == {"test": True}
        raise RuntimeError("deliberate test failure")

    monkeypatch.setattr(pilot, "generate", fail_generation)
    with pytest.raises(RuntimeError, match="deliberate"):
        run(tiny(), destination, provenance={"test": True})
    metadata = json.loads((destination / "config.json").read_text())
    assert metadata["status"] == "failed"
    assert metadata["elapsed_seconds"] >= 0
    assert "deliberate" in metadata["error"]
    with pytest.raises(ValueError, match="not completed"):
        verify(destination)


def test_rejects_malformed_token_artifacts():
    from aiq_kb.task_pilot import summarize

    cfg = tiny()
    valid = np.zeros((cfg.train_sequences, cfg.prefix + cfg.continuation), dtype=np.int64)
    for tokens in (valid[:, :-1], valid.astype(float), valid + cfg.vocab, valid - 1):
        with pytest.raises(ValueError, match="train_tokens"):
            summarize({"train_tokens": tokens}, cfg)


@pytest.mark.parametrize("uncommitted", ["source", "contract"])
@pytest.mark.parametrize("failure", ["untracked", "different"])
def test_cli_requires_committed_source_and_contract_before_creating_run(tmp_path, monkeypatch, uncommitted, failure):
    import subprocess
    import aiq_kb.task_pilot as pilot

    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "task_pilot.py"
    contract = repo / "contract.md"
    source.write_bytes(b"source bytes")
    contract.write_bytes(b"contract bytes")
    monkeypatch.setattr(pilot, "REPO", repo)
    monkeypatch.setattr(pilot, "CONTRACT", contract)
    monkeypatch.setattr(pilot, "__file__", str(source))
    monkeypatch.setenv("AIQ_KB_DATA_ROOT", str(tmp_path / "data"))
    target = source if uncommitted == "source" else contract

    def fake_git(command, **kwargs):
        if "rev-parse" in command:
            return "deadbeef\n"
        path = repo / command[-1].split(":", 1)[1]
        if path == target:
            if failure == "untracked":
                raise subprocess.CalledProcessError(128, command)
            return b"committed bytes differ"
        return path.read_bytes()

    monkeypatch.setattr(pilot.subprocess, "check_output", fake_git)
    with pytest.raises(ValueError, match="[Cc]ommit"):
        pilot.main(["--run-name", "blocked"])
    assert not (tmp_path / "data" / "runs" / "blocked").exists()
