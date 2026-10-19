"""Development-only synthetic-task diagnostics. No solver, API, or benchmark access."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np
import torch
from torch import nn


REPO = Path(__file__).resolve().parents[2]
CONTRACT = REPO / "docs" / "TASK_PLATFORM_PILOT.md"


@dataclass(frozen=True)
class Config:
    vocab: int = 256
    dim: int = 128
    layers: int = 2
    heads: int = 4
    ff_dim: int = 512
    max_positions: int = 64
    train_sequences: int = 64
    audit_sequences: int = 64
    prefix: int = 16
    continuation: int = 16
    short_history: int = 4
    teacher_seed: int = 101
    train_seed: int = 202
    audit_seed: int = 303
    smoothing: float = 1.0

    def validate(self):
        counts = (self.vocab, self.dim, self.layers, self.heads, self.ff_dim,
                  self.max_positions, self.train_sequences, self.audit_sequences,
                  self.prefix, self.continuation, self.short_history)
        if min(counts) <= 0 or self.vocab < 2 or self.dim % self.heads:
            raise ValueError("Invalid positive sizes or attention head divisibility")
        if self.prefix + self.continuation > self.max_positions:
            raise ValueError("Sequence exceeds position capacity")
        if self.train_seed == self.audit_seed or self.smoothing <= 0:
            raise ValueError("Separate train/audit seeds and positive smoothing required")


class FrozenTeacher(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        cfg.validate()
        # Independent layer draws (TransformerEncoder's cloned initial weights are avoided).
        with torch.random.fork_rng():
            torch.manual_seed(cfg.teacher_seed)
            self.tokens = nn.Embedding(cfg.vocab, cfg.dim)
            self.positions = nn.Embedding(cfg.max_positions, cfg.dim)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(cfg.dim, cfg.heads, cfg.ff_dim,
                                           dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
                for _ in range(cfg.layers)
            ])
            self.norm = nn.LayerNorm(cfg.dim)
            self.output = nn.Linear(cfg.dim, cfg.vocab)
        self.requires_grad_(False)
        self.eval()

    def forward(self, tokens, position_offset=0):
        length = tokens.shape[1]
        positions = torch.arange(position_offset, position_offset + length)
        hidden = self.tokens(tokens) + self.positions(positions)
        mask = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            hidden = layer(hidden, src_mask=mask)
        return self.output(self.norm(hidden))


@torch.inference_mode()
def generate(teacher, cfg, count, seed, *, audit=False):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sequence = torch.randint(cfg.vocab, (count, cfg.prefix), generator=generator)
    probabilities, short_probabilities = [], []
    for _ in range(cfg.continuation):
        probability = teacher(sequence)[:, -1].double().softmax(-1)
        if audit:
            probabilities.append(probability.numpy())
            start = max(0, sequence.shape[1] - cfg.short_history)
            short = teacher(sequence[:, start:], position_offset=start)[:, -1]
            short_probabilities.append(short.double().softmax(-1).numpy())
        target = torch.multinomial(probability, 1, generator=generator)
        sequence = torch.cat((sequence, target), dim=1)
    result = {"tokens": sequence.numpy()}
    if audit:
        result["teacher_probs"] = np.stack(probabilities, axis=1)
        result["short_probs"] = np.stack(short_probabilities, axis=1)
    return result


def continuation_pairs(tokens, cfg):
    """Only continuation targets count; the first uses the last prefix token."""
    return (tokens[:, cfg.prefix - 1:cfg.prefix + cfg.continuation - 1],
            tokens[:, cfg.prefix:cfg.prefix + cfg.continuation])


def fit_baselines(train_tokens, cfg):
    previous, targets = continuation_pairs(train_tokens, cfg)
    unigram = np.full(cfg.vocab, cfg.smoothing, dtype=np.float64)
    bigram = np.full((cfg.vocab, cfg.vocab), cfg.smoothing, dtype=np.float64)
    np.add.at(unigram, targets.ravel(), 1)
    np.add.at(bigram, (previous.ravel(), targets.ravel()), 1)
    return unigram / unigram.sum(), bigram / bigram.sum(axis=1, keepdims=True)


def distribution_metrics(teacher_probs, candidate_probs, targets):
    candidate_probs = np.broadcast_to(candidate_probs, teacher_probs.shape)
    log_p, log_q = np.log(teacher_probs), np.log(candidate_probs)
    entropy = -(teacher_probs * log_p).sum(-1).mean()
    expected_ce = -(teacher_probs * log_q).sum(-1).mean()
    forward_kl = (teacher_probs * (log_p - log_q)).sum(-1).mean()
    sampled_ce = -np.take_along_axis(log_q, targets[..., None], axis=-1).mean()
    return {"sampled_ce_nats": float(sampled_ce),
            "expected_ce_nats": float(expected_ce),
            "teacher_entropy_nats": float(entropy),
            "forward_kl_nats": float(forward_kl)}


def summarize(arrays, cfg):
    for name, count in (("train_tokens", cfg.train_sequences), ("audit_tokens", cfg.audit_sequences)):
        tokens = arrays[name]
        if (tokens.shape != (count, cfg.prefix + cfg.continuation)
                or not np.issubdtype(tokens.dtype, np.integer)
                or (tokens < 0).any() or (tokens >= cfg.vocab).any()):
            raise ValueError(f"Invalid {name} shape, dtype, or token range")
    previous, targets = continuation_pairs(arrays["audit_tokens"], cfg)
    train_uni, train_bi = fit_baselines(arrays["train_tokens"], cfg)
    np.testing.assert_allclose(arrays["unigram_probs"], train_uni, rtol=0, atol=0)
    np.testing.assert_allclose(arrays["bigram_probs"], train_bi, rtol=0, atol=0)
    teacher = arrays["teacher_probs"]
    expected_shape = (cfg.audit_sequences, cfg.continuation, cfg.vocab)
    for key in ("teacher_probs", "short_probs"):
        probs = arrays[key]
        if probs.shape != expected_shape or not np.isfinite(probs).all() or (probs <= 0).any():
            raise ValueError(f"Invalid {key}")
        np.testing.assert_allclose(probs.sum(-1), 1, rtol=0, atol=1e-12)
    choices = {"uniform": np.full(cfg.vocab, 1 / cfg.vocab),
               "unigram": train_uni, "bigram": train_bi[previous],
               "teacher": teacher, "short_history": arrays["short_probs"]}
    return {"status": "development-only", "teacher_count": 1, "vocab": cfg.vocab,
            "train_sequences": cfg.train_sequences, "audit_sequences": cfg.audit_sequences,
            "train_continuation_tokens": cfg.train_sequences * cfg.continuation,
            "audit_continuation_tokens": int(targets.size),
            "train_distinct_continuation_tokens": int(np.unique(continuation_pairs(arrays["train_tokens"], cfg)[1]).size),
            "audit_distinct_continuation_tokens": int(np.unique(targets).size),
            "uniform_ce_analytic_nats": float(np.log(cfg.vocab)),
            "metrics": {name: distribution_metrics(teacher, probs, targets)
                        for name, probs in choices.items()},
            "scope": "One frozen random teacher; separate seeded development samples. No solver or generalization benefit measured.",
            "short_history_scope": "Last k tokens with original absolute positions; out-of-distribution intervention, not causal proof of memory dependence."}


def output_path(run_name, data_root=None, repo=REPO):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", run_name):
        raise ValueError("Run name must be a simple alphanumeric filename")
    root_value = data_root if data_root is not None else os.environ.get("AIQ_KB_DATA_ROOT")
    if not root_value:
        raise ValueError("Set AIQ_KB_DATA_ROOT to an external data directory")
    root = Path(root_value).expanduser().resolve()
    repo = Path(repo).resolve()
    runs = (root / "runs").resolve()
    destination = (runs / run_name).resolve()
    if root == repo or repo in root.parents or runs == repo or repo in runs.parents:
        raise ValueError("AIQ_KB_DATA_ROOT and runs must be outside this repository")
    if destination.parent != runs or destination == repo or repo in destination.parents:
        raise ValueError("Run destination escapes the external runs directory")
    return destination


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def render_report(summary):
    rows = []
    for name, values in summary["metrics"].items():
        rows.append(f"| {name} | {values['sampled_ce_nats']:.4f} | {values['expected_ce_nats']:.4f} | {values['forward_kl_nats']:.4f} |")
    entropy = summary["metrics"]["teacher"]["teacher_entropy_nats"]
    return f"""# 开发任务质量诊断

## 结论
本次仅描述单个冻结随机 teacher 的开发分布，不判定可售性或 solver 收益。
数字证据：{summary['audit_sequences']} 条审计序列、{summary['audit_continuation_tokens']} 个续写 token；条件熵 {entropy:.4f} nats/token。

## Formulation
在 teacher 产生的相同历史上比较均匀、训练集拟合 unigram/bigram 与短历史 teacher。只计续写位置；expected CE − teacher entropy = forward KL。

## 成立程度
只有一个 teacher 和一组固定开发采样 seed，无跨 teacher、机制或 solver 泛化证据。均匀基线 log(V) 是解析事实，其余指标是测量。

## 方法
冻结随机初始化因果 Transformer；均匀独立前缀后条件采样。基线只拟合训练续写，审计样本使用独立 seed。所有数值单位均为 nats/token。

| 模型 | sampled CE | expected CE | forward KL |
|---|---:|---:|---:|
{chr(10).join(rows)}

## 失败
未定义商品通过阈值；低阶预测器接近 teacher 或短历史差异微小时同样保留结果。不据单次诊断宣布任务成功。程序失败会留下不完整目录，不能作为完成记录。

## 未决
训练集有 {summary['train_continuation_tokens']} 个续写标签，词表为 {summary['vocab']}，bigram 共有 {summary['vocab'] ** 2} 个转移计数槽；有限样本和加 1 平滑会影响估计，bigram 的差表现不能排除低阶结构。候选架构排名、重复训练方差、独立机制泛化、solver 收益与买方验收均未测。短历史删除上下文但保持位置，是离分布干预，不是纯粹记忆因果证据。

## 证据
config.json 记录配置、合同与代码 hash、Git commit 和版本；teacher.pt 保存冻结权重；arrays.npz 保存训练/审计 token、teacher 与短历史概率、基线参数；summary.json 保存完整精度。`--verify RUN_NAME` 只复算，不重采样。
"""


def run(cfg, destination, *, provenance=None):
    cfg.validate()
    destination = Path(destination)
    # Recheck the direct API as well as CLI; no silent overwrite or repository artifacts.
    checked = output_path(destination.name, destination.parent.parent)
    if checked != destination.resolve() or destination.parent.name != "runs":
        raise ValueError("Destination must be external DATA_ROOT/runs/RUN_NAME")
    destination.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    metadata = {"config": asdict(cfg), "status": "running", "scope": "development-only",
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "device": "cpu", "torch_threads": 1, "provenance": provenance or {},
                "versions": {"torch": torch.__version__, "numpy": np.__version__}}
    write_json(destination / "config.json", metadata)
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        teacher = FrozenTeacher(cfg)
        torch.save(teacher.state_dict(), destination / "teacher.pt")
        metadata["teacher_sha256"] = sha256(destination / "teacher.pt")
        write_json(destination / "config.json", metadata)
        train = generate(teacher, cfg, cfg.train_sequences, cfg.train_seed)
        audit = generate(teacher, cfg, cfg.audit_sequences, cfg.audit_seed, audit=True)
        unigram, bigram = fit_baselines(train["tokens"], cfg)
        arrays = {"train_tokens": train["tokens"], "audit_tokens": audit["tokens"],
                  "teacher_probs": audit["teacher_probs"], "short_probs": audit["short_probs"],
                  "unigram_probs": unigram, "bigram_probs": bigram}
        np.savez_compressed(destination / "arrays.npz", **arrays)
        summary = summarize(arrays, cfg)
        write_json(destination / "summary.json", summary)
        (destination / "report.md").write_text(render_report(summary))
        metadata.update(status="completed", arrays_sha256=sha256(destination / "arrays.npz"),
                        completed_utc=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=time.perf_counter() - started)
        write_json(destination / "config.json", metadata)
        return summary
    except BaseException as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {str(error)[:240]}",
                        failed_utc=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=time.perf_counter() - started)
        write_json(destination / "config.json", metadata)
        raise


def verify(destination):
    destination = Path(destination)
    metadata = json.loads((destination / "config.json").read_text())
    if metadata["status"] != "completed":
        raise ValueError("Run is not completed")
    if sha256(destination / "arrays.npz") != metadata["arrays_sha256"]:
        raise ValueError("Artifact hash mismatch")
    if sha256(destination / "teacher.pt") != metadata["teacher_sha256"]:
        raise ValueError("Teacher checkpoint hash mismatch")
    cfg = Config(**metadata["config"])
    cfg.validate()
    with np.load(destination / "arrays.npz", allow_pickle=False) as saved:
        summary = summarize(saved, cfg)
    expected = json.loads((destination / "summary.json").read_text())
    if summary != expected:
        raise ValueError("Recomputed summary differs from saved summary")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-name")
    mode.add_argument("--verify", metavar="RUN_NAME")
    args = parser.parse_args(argv)
    destination = output_path(args.run_name or args.verify)
    if args.verify:
        verify(destination)
        print(f"Verified saved artifacts: {destination}")
        return
    commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    for source in (Path(__file__).resolve(), CONTRACT):
        relative = source.relative_to(REPO).as_posix()
        try:
            committed = subprocess.check_output(["git", "-C", str(REPO), "show", f"{commit}:{relative}"], stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as error:
            raise ValueError(f"Commit the source and contract before running: {relative}") from error
        if source.read_bytes() != committed:
            raise ValueError(f"Source or contract differs from HEAD; commit preregistration first: {relative}")
    provenance = {"git_commit": commit, "code_sha256": sha256(__file__),
                  "contract_sha256": sha256(CONTRACT), "contract_path": str(CONTRACT.relative_to(REPO))}
    run(Config(), destination, provenance=provenance)
    print(f"Development diagnostics saved: {destination}")


if __name__ == "__main__":
    main()
