#!/usr/bin/env python3
"""Freeze a reproducible, group-isolated sample of the four normalized KB sources."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys

SOURCES = ("arch170", "ranking_v2", "ranking_v3", "dataflip500")
DEFAULT_COUNTS = dict(zip(SOURCES, (80, 400, 300, 200)))
SPLITS = ("train", "val", "test")
IDENTITY_FIELDS = ("group", "dataset_id", "dataset_path", "dataset", "candidate_set",
                   "pair_id", "flip_pair_id", "dataset_seed_identity")
RANK_CREDIT = {0: 1.0, 1: 0.75, 2: 0.5, 3: 0.25}
PAYLOAD_FILES = ("questions.jsonl", "split.json", "selection.json", "README.md")


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def rows(path):
    try:
        raw = path.read_bytes()
        result = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    except (OSError, ValueError, UnicodeError) as error:
        raise ValueError(f"cannot read JSONL {path}: {error}") from error
    if any(not isinstance(row, dict) for row in result):
        raise ValueError(f"{path}: expected one normalized question object per line")
    return result, digest(raw)


def validate_question(question):
    qid = question.get("question_id")
    if not isinstance(qid, str) or not qid.strip():
        raise ValueError("question_id must be a nonempty string")
    if question.get("source") not in SOURCES:
        raise ValueError(f"{qid}: unknown source")
    if not isinstance(question.get("group"), (str, int)) or str(question["group"]).strip() == "":
        raise ValueError(f"{qid}: a nonempty group is required")
    n = question.get("num_choices")
    if type(n) is not int or not 2 <= n <= 5:
        raise ValueError(f"{qid}: num_choices must be an integer from 2 to 5")
    letters = list("ABCDE"[:n])
    answer = question.get("answer")
    if not isinstance(answer, str):
        raise ValueError(f"{qid}: answer must be a string")
    if question.get("task") == "select":
        valid = answer in letters
    elif question.get("task") == "ranking":
        valid = bool(re.fullmatch(r"[A-E](?:<[A-E])+", answer)) and sorted(answer.split("<")) == letters
    else:
        raise ValueError(f"{qid}: task must be select or ranking")
    if not valid:
        raise ValueError(f"{qid}: invalid gold answer {answer!r}")
    messages = question.get("messages")
    if not isinstance(messages, list) or not messages or any(
        not isinstance(m, dict) or not isinstance(m.get("role"), str)
        or not isinstance(m.get("content"), str) for m in messages
    ):
        raise ValueError(f"{qid}: original messages are required")
    if not isinstance(question.get("meta", {}), dict):
        raise ValueError(f"{qid}: meta must be an object")


def identity_tags(record):
    tags = set()
    if record.get("question_id"):
        tags.add("question:" + str(record["question_id"]))
    for obj in (record, record.get("meta", {})):
        if not isinstance(obj, dict):
            continue
        for field in IDENTITY_FIELDS:
            value = obj.get(field)
            if not isinstance(value, (str, int)) or isinstance(value, bool):
                continue
            value = str(value).strip().replace("\\", "/")
            if not value:
                continue
            if "datasets/" in value:
                value = value.split("datasets/", 1)[1]
            value = re.sub(r"/+", "/", value).strip("/")
            if field == "candidate_set":
                tags.add("candidate:" + value)
                if "/candidates/" in value:
                    tags.add("identity:" + value.split("/candidates/", 1)[0])
            else:
                tags.add("identity:" + value)
    return tags


class Groups:
    def __init__(self):
        self.parent = {}

    def find(self, tag):
        self.parent.setdefault(tag, tag)
        while self.parent[tag] != tag:
            self.parent[tag] = self.parent[self.parent[tag]]
            tag = self.parent[tag]
        return tag

    def join(self, tags):
        tags = sorted(tags)
        if not tags:
            return
        first = self.find(tags[0])
        for tag in tags[1:]:
            other = self.find(tag)
            if first != other:
                first, other = sorted((first, other))
                self.parent[other] = first


def seen_records(paths):
    records, provenance = [], []
    for path in paths:
        files = sorted(path.glob("epochs/e*/records.jsonl")) if path.is_dir() else [path]
        if not files:
            raise ValueError(f"{path}: no epoch records found")
        for file in files:
            try:
                raw = file.read_bytes()
                text = raw.decode("utf-8").strip()
                if not text:
                    entries = []
                elif text.startswith("["):
                    entries = json.loads(text)
                    if not isinstance(entries, list):
                        raise ValueError("expected a list")
                else:
                    entries = [json.loads(line) if line.lstrip().startswith(("{", '"')) else line.strip()
                               for line in text.splitlines() if line.strip()]
                for entry in entries:
                    if isinstance(entry, str):
                        records.append({"question_id": entry})
                    elif isinstance(entry, dict) and identity_tags(entry):
                        records.append(entry)
                    else:
                        raise ValueError("expected question IDs or records with explicit identities")
            except (OSError, ValueError, UnicodeError) as error:
                raise ValueError(f"cannot read seen records {file}: {error}") from error
            provenance.append({"path": str(file.resolve()), "sha256": digest(raw), "records": len(entries)})
    return records, provenance


def build_groups(questions, seen):
    union = Groups()
    for question in questions:
        union.join(identity_tags(question))
    seen_tags = set()
    for record in seen:
        tags = identity_tags(record)
        union.join(tags)
        seen_tags.update(tags)
    components = defaultdict(list)
    for question in questions:
        components[union.find("question:" + question["question_id"])].append(question)
    all_tags = defaultdict(list)
    for tag in sorted(union.parent):
        all_tags[union.find(tag)].append(tag)
    seen_roots = {union.find(tag) for tag in seen_tags}
    groups = {}
    for root, members in components.items():
        tags = all_tags[root]
        gid = "G" + digest("\n".join(tags).encode())[:24]
        groups[gid] = {"questions": members, "identifiers": tags,
                       "source_counts": dict(Counter(q["source"] for q in members)),
                       "seen_before": root in seen_roots}
    return groups


def shuffled(items, seed, label):
    result = sorted(items)
    random.Random(digest(f"{seed}:{label}".encode())).shuffle(result)
    return result


def select_groups(groups, counts, seed, val_fraction, test_fraction):
    strata = defaultdict(list)
    for gid, group in groups.items():
        group["split"] = "train"
        if not group["seen_before"]:
            strata[tuple(sorted(group["source_counts"]))].append(gid)
    for signature, gids in sorted(strata.items()):
        gids = shuffled(gids, seed, "split:" + "+".join(signature))
        n_val, n_test = int(len(gids) * val_fraction), int(len(gids) * test_fraction)
        for gid in gids[:n_val]:
            groups[gid]["split"] = "val"
        for gid in gids[n_val:n_val + n_test]:
            groups[gid]["split"] = "test"
    requested = {}
    for source, n in counts.items():
        n_val, n_test = int(n * val_fraction), int(n * test_fraction)
        requested[source] = dict(zip(SPLITS, (n - n_val - n_test, n_val, n_test)))
    selected, actual = set(), {source: Counter() for source in SOURCES}
    for split in SPLITS:
        for source in SOURCES:
            eligible = [gid for gid, group in groups.items()
                        if group["split"] == split and source in group["source_counts"]]
            for gid in shuffled(eligible, seed, f"sample:{split}:{source}"):
                if actual[source][split] >= requested[source][split]:
                    break
                if gid in selected:
                    continue
                selected.add(gid)
                for member_source, n in groups[gid]["source_counts"].items():
                    actual[member_source][split] += n
    # Preserve the requested source totals when prior exposure leaves too few fresh holdout groups.
    for source in SOURCES:
        eligible = [gid for gid, group in groups.items()
                    if group["split"] == "train" and source in group["source_counts"]]
        for gid in shuffled(eligible, seed, f"supplement:train:{source}"):
            if sum(actual[source].values()) >= counts[source]:
                break
            if gid in selected:
                continue
            selected.add(gid)
            for member_source, n in groups[gid]["source_counts"].items():
                actual[member_source]["train"] += n
    return selected, requested, actual


def build(source_paths, counts, exclusions, output, seed=20261004, val_fraction=0.15,
          test_fraction=0.15, name="kb_science_pool_v2", provenance=None):
    if set(source_paths) != set(SOURCES):
        raise ValueError(f"all four --source inputs required: {', '.join(SOURCES)}")
    if set(counts) != set(SOURCES) or any(type(n) is not int or n < 0 for n in counts.values()):
        raise ValueError("each sample count must be a nonnegative integer")
    if (not all(math.isfinite(x) and 0 <= x < 1 for x in (val_fraction, test_fraction))
            or val_fraction + test_fraction >= 1):
        raise ValueError("val/test fractions must be finite, nonnegative and sum to less than one")
    if output.exists():
        raise ValueError(f"refusing to overwrite immutable release {output}")
    questions, inputs, cache = {}, {}, {}
    for source in SOURCES:
        path = source_paths[source].resolve()
        if path not in cache:
            cache[path] = rows(path)
        all_rows, sha256 = cache[path]
        chosen = [q for q in all_rows if q.get("source") == source]
        if not chosen:
            raise ValueError(f"{path}: no questions for source {source}")
        duplicates = 0
        for question in chosen:
            validate_question(question)
            qid = question["question_id"]
            if qid in questions:
                if questions[qid] != question:
                    raise ValueError(f"conflicting duplicate question_id {qid}")
                duplicates += 1
            else:
                questions[qid] = question
        unique = [q for q in questions.values() if q["source"] == source]
        gates = Counter(json.dumps(q.get("meta", {}).get("gate", "unspecified"), ensure_ascii=False, sort_keys=True) for q in unique)
        inputs[source] = {"path": str(path), "sha256": sha256, "input_questions": len(unique),
                          "identical_duplicates_removed": duplicates, "gate_counts": dict(sorted(gates.items()))}
    seen, seen_inputs = seen_records(exclusions)
    groups = build_groups(list(questions.values()), seen)
    selected, requested, actual = select_groups(groups, counts, seed, val_fraction, test_fraction)
    chosen = sorted((q for gid in selected for q in groups[gid]["questions"]),
                    key=lambda q: (SOURCES.index(q["source"]), q["question_id"]))
    if not chosen:
        raise ValueError("sampling selected no questions")
    split_doc = {split: [] for split in SPLITS}
    split_doc.update(group_by_question={}, groups={})
    for gid in sorted(selected):
        group = groups[gid]
        qids = sorted(q["question_id"] for q in group["questions"])
        split_doc["groups"][gid] = {k: v for k, v in group.items() if k != "questions"}
        split_doc["groups"][gid]["question_ids"] = qids
        split_doc[group["split"]].extend(qids)
        split_doc["group_by_question"].update({qid: gid for qid in qids})
    for split in SPLITS:
        split_doc[split].sort()
    selection = {"seed": seed, "requested_by_split": requested,
                 "actual_by_split": {source: {split: actual[source][split] for split in SPLITS} for source in SOURCES},
                 "holdout_shortfalls": {source: {split: max(0, requested[source][split] - actual[source][split])
                                                 for split in ("val", "test")} for source in SOURCES},
                 "selected_group_ids": sorted(selected),
                 "selected_question_ids": sorted(q["question_id"] for q in chosen)}
    for source in SOURCES:
        n = sum(actual[source].values())
        inputs[source].update(input_groups=sum(source in g["source_counts"] for g in groups.values()),
                              requested_questions=counts[source], selected_questions=n,
                              split_counts=selection["actual_by_split"][source],
                              difference_from_requested=n - counts[source])
    table = "\n".join(f"| `{s}` | {inputs[s]['input_questions']} | {counts[s]} | {inputs[s]['selected_questions']} | "
                       + " / ".join(str(actual[s][p]) for p in SPLITS) + " |" for s in SOURCES)
    readme = f"""# {name}

四类题池的固定随机抽样；seed={seed}。下表数量来自输入文件和本次选择。

| 来源 | 输入题数 | 目标题数 | 实际题数 | train / val / test |
|---|---:|---:|---:|---|
{table}

arch170是v1.5显著性筛选的三选一架构题；ranking_v2是五选排序的legacy题池，涵盖架构、优化器与混合比较；ranking_v3是v1.5 full-separation筛选的五选架构排序题；dataflip500是共享候选设置、改变数据后赢家翻转的三选一Hard500题，成对保留。各来源的筛选条件保留在meta.gate，不能视为同一个统计保证。

按 group、显式数据集标识、candidate_set及已提供的数据生成种子标识联合整组划分；跨来源的共享标识合并。已见组只进入 train。未见组按来源组合随机划分，val/test 比例为 {val_fraction:g}/{test_fraction:g}（按组数向下取整）。每个 split 再按来源抽样，完整保留所选组，故实际数量可能超过目标。新holdout不足时把数量补回train，不把旧曝光组放入holdout；requested_by_split、actual_by_split和holdout_shortfalls登记在selection.json。仅能核验输入中提供的标识，未提供的共享数据关系无法判断。

选择题完全正确得1分。排序题按逆序对数0/1/2/3给1/0.75/0.5/0.25分，≥4或无法解析给0分。引用的KB claim记 `s += score, f += 1-score`；此记账不是规律真实的概率。

输入路径、SHA256、实际题数、gate及旧实验排除来源见 `manifest.json`；标签与原始prompt保留在 `questions.jsonl`，划分见 `split.json`，选样见 `selection.json`。ranking_v2的legacy gate与ranking_v3的显著性gate保持原记录，不合并为同一保证。重新构建必须写入新目录。
"""
    payloads = {"questions.jsonl": b"".join((json.dumps(q, ensure_ascii=False, sort_keys=True) + "\n").encode() for q in chosen),
                "split.json": json_bytes(split_doc), "selection.json": json_bytes(selection),
                "README.md": readme.encode()}
    if provenance is not None:
        source_provenance = json.loads(Path(provenance).read_text())
        if not isinstance(source_provenance, dict):
            raise ValueError("source provenance must be an object")
        payloads["source_provenance.json"] = json_bytes(source_provenance)
        payloads["README.md"] += ("\n原始输入与历史曝光清单见source_provenance.json。新holdout只排除清单中可追踪的旧曝光；旧seed无法恢复的来源仍是未决边界，不能据此宣称完全无污染。\n").encode()
    manifest = {"schema_version": 1, "release": name, "seed": seed, "sources": inputs,
                "split_fractions": {"val": val_fraction, "test": test_fraction},
                "grouping_fields": list(IDENTITY_FIELDS), "seen_inputs": seen_inputs,
                "seen_records": len(seen), "all_input_groups": len(groups), "selected_groups": len(selected),
                "ranking_credit": {str(k): v for k, v in RANK_CREDIT.items()}, "other_ranking_credit": 0,
                "files": {file: digest(content) for file, content in payloads.items()}}
    output.mkdir(parents=True, exist_ok=False)
    for file, content in {**payloads, "manifest.json": json_bytes(manifest)}.items():
        with (output / file).open("xb") as handle:
            handle.write(content)
    verify(output)
    return manifest


def verify(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    listed = set(manifest.get("files", {}))
    if not set(PAYLOAD_FILES) <= listed or listed - set(PAYLOAD_FILES) - {"source_provenance.json"}:
        raise ValueError("manifest file list is incomplete or unexpected")
    for file, sha256 in manifest["files"].items():
        if digest((directory / file).read_bytes()) != sha256:
            raise ValueError(f"hash mismatch: {file}")
    questions, _ = rows(directory / "questions.jsonl")
    qids = [q["question_id"] for q in questions]
    if len(qids) != len(set(qids)):
        raise ValueError("release contains duplicate question IDs")
    for question in questions:
        validate_question(question)
    split_doc = json.loads((directory / "split.json").read_text())
    assigned = [qid for split in SPLITS for qid in split_doc[split]]
    if sorted(assigned) != sorted(qids) or set(split_doc["group_by_question"]) != set(qids):
        raise ValueError("split must assign every question exactly once")
    splits = {qid: split for split in SPLITS for qid in split_doc[split]}
    group_rows = defaultdict(list)
    for question in questions:
        group_rows[split_doc["group_by_question"][question["question_id"]]].append(question)
    if set(group_rows) != set(split_doc["groups"]):
        raise ValueError("group map is incomplete or unexpected")
    for gid, members in group_rows.items():
        group = split_doc["groups"][gid]
        if sorted(q["question_id"] for q in members) != group["question_ids"]:
            raise ValueError(f"{gid}: inconsistent membership")
        if any(splits[q["question_id"]] != group["split"] for q in members):
            raise ValueError(f"{gid}: group crosses splits")
        if group["seen_before"] and group["split"] != "train":
            raise ValueError(f"{gid}: seen group in holdout")
    for group in build_groups(questions, []).values():
        if len({splits[q["question_id"]] for q in group["questions"]}) > 1:
            raise ValueError("shared group/dataset/candidate_set crosses splits")
    actual = Counter(q["source"] for q in questions)
    for source in SOURCES:
        if actual[source] != manifest["sources"][source]["selected_questions"]:
            raise ValueError(f"{source}: manifest count differs from payload")
    selection = json.loads((directory / "selection.json").read_text())
    if selection["selected_question_ids"] != sorted(qids):
        raise ValueError("selection differs from payload")
    if selection["selected_group_ids"] != sorted(group_rows):
        raise ValueError("selection group map differs from payload")
    for source in SOURCES:
        counts = {split: sum(q["source"] == source and splits[q["question_id"]] == split
                             for q in questions) for split in SPLITS}
        if (counts != manifest["sources"][source]["split_counts"]
                or counts != selection["actual_by_split"][source]):
            raise ValueError(f"{source}: split counts differ from payload")
    if manifest["ranking_credit"] != {str(k): v for k, v in RANK_CREDIT.items()} or manifest["other_ranking_credit"] != 0:
        raise ValueError("unexpected ranking credit rule")
    return manifest


def parse_pairs(items, value_type):
    pairs = {}
    for item in items:
        source, separator, value = item.partition("=")
        if not separator or source not in SOURCES or source in pairs:
            raise ValueError(f"expected one SOURCE=VALUE for each known source, got {item!r}")
        pairs[source] = value_type(value)
    return pairs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", default=[], metavar="SOURCE=PATH")
    parser.add_argument("--sample", action="append", default=[], metavar="SOURCE=N",
                        help="total target counts; defaults: arch170=80, ranking_v2=400, ranking_v3=300, dataflip500=200")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--name", default="kb_science_pool_v2")
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--provenance", type=Path, help="freeze a source and historical-exposure inventory with the release")
    parser.add_argument("--exclude-seen", action="append", type=Path, default=[],
                        help="JSONL records, JSON list or text question IDs, or a run directory with epochs/e*/records.jsonl")
    args = parser.parse_args(argv)
    try:
        if args.verify:
            if args.output or args.source or args.sample or args.exclude_seen or args.provenance:
                raise ValueError("--verify does not accept build inputs")
            manifest = verify(args.verify)
            directory = args.verify
        else:
            if not args.output:
                raise ValueError("--output is required for a new release")
            counts = {**DEFAULT_COUNTS, **parse_pairs(args.sample, int)}
            manifest = build(parse_pairs(args.source, Path), counts, args.exclude_seen, args.output,
                             args.seed, args.val_fraction, args.test_fraction, args.name, args.provenance)
            directory = args.output
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({"release": manifest["release"], "directory": str(directory),
                      "questions": sum(s["selected_questions"] for s in manifest["sources"].values()),
                      "manifest_sha256": digest((directory / "manifest.json").read_bytes())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
