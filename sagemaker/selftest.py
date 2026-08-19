#!/usr/bin/env python3
"""Offline checks of the launcher <-> container seam. Torch-free, AWS-free.

    python3 sagemaker/selftest.py

Covers: the train_argv round trip, placeholder expansion, per-key override
merging, resourceconfig.json -> multi-node params, double-encoded
hyperparameter decoding, auto-resume checkpoint discovery, target-YAML
validation (placeholders name real channels), queue/instance tables, and the
repo paths entry.py depends on. Run after every change to sagemaker/.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
import tempfile
from pathlib import Path

SM_DIR = Path(__file__).resolve().parent
REPO_ROOT = SM_DIR.parent
sys.path.insert(0, str(SM_DIR))

import entry  # noqa: E402
import launch_sm  # noqa: E402
import sm_env  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}  {detail}")
        FAILURES.append(name)


def test_env_prefix_contract() -> None:
    print("ENV_PREFIX contract (launch_sm and sm_env are kept in sync by hand):")
    check("launch_sm.ENV_PREFIX == sm_env.ENV_PREFIX",
          launch_sm.ENV_PREFIX == sm_env.ENV_PREFIX,
          f"{launch_sm.ENV_PREFIX!r} vs {sm_env.ENV_PREFIX!r}")


def test_train_argv_roundtrip() -> None:
    print("train_argv shlex round trip:")
    argv = ["task=robotwin_rolling_3cam_384_1e-4", "batch_size=8",
            "+data.train.override_instruction=Pick the sauce can, then stop"]
    check("join/split identity", shlex.split(shlex.join(argv)) == argv)


def test_hyperparameter_double_decode() -> None:
    print("double-encoded hyperparameter decoding:")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({
            "train_argv": json.dumps("task=x batch_size=2"),   # double-encoded
            "sm_overrides": "output_dir={checkpoint_dir}/x",   # plain
            "count": 3,                                        # non-string
            "listish": json.dumps([1, 2]),                     # decodes to non-str
        }, f)
        path = f.name
    orig = sm_env.SM_HYPERPARAMS
    try:
        sm_env.SM_HYPERPARAMS = Path(path)
        hp = sm_env.read_hyperparameters()
    finally:
        sm_env.SM_HYPERPARAMS = orig
        os.unlink(path)
    check("double-encoded decoded once", hp["train_argv"] == "task=x batch_size=2")
    check("plain string untouched", hp["sm_overrides"] == "output_dir={checkpoint_dir}/x")
    check("non-string passthrough", hp["count"] == 3)
    check("non-str JSON kept raw", hp["listish"] == json.dumps([1, 2]))


def test_expand() -> None:
    print("placeholder expansion:")
    out = entry.expand(
        "data.train.dataset_dirs=[{channel:robotwin}]", task="t")
    check("{channel:NAME}", out == "data.train.dataset_dirs=[/opt/ml/input/data/robotwin]", out)
    out = entry.expand("{checkpoint_dir}/{task}", task="mytask")
    check("{checkpoint_dir}/{task}", out == "/opt/ml/checkpoints/mytask", out)
    out = entry.expand("a={channel:x}/f.pt b={channel:y}", task="t")
    check("multiple channels", out == "a=/opt/ml/input/data/x/f.pt b=/opt/ml/input/data/y", out)
    check("{pretrained_dir}", entry.expand("{pretrained_dir}", task="t") == "/opt/ml/pretrained")
    check("{job} fallback", entry.expand("{job}", task="t") == "job")


def test_merge_overrides() -> None:
    print("per-key override merging:")
    base = ["output_dir=/opt/ml/checkpoints/t", "batch_size=16", "+data.train.foo=1"]
    user = ["batch_size=4", "data.train.foo=2", "max_steps=10"]
    merged = entry.merge_overrides(base, user)
    check("user wins per key", "batch_size=4" in merged and "batch_size=16" not in merged)
    check("+prefix dedupes against plain",
          "data.train.foo=2" in merged and "+data.train.foo=1" not in merged)
    check("base survives when unopposed", "output_dir=/opt/ml/checkpoints/t" in merged)
    check("new user keys appended", "max_steps=10" in merged)
    keys = [entry.override_key(t) for t in merged]
    check("no duplicate keys", len(keys) == len(set(keys)))
    check("resolve_task", entry.resolve_task(["a=1", "task=foo"]) == "foo")
    check("resolve_task default", entry.resolve_task(["a=1"]) == "train")


def test_distributed_env() -> None:
    print("resourceconfig.json -> multi-node params:")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"hosts": ["algo-2", "algo-1", "algo-3"], "current_host": "algo-3"}, f)
        path = f.name
    orig = sm_env.SM_RESOURCECONFIG
    os.environ["SM_NUM_GPUS"] = "8"
    try:
        sm_env.SM_RESOURCECONFIG = Path(path)
        dist = sm_env.distributed_env()
    finally:
        sm_env.SM_RESOURCECONFIG = orig
        del os.environ["SM_NUM_GPUS"]
        os.unlink(path)
    check("hosts sorted, algo-1 is master", dist["main_process_ip"] == "algo-1")
    check("machine_rank from sorted order", dist["machine_rank"] == 2)
    check("num_machines", dist["num_machines"] == 3)
    check("nproc from SM_NUM_GPUS", dist["nproc_per_node"] == 8)
    check("world size math", dist["nproc_per_node"] * dist["num_machines"] == 24)


def test_auto_resume() -> None:
    print("auto-resume checkpoint discovery (world_size=2):")

    def make_step(state: Path, name: str, *, sentinel: bool, shards: int) -> None:
        d = state / name / "pytorch_model"
        d.mkdir(parents=True)
        if sentinel:
            (state / name / "trainer_state.json").write_text("{}")
        for r in range(shards):
            (d / f"zero_pp_rank_{r}_mp_rank_00_optim_states.pt").write_text("x")

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "run"
        overrides = [f"output_dir={out_dir}", "task=x"]
        check("no state dir -> None", entry.find_auto_resume(overrides, 2) is None)
        state = out_dir / "checkpoints" / "state"
        make_step(state, "step_000100", sentinel=True, shards=2)
        make_step(state, "step_000250", sentinel=True, shards=2)
        (state / "not_a_step").mkdir()
        got = entry.find_auto_resume(overrides, 2)
        check("picks highest complete step", got == str(state / "step_000250"), str(got))
        # partial local save: shards synced but trainer_state.json missing
        make_step(state, "step_000300", sentinel=False, shards=2)
        got = entry.find_auto_resume(overrides, 2)
        check("skips dir without sentinel", got == str(state / "step_000250"), str(got))
        # partial S3 sync / wrong world size: sentinel present, shards short
        make_step(state, "step_000400", sentinel=True, shards=1)
        got = entry.find_auto_resume(overrides, 2)
        check("skips dir with missing shards", got == str(state / "step_000250"), str(got))
        check("explicit resume wins",
              entry.find_auto_resume([*overrides, "resume=/x"], 2) is None)
        os.environ[f"{sm_env.ENV_PREFIX}_AUTO_RESUME"] = "false"
        try:
            check("env disable", entry.find_auto_resume(overrides, 2) is None)
        finally:
            del os.environ[f"{sm_env.ENV_PREFIX}_AUTO_RESUME"]
        check("no output_dir -> None", entry.find_auto_resume(["task=x"], 2) is None)


def test_targets() -> None:
    print("target YAMLs (channels + overrides + placeholder wiring):")
    placeholder_re = re.compile(r"\{channel:([^}]+)\}")
    known = {"{checkpoint_dir}", "{pretrained_dir}", "{task}", "{job}"}
    for path in sorted((SM_DIR / "configs").glob("*.yaml")):
        cfg = launch_sm.load_target(path.stem)
        channels = cfg["channels"]
        check(f"{path.stem}: has task or requires CLI task",
              bool(cfg.get("task")), "no task: key")
        for name, spec in channels.items():
            check(f"{path.stem}: channel {name} s3 URI",
                  str(spec.get("s3", "")).startswith("s3://"), str(spec.get("s3")))
            check(f"{path.stem}: channel {name} mode valid",
                  spec.get("mode", "FastFile") in ("File", "FastFile"))
        strings = [str(o) for o in cfg["overrides"]] + [str(v) for v in (cfg.get("env") or {}).values()]
        for s in strings:
            for ch in placeholder_re.findall(s):
                check(f"{path.stem}: placeholder channel {ch!r} declared",
                      ch in channels, s)
            for brace in re.findall(r"\{[a-z_]+\}", s):
                check(f"{path.stem}: placeholder {brace} known", brace in known, s)
        if cfg.get("queue"):
            try:
                launch_sm.resolve_queue(cfg["queue"])
                check(f"{path.stem}: queue alias resolves", True)
            except ValueError as e:
                check(f"{path.stem}: queue alias resolves", False, str(e))


def test_launcher_tables() -> None:
    print("queue/instance tables:")
    seen: dict[str, str] = {}
    for aliases, (queue_name, inst) in launch_sm.QUEUE_MAP.items():
        for a in aliases:
            check(f"alias {a!r} unique", a not in seen, f"also maps to {seen.get(a)}")
            seen[a] = queue_name
        check(f"queue {queue_name}: instance key {inst!r} mapped",
              inst in launch_sm.INSTANCE_MAPPER)
    name = launch_sm.make_job_name(
        "Robotwin_Selected_Tasks_Rolling_3cam_384_1e-4!!" * 3, "some.user", "20260819-120000-123")
    check("job name <= 63 chars", len(name) <= 63, name)
    check("job name sanitized", re.fullmatch(r"[a-z0-9][a-z0-9-]*", name) is not None, name)


def test_repo_paths() -> None:
    print("repo paths entry.py depends on:")
    acc = REPO_ROOT / entry.ACCELERATE_CONFIG
    check(f"{entry.ACCELERATE_CONFIG} exists", acc.is_file())
    check(f"{entry.TRAIN_SCRIPT} exists", (REPO_ROOT / entry.TRAIN_SCRIPT).is_file())
    if acc.is_file():
        m = re.search(r"deepspeed_config_file:\s*(\S+)", acc.read_text())
        check("accelerate yaml names a ds config", m is not None)
        if m:
            check(f"ds config {m.group(1)} exists (repo-root relative)",
                  (REPO_ROOT / m.group(1)).is_file())
    check("rollingwam_sm package present", (SM_DIR / "rollingwam_sm" / "dataset.py").is_file())
    check("no sagemaker/__init__.py (would shadow the AWS SDK)",
          not (SM_DIR / "__init__.py").exists())


def main() -> int:
    tests = [
        test_env_prefix_contract, test_train_argv_roundtrip,
        test_hyperparameter_double_decode, test_expand, test_merge_overrides,
        test_distributed_env, test_auto_resume, test_targets,
        test_launcher_tables, test_repo_paths,
    ]
    for t in tests:
        t()
        print()
    if FAILURES:
        print(f"SELFTEST FAILED: {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("SELFTEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
