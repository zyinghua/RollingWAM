"""Run the official RoboTwin evaluator against a separate RollingWAM server.

The simulator process needs RoboTwin and the remote policy's transport
dependencies; checkpoints, normalization statistics, and model dependencies
remain on the server. Tasks and phases run sequentially on one server session
at a time.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_NAME = "rollingwam_remote_policy"
SELECTED_TASKS_CONFIG = "configs/data/robotwin_selected_tasks.yaml"
# Independent remote-evaluation copy of the original launcher's shard grouping.
SHARDS_CONFIG = Path(__file__).resolve().with_name("remote_eval_shards.txt")


def _positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return result


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def _nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-uri", required=True, help="Model server URI, e.g. ws://model-host:8000")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--task", help="One RoboTwin task name")
    selection.add_argument("--tasks", nargs="+", help="Task names, evaluated in this order")
    selection.add_argument(
        "--selected-tasks", nargs="?", const=SELECTED_TASKS_CONFIG, metavar="YAML",
        help=f"Read selected_task_names from YAML (default: {SELECTED_TASKS_CONFIG})",
    )
    selection.add_argument("--shard-id", type=int, choices=range(17), metavar="0-16", help="Use the same task shard as eval_full_tasks_rolling.sh")
    selection.add_argument("--all-tasks", action="store_true", help="Evaluate all tasks in shard order")
    parser.add_argument(
        "--task-config", choices=("demo_clean", "demo_randomized", "both"),
        help="Default: both for --shard-id/--all-tasks; demo_randomized otherwise",
    )
    parser.add_argument("--robotwin-root", default="third_party/RoboTwin")
    parser.add_argument("--gpu-id", help="Simulator CUDA_VISIBLE_DEVICES; defaults to the current environment")
    parser.add_argument("--seed", type=int, default=42, help="RoboTwin evaluation seed and model sampling seed (same default as local evaluation)")
    parser.add_argument("--instruction-type", choices=("seen", "unseen"), default="unseen")
    parser.add_argument("--eval-num-episodes", type=_positive_int, default=100)
    parser.add_argument(
        "--skip-completed", type=_nonnegative_int, default=0, metavar="N",
        help="Skip the first N task/phase evaluations, as in eval_full_tasks_rolling.sh (one clean or randomized run counts as one)",
    )
    parser.add_argument("--output-dir", help="New or empty output directory; paths are relative to the repository")
    parser.add_argument("--run-name", default="rollingwam_remote", help="Result label; no checkpoint is needed here")
    parser.add_argument("--smoothness-dir", help="Opt-in executed-action traces; paths are relative to the repository")
    parser.add_argument("--smoothness-method", default="Rolling-WAM", help="Method label stored in smoothness traces")
    parser.add_argument(
        "--skip-get-obs-within-replan", action=argparse.BooleanOptionalAction, default=None,
        help="Default: enabled for --shard-id/--all-tasks, disabled otherwise; affects randomized lighting and video",
    )
    parser.add_argument("--connect-timeout", type=_positive_float, default=30.0)
    parser.add_argument("--request-timeout", type=_positive_float, default=None, help="Response timeout in seconds; default: no limit")
    return parser


def _resolve_path(value: str) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(value)))
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _read_yaml(path: Path) -> Any:
    # Keep --help and explicit-task argument handling independent of PyYAML.
    import yaml

    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def _resolve_tasks(args: argparse.Namespace) -> list[str]:
    if args.task is not None:
        tasks = [args.task]
    elif args.tasks is not None:
        tasks = args.tasks
    elif args.selected_tasks is not None:
        path = _resolve_path(args.selected_tasks)
        config = _read_yaml(path)
        tasks = config.get("selected_task_names") if isinstance(config, dict) else None
        if not isinstance(tasks, list):
            raise ValueError(f"Expected a selected_task_names list in {path}")
    else:
        shards = [line.split() for line in SHARDS_CONFIG.read_text(encoding="utf-8").splitlines()]
        if len(shards) != 17 or any(len(shard) != (2 if i == 16 else 3) for i, shard in enumerate(shards)):
            raise ValueError(f"Expected 17 shard lines with three tasks each (two in shard 16) in {SHARDS_CONFIG}")
        if args.shard_id is not None:
            tasks = shards[args.shard_id]
        else:
            tasks = [task for shard in shards for task in shard]
    if not tasks:
        raise ValueError("The task selection is empty")
    if any(not isinstance(task, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", task) for task in tasks):
        raise ValueError("Task names must be valid RoboTwin module names (letters, numbers, underscores)")
    if len(tasks) != len(set(tasks)):
        raise ValueError("The task selection contains duplicate names")
    return list(tasks)


def _ensure_policy_symlink(robotwin_root: Path) -> None:
    source = PROJECT_ROOT / "experiments" / "robotwin" / POLICY_NAME
    target = robotwin_root / "policy" / POLICY_NAME
    if not (source / "deploy_policy.yml").is_file():
        raise FileNotFoundError(f"Remote policy configuration not found: {source / 'deploy_policy.yml'}")
    if not target.parent.is_dir():
        raise FileNotFoundError(f"RoboTwin policy directory not found: {target.parent}")
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise RuntimeError(f"Policy symlink conflict: {target} points to {target.resolve()}, expected {source}")
    elif target.exists():
        raise RuntimeError(f"Policy path already exists and is not a symlink: {target}")
    else:
        target.symlink_to(source.resolve(), target_is_directory=True)


def _build_command(args: argparse.Namespace, task: str, task_config: str, output_dir: Path) -> list[str]:
    overrides = {
        "task_name": task,
        "task_config": task_config,
        "ckpt_setting": args.run_name,
        "policy_name": POLICY_NAME,
        "server_uri": args.server_uri,
        "seed": args.seed,
        "instruction_type": args.instruction_type,
        "eval_num_episodes": args.eval_num_episodes,
        "eval_output_dir": str(output_dir),
        "skip_get_obs_within_replan": args.skip_get_obs_within_replan,
        "connect_timeout": args.connect_timeout,
        "request_timeout": args.request_timeout,
        "smoothness_dir": str(_resolve_path(args.smoothness_dir)) if args.smoothness_dir else None,
        "smoothness_method": args.smoothness_method,
    }
    command = [sys.executable, "-u", "script/eval_policy.py", "--config", f"policy/{POLICY_NAME}/deploy_policy.yml", "--overrides"]
    for key, value in overrides.items():
        # RoboTwin parses each value as a Python literal. Popen receives an argv
        # list, so quotes, spaces, and shell metacharacters stay literal.
        command.extend((f"--{key}", repr(value)))
    return command


def _run_child(command: list[str], robotwin_root: Path, env: dict[str, str], log_file: Path) -> int:
    process: subprocess.Popen[str] | None = None
    with log_file.open("w", encoding="utf-8") as log:
        log.write(f"Command: {shlex.join(command)}\n\n")
        log.flush()
        try:
            process = subprocess.Popen(
                command, cwd=robotwin_root, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            return process.wait()
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def _parse_success_rate(path: Path) -> float:
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        try:
            rate = float(line.strip())
        except ValueError:
            continue
        if not math.isfinite(rate) or not 0 <= rate <= 1:
            raise ValueError(f"Invalid success rate in {path}: {rate}")
        return rate
    raise ValueError(f"No success rate found in {path}")


def _write_summary(output_dir: Path, records: list[dict[str, Any]]) -> None:
    means = {}
    for config in dict.fromkeys(record["task_config"] for record in records):
        completed = [r["success_rate"] for r in records if r["task_config"] == config and r["status"] == "completed"]
        means[config] = sum(completed) / len(completed) if completed else None
    summary = {"results": records, "mean_success_rate_completed_tasks": means}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    full_evaluation = args.shard_id is not None or args.all_tasks
    if args.task_config is None:
        args.task_config = "both" if full_evaluation else "demo_randomized"
    if args.skip_get_obs_within_replan is None:
        args.skip_get_obs_within_replan = full_evaluation
    uri = urlsplit(args.server_uri)
    if uri.scheme not in {"ws", "wss"} or not uri.hostname or uri.fragment:
        raise ValueError("--server-uri must be a ws:// or wss:// URI with a hostname and no fragment")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", args.run_name):
        raise ValueError("--run-name must start with a letter, number, or underscore and contain only letters, numbers, _, -, or .")
    robotwin_root = _resolve_path(args.robotwin_root)
    if not (robotwin_root / "script" / "eval_policy.py").is_file():
        raise FileNotFoundError(f"RoboTwin evaluator not found under {robotwin_root}")
    tasks = _resolve_tasks(args)
    task_configs = ["demo_clean", "demo_randomized"] if args.task_config == "both" else [args.task_config]
    records = [
        {"task_name": task, "task_config": task_config, "status": "pending", "success_rate": None}
        for task in tasks for task_config in task_configs
    ]
    if args.skip_completed > len(records):
        raise ValueError(f"--skip-completed cannot exceed {len(records)} for this task/phase selection")
    for record in records[: args.skip_completed]:
        record["status"] = "skipped"
    for task_config in task_configs:
        if not (robotwin_root / "task_config" / f"{task_config}.yml").is_file():
            raise FileNotFoundError(f"RoboTwin task configuration not found: {task_config}")
    output_dir = _resolve_path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "evaluate_results" / "robotwin_remote" / args.run_name / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Output directory must be new or empty to protect existing results: {output_dir}")
    _ensure_policy_symlink(robotwin_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_args = {**vars(args), "robotwin_root": str(robotwin_root), "output_dir": str(output_dir), "resolved_tasks": tasks}
    (output_dir / "run_args.json").write_text(json.dumps(run_args, indent=2) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if args.gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    _write_summary(output_dir, records)
    print(f"Evaluating {len(tasks)} task(s) sequentially via {args.server_uri}. Output: {output_dir}", flush=True)
    for run_index, record in enumerate(records, start=1):
        task, task_config = record["task_name"], record["task_config"]
        if record["status"] == "skipped":
            print(f"[{run_index}/{len(records)}] skipping completed task={task} config={task_config}", flush=True)
            continue
        task_dir = output_dir / task / task_config
        task_dir.mkdir(parents=True)
        log_file = task_dir / "eval.log"
        record.update(status="running", output_dir=str(task_dir), log_file=str(log_file))
        _write_summary(output_dir, records)
        print(f"\n[{run_index}/{len(records)}] {task} / {task_config}", flush=True)
        try:
            return_code = _run_child(_build_command(args, task, task_config, task_dir), robotwin_root, env, log_file)
            record["return_code"] = return_code
            if return_code != 0:
                raise RuntimeError(f"RoboTwin evaluation failed with return code {return_code}. Log: {log_file}")
            suffix = "clean" if task_config == "demo_clean" else "random"
            record["success_rate"] = _parse_success_rate(task_dir / f"_result_{suffix}.txt")
            record["status"] = "completed"
        except BaseException as error:
            record["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            record["error"] = str(error)
            _write_summary(output_dir, records)
            raise
        _write_summary(output_dir, records)
        print(f"Success rate: {record['success_rate']:.2%}", flush=True)
    print(f"Evaluation finished. Summary: {output_dir / 'summary.json'}", flush=True)
    return output_dir


def _handle_termination(_signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, _handle_termination)
    try:
        run(args)
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
