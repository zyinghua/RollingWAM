"""Run the nine official RMBench tasks and aggregate success and reward."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SINGLE_ENTRY = PROJECT_ROOT / "experiments" / "rmbench" / "eval_rmbench_single.py"
TERMINATE_TIMEOUT_SEC = 10
POLL_INTERVAL_SEC = 2

M1_TASKS = (
    "observe_and_pickup",
    "rearrange_blocks",
    "put_back_block",
    "swap_blocks",
    "swap_T",
)
MN_TASKS = (
    "battery_try",
    "blocks_ranking_try",
    "cover_blocks",
    "press_button",
)
OFFICIAL_TASKS = M1_TASKS + MN_TASKS
TASK_TIER = {task: "M(1)" for task in M1_TASKS} | {task: "M(n)" for task in MN_TASKS}


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path.resolve()


def _resolve_ckpt_tag(ckpt_path: Path) -> str:
    parts = ckpt_path.resolve().parts
    if "runs" in parts:
        runs_idx = parts.index("runs")
        if runs_idx + 2 >= len(parts):
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., got: {ckpt_path}"
            )
        return f"{parts[runs_idx + 1]}_{parts[runs_idx + 2]}"
    return ckpt_path.stem


def _is_blocked_override(raw_override: str) -> bool:
    key = raw_override.split("=", 1)[0].lstrip("+~")
    if key in {
        "ckpt",
        "gpu_id",
        "EVALUATION.task_name",
        "EVALUATION.task_config",
        "EVALUATION.instruction_type",
        "EVALUATION.output_dir",
    }:
        return True
    return key.startswith("MULTIRUN.") or key.startswith("hydra.")


def _collect_worker_overrides() -> list[str]:
    return [ov for ov in HydraConfig.get().overrides.task if not _is_blocked_override(ov)]


def _load_result(result_file: Path) -> dict[str, Any]:
    if not result_file.is_file():
        raise FileNotFoundError(f"Result file not found: {result_file}")
    payload = json.loads(result_file.read_text(encoding="utf-8"))
    for key in ("success_rate", "reward", "episodes"):
        if key not in payload:
            raise ValueError(f"Missing {key!r} in result file: {result_file}")
    payload["success_rate"] = float(payload["success_rate"])
    payload["episodes"] = int(payload["episodes"])
    reward = payload["reward"]
    if isinstance(reward, list):
        payload["reward"] = [float(value) for value in reward]
    else:
        payload["reward"] = float(reward)
    return payload


def _reward_scalar(reward: float | list[float]) -> float:
    if isinstance(reward, list):
        if not reward:
            raise ValueError("Reward list must not be empty.")
        return float(sum(reward) / len(reward))
    return float(reward)


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


@dataclass
class RunningState:
    task_name: str
    gpu_id: int
    process: subprocess.Popen[str]


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_rmbench.yaml")
def main(cfg: DictConfig) -> None:
    if cfg.ckpt is None:
        raise ValueError("`ckpt` must not be None.")
    if not SINGLE_ENTRY.is_file():
        raise FileNotFoundError(f"Single evaluation entry not found: {SINGLE_ENTRY}")

    ckpt_path = _resolve_path(str(cfg.ckpt), base=PROJECT_ROOT)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt_tag = _resolve_ckpt_tag(ckpt_path)

    rmbench_root = _resolve_path(str(cfg.EVALUATION.rmbench_root), base=PROJECT_ROOT)
    if not rmbench_root.is_dir():
        raise FileNotFoundError(f"RMBench root not found: {rmbench_root}")

    configured_task = cfg.EVALUATION.task_name
    if configured_task is None or str(configured_task).strip() == "":
        tasks = list(OFFICIAL_TASKS)
    else:
        task_name = str(configured_task)
        if task_name not in OFFICIAL_TASKS:
            raise ValueError(
                f"Not an official RMBench task: {task_name}. Expected one of {OFFICIAL_TASKS}."
            )
        tasks = [task_name]

    if str(cfg.EVALUATION.task_config) != "demo_clean":
        raise ValueError("RMBench evaluation must use EVALUATION.task_config=demo_clean.")
    instruction_type = str(cfg.EVALUATION.instruction_type)
    if instruction_type not in {"seen", "unseen"}:
        raise ValueError("EVALUATION.instruction_type must be 'seen' or 'unseen'.")

    num_gpus = int(cfg.MULTIRUN.num_gpus)
    max_tasks_per_gpu = int(cfg.MULTIRUN.max_tasks_per_gpu)
    first_gpu_id = int(cfg.gpu_id)
    if num_gpus <= 0 or max_tasks_per_gpu <= 0 or first_gpu_id < 0:
        raise ValueError(
            "MULTIRUN.num_gpus and max_tasks_per_gpu must be > 0; gpu_id must be >= 0."
        )
    gpu_ids = list(range(first_gpu_id, first_gpu_id + num_gpus))

    output_dir = _resolve_path(str(cfg.EVALUATION.output_dir), base=PROJECT_ROOT)
    run_ts = output_dir.name
    run_output_dir = PROJECT_ROOT / "evaluate_results" / "rmbench" / ckpt_tag / run_ts
    run_output_dir.mkdir(parents=True, exist_ok=True)

    manager_log = run_output_dir / "manager.log"
    failed_tasks_file = run_output_dir / "failed_tasks.txt"
    summary_csv = run_output_dir / "summary.csv"
    summary_json = run_output_dir / "summary.json"
    OmegaConf.save(config=cfg, f=str(run_output_dir / "manager_config.yaml"))

    extra_overrides = _collect_worker_overrides()
    pending_tasks = deque(tasks)
    running: list[RunningState] = []
    results: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []

    def log(message: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with manager_log.open("a", encoding="utf-8") as file:
            file.write(line + "\n")

    def build_cmd(task_name: str, gpu_id: int) -> list[str]:
        cmd = [
            sys.executable,
            str(SINGLE_ENTRY),
            f"ckpt={ckpt_path}",
            f"gpu_id={gpu_id}",
            f"EVALUATION.task_name={task_name}",
            "EVALUATION.task_config=demo_clean",
            f"EVALUATION.instruction_type={instruction_type}",
            f"EVALUATION.output_dir={output_dir}",
        ]
        cmd.extend(extra_overrides)
        return cmd

    def launch(task_name: str, gpu_id: int) -> RunningState:
        cmd = build_cmd(task_name, gpu_id)
        log(f"launch task={task_name} tier={TASK_TIER[task_name]} gpu={gpu_id}")
        return RunningState(
            task_name=task_name,
            gpu_id=gpu_id,
            process=subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), text=True),
        )

    def gpu_running_count(gpu_id: int) -> int:
        return sum(
            state.gpu_id == gpu_id and state.process.poll() is None for state in running
        )

    def launch_pending(gpu_id: int) -> None:
        while pending_tasks and gpu_running_count(gpu_id) < max_tasks_per_gpu:
            running.append(launch(pending_tasks.popleft(), gpu_id))

    def terminate_running() -> None:
        for state in running:
            if state.process.poll() is None:
                state.process.terminate()
        deadline = time.time() + TERMINATE_TIMEOUT_SEC
        for state in running:
            if state.process.poll() is not None:
                continue
            try:
                state.process.wait(timeout=max(0.0, deadline - time.time()))
            except subprocess.TimeoutExpired:
                state.process.kill()
                state.process.wait()

    def write_outputs() -> None:
        rows = []
        for task_name in tasks:
            result = results.get(task_name)
            rows.append(
                {
                    "task_name": task_name,
                    "tier": TASK_TIER[task_name],
                    "instruction_type": instruction_type,
                    "episodes": None if result is None else result["episodes"],
                    "success_rate": None if result is None else result["success_rate"],
                    "reward": None if result is None else result["reward"],
                }
            )

        aggregates: dict[str, dict[str, float | None]] = {}
        tier_groups = (
            ("M(1)", M1_TASKS),
            ("M(n)", MN_TASKS),
            ("overall", OFFICIAL_TASKS),
        )
        for tier, tier_tasks in tier_groups:
            selected = [results[task] for task in tier_tasks if task in results]
            aggregates[tier] = {
                "mean_success_rate": _mean([item["success_rate"] for item in selected]),
                "mean_reward": _mean([_reward_scalar(item["reward"]) for item in selected]),
            }

        with summary_csv.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=(
                    "task_name",
                    "tier",
                    "instruction_type",
                    "episodes",
                    "success_rate",
                    "reward",
                ),
            )
            writer.writeheader()
            for row in rows:
                csv_row = dict(row)
                if isinstance(csv_row["reward"], list):
                    csv_row["reward"] = json.dumps(csv_row["reward"])
                writer.writerow(csv_row)

        summary_json.write_text(
            json.dumps(
                {
                    "instruction_type": instruction_type,
                    "per_task": rows,
                    "aggregates": aggregates,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        failed_tasks_file.write_text(
            "".join(
                f"{item['task_name']},gpu={item['gpu_id']},"
                f"return_code={item['return_code']},reason={item['reason']}\n"
                for item in failures
            ),
            encoding="utf-8",
        )

    log(
        f"manager start tasks={len(tasks)} instruction={instruction_type} "
        f"gpu_ids={gpu_ids} max_tasks_per_gpu={max_tasks_per_gpu}"
    )
    for gpu_id in gpu_ids:
        launch_pending(gpu_id)

    failure_message: str | None = None
    try:
        while running:
            progressed = False
            for state in list(running):
                return_code = state.process.poll()
                if return_code is None:
                    continue
                progressed = True
                running.remove(state)
                if return_code != 0:
                    failure_message = (
                        f"worker failed: task={state.task_name}, gpu={state.gpu_id}, "
                        f"return_code={return_code}"
                    )
                    failures.append(
                        {
                            "task_name": state.task_name,
                            "gpu_id": state.gpu_id,
                            "return_code": return_code,
                            "reason": "process_failed",
                        }
                    )
                    log(failure_message)
                    terminate_running()
                    running.clear()
                    break

                result_file = (
                    run_output_dir / state.task_name / instruction_type / "result.json"
                )
                try:
                    result = _load_result(result_file)
                except Exception as exc:
                    failure_message = f"result parse failed for {state.task_name}: {exc!r}"
                    failures.append(
                        {
                            "task_name": state.task_name,
                            "gpu_id": state.gpu_id,
                            "return_code": return_code,
                            "reason": "result_parse_failed",
                        }
                    )
                    log(failure_message)
                    terminate_running()
                    running.clear()
                    break

                results[state.task_name] = result
                log(
                    f"done task={state.task_name} success={result['success_rate']:.4f} "
                    f"reward={result['reward']}"
                )
                launch_pending(state.gpu_id)

            if failure_message is not None:
                break
            if not progressed:
                time.sleep(POLL_INTERVAL_SEC)
    except (KeyboardInterrupt, SystemExit):
        terminate_running()
        raise

    if failure_message is not None:
        for task_name in pending_tasks:
            failures.append(
                {
                    "task_name": task_name,
                    "gpu_id": -1,
                    "return_code": -1,
                    "reason": "aborted_not_started",
                }
            )

    write_outputs()
    log(f"summary saved: {summary_csv} and {summary_json}")
    if failure_message is not None:
        raise RuntimeError(failure_message)
    log("manager finished successfully")


if __name__ == "__main__":
    main()
