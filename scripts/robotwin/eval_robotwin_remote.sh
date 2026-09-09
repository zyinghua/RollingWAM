#!/usr/bin/env bash
# Run on the RoboTwin simulator machine, with its RoboTwin environment active.
# The model server must already be running at --server-uri.
#
# Same numbered shards as eval_full_tasks_rolling.sh (0-16):
#   bash scripts/robotwin/eval_robotwin_remote.sh \
#     --server-uri ws://model-host:8000 --shard-id 0 --gpu-id 0
# Each shard defaults to clean then randomized per task, 100 episodes, seed 42,
# and observation skipping, matching the local full-evaluation script.
# Resume after two completed task/phase evaluations: add --skip-completed 2.
# Shard 16 contains two tasks (four evaluations); other shards contain three (six).
# One server accepts one client; use separate server ports for concurrent shards.
#
# One task:
#   bash scripts/robotwin/eval_robotwin_remote.sh \
#     --server-uri ws://model-host:8000 --task lift_pot --task-config demo_randomized --seed 42 --gpu-id 0
#
# Selected six tasks, clean and randomized:
#   bash scripts/robotwin/eval_robotwin_remote.sh \
#     --server-uri ws://model-host:8000 --selected-tasks --task-config both \
#     --skip-get-obs-within-replan --gpu-id 0
#
# All 50 tasks in shard order, clean and randomized:
#   bash scripts/robotwin/eval_robotwin_remote.sh \
#     --server-uri ws://model-host:8000 --all-tasks --gpu-id 0
# Match eval_selected_tasks_rolling.sh defaults with:
#   --selected-tasks --task-config demo_clean --seed 42 --skip-get-obs-within-replan
# Pass --help for evaluation, output, and transport options.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
exec "${PYTHON_BIN:-python}" experiments/robotwin/eval_robotwin_remote.py "$@"
