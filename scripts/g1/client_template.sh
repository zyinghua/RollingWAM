#!/usr/bin/env bash
# Send one synthetic observation to a RollingWAM policy server.
# Usage:
#   bash scripts/g1/client_template.sh [server_uri] [task_instruction]
# Example:
# bash scripts/g1/client_template.sh \
#  ws://127.0.0.1:8000 \
#  "pick up the container and pour its contents"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

SERVER_URI="${1:-ws://127.0.0.1:8000}"
TASK_INSTRUCTION="${2:-pick up the container and pour its contents}"

SERVER_URI="$SERVER_URI" \
TASK_INSTRUCTION="$TASK_INSTRUCTION" \
PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
PYTHONDONTWRITEBYTECODE=1 \
python - <<'PY'
import asyncio
import os

import numpy as np
import websockets

from rollingwam.serving import msgpack_numpy


async def receive_message(websocket):
    raw = await websocket.recv()
    if isinstance(raw, str):
        raise RuntimeError(raw)
    message = msgpack_numpy.unpackb(raw)
    if not isinstance(message, dict):
        raise TypeError(f"Expected a dictionary, got {type(message).__name__}.")
    return message


async def main():
    async with websockets.connect(
        os.environ["SERVER_URI"],
        compression=None,
        max_size=None,
    ) as websocket:
        metadata = await receive_message(websocket)
        print("Server metadata:", metadata)

        request = {
            "images": {
                # Replace this array with the latest HWC RGB camera frame.
                "ego_view": np.zeros((480, 640, 3), dtype=np.uint8),
            },
            "states": {
                # Replace this array with the latest 43D robot state.
                "state": np.zeros((43,), dtype=np.float32),
            },
            "text": os.environ["TASK_INSTRUCTION"],
            "embodiment_tag": "unitree_g1_sonic",
        }

        await websocket.send(msgpack_numpy.packb(request))
        response = await receive_message(websocket)

        if "action" not in response:
            raise KeyError(f"Response is missing 'action': {response.keys()}")

        action = np.asarray(response["action"])
        if action.ndim != 2 or action.shape[1] != 78:
            raise ValueError(f"Expected action shape [horizon, 78], got {action.shape}.")
        if action.dtype != np.float32:
            raise TypeError(f"Expected float32 actions, got {action.dtype}.")
        if not np.isfinite(action).all():
            raise ValueError("Response contains non-finite actions.")

        print("Response keys:", response.keys())
        print("Action shape:", action.shape)
        print("Action dtype:", action.dtype)
        print("All finite:", True)
        print("Action:\n", action)


asyncio.run(main())
PY
