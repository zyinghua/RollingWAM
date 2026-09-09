"""Run RoboTwin on this machine and send chunk inference to a model server.

Only raw camera images, joint state, and the instruction cross the connection.
The server owns preprocessing and RollingWAM's rolling inference state; this
client owns the simulator and executes each returned action chunk in order.
"""

from __future__ import annotations

import atexit
import json
import logging
import math
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

# RoboTwin loads policies through a symlink in its own policy directory.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rollingwam.serving import msgpack_numpy


LOGGER = logging.getLogger(__name__)
PROTOCOL = "rollingwam.robotwin.v1"
CAMERAS = ("head_camera", "left_camera", "right_camera")


def _positive_timeout(value: Any, name: str, *, optional: bool = False) -> float | None:
    if value is None and optional:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive number" + (" or null" if optional else "")) from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _integer(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {result}")
    return result


class RemoteRollingWAMRobotWinPolicy:
    """Persistent, synchronous RoboTwin client for one model server session."""

    def __init__(
        self,
        server_uri: str,
        *,
        seed: int | None = None,
        open_timeout: float = 30.0,
        request_timeout: float | None = None,
    ) -> None:
        if not isinstance(server_uri, str) or not server_uri.startswith(("ws://", "wss://")):
            raise ValueError("server_uri must be a ws:// or wss:// WebSocket URL")
        self.server_uri = server_uri
        self.seed = None if seed is None else _integer(seed, "seed")
        self.open_timeout = _positive_timeout(open_timeout, "open_timeout")
        self.request_timeout = _positive_timeout(request_timeout, "request_timeout", optional=True)
        self.pending_actions: deque[np.ndarray] = deque()
        self._socket = None
        self.metadata: dict[str, Any] = {}

        # Import lazily so inspecting policy configuration does not open a socket.
        from websockets.sync.client import connect

        try:
            self._socket = connect(
                self.server_uri,
                compression=None,
                max_size=None,
                open_timeout=self.open_timeout,
                # The server runs synchronous GPU inference. Its event loop may
                # not answer pings during warmup; request_timeout governs waits.
                ping_interval=None,
            )
            # Metadata is sent only after the server accepts this exclusive session.
            self.metadata = self._receive(timeout=self.open_timeout)
            self._validate_metadata()
        except Exception as exc:
            self.close()
            raise RuntimeError(f"Could not initialize RoboTwin policy at {self.server_uri}: {exc}") from exc
        atexit.register(self.close)

    def _validate_metadata(self) -> None:
        if self.metadata.get("protocol") != PROTOCOL:
            raise ValueError(f"Expected server protocol {PROTOCOL!r}, got {self.metadata.get('protocol')!r}")
        self.actions_per_chunk = _integer(self.metadata.get("actions_per_chunk"), "actions_per_chunk", minimum=1)
        self.execute_horizon = _integer(self.metadata.get("execute_horizon"), "execute_horizon", minimum=1)
        self.window_blocks = _integer(self.metadata.get("window_blocks"), "window_blocks", minimum=1)
        for key in ("action_dim", "state_dim"):
            if _integer(self.metadata.get(key), key) != 14:
                raise ValueError(f"RoboTwin requires {key}=14, got {self.metadata[key]}")
        if self.execute_horizon > self.actions_per_chunk:
            raise ValueError("Server execute_horizon exceeds actions_per_chunk")
        if self.window_blocks > 1 and self.execute_horizon != self.actions_per_chunk:
            raise ValueError("RollingWAM with window_blocks > 1 must execute each complete action chunk")
        if self.metadata.get("seed") is not None:
            _integer(self.metadata["seed"], "server seed")

    def _receive(self, *, timeout: float | None) -> dict[str, Any]:
        response = self._socket.recv(timeout=timeout)
        if isinstance(response, str):
            raise RuntimeError(f"Model server reported an error:\n{response}")
        if not isinstance(response, (bytes, bytearray, memoryview)):
            raise ValueError("Model server sent a non-binary response")
        unpacked = msgpack_numpy.unpackb(response)
        if not isinstance(unpacked, dict):
            raise ValueError("Model server response must be a dictionary")
        return unpacked

    def _request(self, request: dict[str, Any]) -> dict[str, Any]:
        if self._socket is None:
            raise RuntimeError("RoboTwin policy connection is closed; start a new evaluation session")
        try:
            self._socket.send(msgpack_numpy.packb(request))
            response = self._receive(timeout=self.request_timeout)
            if response.get("op") != request["op"]:
                raise ValueError(f"Expected {request['op']!r} response, got {response.get('op')!r}")
            return response
        except Exception as exc:
            self.close()
            raise RuntimeError(
                f"RoboTwin policy {request['op']} failed at {self.server_uri}. "
                "The connection was closed because the model's rolling state may have advanced; "
                f"the request was not retried. {exc}"
            ) from exc

    def reset(self) -> None:
        """Reset both ends at every episode, including repeated instructions."""
        self.pending_actions.clear()
        response = self._request({"op": "reset", "seed": self.seed})
        if response.get("ok") is not True:
            self.close()
            raise RuntimeError(f"Model server did not acknowledge the episode reset: {response!r}")

    def close(self) -> None:
        """Release the server's exclusive session; safe to call more than once."""
        self.pending_actions.clear()
        socket, self._socket = self._socket, None
        if socket is not None:
            try:
                socket.close()
            except Exception:
                LOGGER.debug("Error closing RoboTwin policy connection", exc_info=True)

    def should_request_observation(self) -> bool:
        return not self.pending_actions

    @staticmethod
    def _wire_observation(observation: dict[str, Any]) -> dict[str, Any]:
        camera_payload: dict[str, Any] = {}
        for camera in CAMERAS:
            rgb = np.asarray(observation["observation"][camera]["rgb"])
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3 or min(rgb.shape[:2]) < 1:
                raise ValueError(f"{camera}.rgb must be a nonempty uint8 HWC RGB image, got {rgb.shape}/{rgb.dtype}")
            camera_payload[camera] = {"rgb": np.ascontiguousarray(rgb)}
        state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError(f"joint_action.vector must contain 14 finite values, got shape {state.shape}")
        return {"observation": camera_payload, "joint_action": {"vector": np.ascontiguousarray(state)}}

    def _fill_action_queue(self, observation: dict[str, Any], instruction: str) -> None:
        response = self._request(
            {"op": "infer", "observation": self._wire_observation(observation), "instruction": instruction}
        )
        try:
            action = response.get("action")
            if not isinstance(action, np.ndarray) or action.dtype != np.float32:
                raise ValueError("Server action must be a float32 NumPy array")
            if action.shape != (self.execute_horizon, 14) or not np.isfinite(action).all():
                raise ValueError(
                    f"Server action must have shape ({self.execute_horizon}, 14) and finite values, got {action.shape}"
                )
        except Exception:
            self.close()
            raise
        # Decoded arrays reference the immutable MessagePack payload; copy rows for the simulator.
        self.pending_actions.extend(row.copy() for row in action)

    def step(self, env: Any, observation: dict[str, Any] | None) -> None:
        if not self.pending_actions:
            if observation is None:
                raise ValueError("A fresh observation is required when requesting an action chunk")
            # Match the local policy: read the instruction at chunk boundaries,
            # and reset the rolling stream only through the episode reset hook.
            instruction = env.get_instruction()
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("RoboTwin instruction must be a nonempty string")
            self._fill_action_queue(observation, instruction)
        env.take_action(self.pending_actions.popleft(), action_type="qpos")


def encode_obs(observation: Any) -> Any:
    return observation


def get_model(usr_args: dict[str, Any]) -> RemoteRollingWAMRobotWinPolicy:
    model = RemoteRollingWAMRobotWinPolicy(
        usr_args.get("server_uri"),
        seed=usr_args.get("seed"),
        open_timeout=usr_args.get("connect_timeout", 30.0),
        request_timeout=usr_args.get("request_timeout"),
    )
    output_dir = usr_args.get("eval_output_dir")
    if output_dir:
        try:
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)
            metadata = {**model.metadata, "evaluation_seed": model.seed}
            (path / "server_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        except Exception:
            model.close()
            raise
    LOGGER.info(
        "Connected to %s: %d actions per chunk, executing %d, window_blocks=%d",
        model.server_uri,
        model.actions_per_chunk,
        model.execute_horizon,
        model.window_blocks,
    )
    return model


def eval(TASK_ENV: Any, model: RemoteRollingWAMRobotWinPolicy, observation: Any) -> None:
    model.step(TASK_ENV, observation)


def reset_model(model: RemoteRollingWAMRobotWinPolicy) -> None:
    model.reset()
