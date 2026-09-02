#!/usr/bin/env python3
"""Serve a trained G1 RollingWAM over OmniRobot's native websocket protocol."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from experiments.g1.rollingwam_policy import (  # noqa: E402
    DEFAULT_EMBODIMENT,
    DEFAULT_TASK_CONFIG,
    RollingWAMG1Policy,
)
from experiments.g1.websocket_server import G1WebsocketPolicyServer  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a G1 RollingWAM checkpoint over an OmniRobot-compatible websocket."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="RollingWAM weights file, for example step_000530.pt.",
    )
    parser.add_argument(
        "--dataset-stats",
        default=None,
        help="dataset_stats.json. Defaults to the checkpoint's training run directory.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Resolved training config.yaml. Defaults to the checkpoint's run directory.",
    )
    parser.add_argument(
        "--task-config",
        default=DEFAULT_TASK_CONFIG,
        help="Fallback Hydra task config when no run config.yaml is available.",
    )
    parser.add_argument(
        "--embodiment",
        default=DEFAULT_EMBODIMENT,
        help="Embodiment tag advertised to OmniRobot clients.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=10,
        help="Rolling diffusion inference steps; must be divisible by checkpoint W.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--mixed-precision",
        choices=("no", "fp16", "bf16"),
        default="bf16",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--transport",
        choices=("websocket",),
        default="websocket",
        help="G1 uses OmniRobot's native websocket transport.",
    )
    parser.add_argument("--control-hz", "--fps", dest="control_hz", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--text-cfg-scale", type=float, default=1.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument(
        "--default-instruction",
        default="",
        help="Used only when a client omits or sends an empty text field.",
    )
    parser.add_argument("--compile-action-infer", action="store_true")
    parser.add_argument("--compile-vae-encode", action="store_true")
    parser.add_argument("--vae-encode-batch-size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    policy = RollingWAMG1Policy.from_checkpoint(
        args.checkpoint,
        dataset_stats_path=args.dataset_stats,
        config_path=args.config,
        task_config=args.task_config,
        device=args.device,
        mixed_precision=args.mixed_precision,
        num_inference_steps=args.num_steps,
        text_cfg_scale=args.text_cfg_scale,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        compile_action_infer=args.compile_action_infer,
        compile_vae_encode=args.compile_vae_encode,
        vae_encode_batch_size=args.vae_encode_batch_size,
        embodiment=args.embodiment,
        default_instruction=args.default_instruction,
        control_hz=args.control_hz,
    )
    server = G1WebsocketPolicyServer(
        policy,
        host=args.host,
        port=args.port,
        metadata=policy.server_metadata(),
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
