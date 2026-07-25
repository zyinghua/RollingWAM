"""Predict ONE rolling chunk with the real model, verify it is `actions_per_chunk`
steps, then stop. Standalone — no repo files modified; delete this file to revert.

Run on the server (GPU):
    PYTHONPATH=src python probe_chunk.py <checkpoint.pt> [num_inference_steps]

It skips the T5 text encoder (feeds a dummy conditioning context, since only the action
SHAPE is being checked) and prints progress at each stage so slow loads are visible.
"""
import sys
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from rollingwam.utils.config_resolvers import register_default_resolvers


def log(msg, t0):
    print(f"[{time.perf_counter() - t0:7.1f}s] {msg}", flush=True)


CKPT = sys.argv[1] if len(sys.argv) > 1 else None
S = int(sys.argv[2]) if len(sys.argv) > 2 else None
if CKPT is None:
    sys.exit("usage: python probe_chunk.py <checkpoint.pt> [num_inference_steps]")
if not Path(CKPT).exists():
    sys.exit(f"checkpoint not found: {CKPT}")

t0 = time.perf_counter()
register_default_resolvers()
with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base="1.3"):
    cfg = compose(config_name="sim_robotwin")
if S is None:
    S = int(cfg.EVALUATION.num_inference_steps)
log("config composed", t0)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
model_cfg.load_text_encoder = False  # shape check only; skip the heavy T5 load
context_len = int(model_cfg.tokenizer_max_len)
text_dim = int(model_cfg.video_dit_config.text_dim)

log("building model (5B DiT + VAE + ActionDiT; may take minutes) ...", t0)
model = instantiate(model_cfg, model_dtype=dtype, device=device)
log("model built", t0)

model.load_checkpoint(str(CKPT))
log("checkpoint loaded", t0)

model = model.to(device).eval()
log(f"model on {device}", t0)

aspc = int(model.actions_per_chunk)
proprio_dim = int(model.proprio_dim)

frame = (torch.rand(1, 3, 1, 384, 320, device=device, dtype=dtype) * 2 - 1)
context = torch.zeros(1, context_len, text_dim, device=device, dtype=dtype)
context_mask = torch.ones(1, context_len, device=device, dtype=torch.bool)
proprio = torch.zeros(proprio_dim, device=device, dtype=dtype)

log(f"running rolling_act (boundary init = {S} passes, first CUDA call warms up) ...", t0)
model.rolling_reset()
with torch.no_grad():
    out = model.rolling_act(
        new_frames=frame,
        context=context,
        context_mask=context_mask,
        proprio=proprio,
        num_inference_steps=S,
    )
log("rolling_act done", t0)

action = out["action"]
print(f"\nS={S}  W={model.window_blocks}  configured actions_per_chunk={aspc}")
print(f"emitted chunk action shape = {tuple(action.shape)}   (expected [{aspc}, {proprio_dim}])")
print(f"video from rolling_act = {out['video']}   (None on the non-joint deploy path)")
assert action.shape[0] == aspc, f"chunk has {action.shape[0]} steps, expected {aspc}"
print(f"OK: one predicted chunk == {aspc} action steps")
