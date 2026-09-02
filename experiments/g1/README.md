# G1 real-robot inference

`scripts/g1/serve.py` exposes a trained G1 RollingWAM checkpoint through
OmniRobot's native websocket protocol.

For an existing container created before the serving dependencies were added:

```bash
pip install 'msgpack==1.1.2' 'websockets==16.0'
```

Start the server on one GPU:

```bash
CUDA_VISIBLE_DEVICES=7 \
DIFFSYNTH_MODEL_BASE_PATH=/workspace/RollingWAM/checkpoints \
DIFFSYNTH_SKIP_DOWNLOAD=true \
python scripts/g1/serve.py \
  --checkpoint /workspace/RollingWAM/runs/g1_pnp_pour_rolling_1cam_320_1e-4/<run>/checkpoints/weights/step_000530.pt \
  --device cuda:0 \
  --num-steps 10 \
  --host 0.0.0.0 \
  --port 8000
```

The resolved `config.yaml` and `dataset_stats.json` are loaded automatically
from the checkpoint's run directory. Use `--config` or `--dataset-stats` only
for a checkpoint stored outside its original run.

The websocket sends metadata immediately after connecting. Requests and
responses use OmniRobot's NumPy-aware MessagePack encoding:

```python
request = {
    "images": {"ego_view": rgb_uint8_hwc},
    "states": {"state": state_float32_43},
    "text": "the task instruction",  # required unless --default-instruction is set
}

response = {
    "action": action_float32_4_by_78,
}
```

The 78 action values remain in the trained SONIC layout: 64 motion-token
values, 7 left-hand values, and 7 right-hand values. The client must execute
all four rows at 10 Hz before requesting the next chunk. Robot-side safety
checks, rate limits, emergency stop, and controller delivery are intentionally
not implemented in the model server.

RollingWAM is stateful. Opening a connection resets its rolling window, closing
the connection resets it again, and changing `text` resets it before inference.
Use a fresh websocket connection for every new episode. Only one client may be
connected at a time.
