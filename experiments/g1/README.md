# G1 real-robot inference

`scripts/serve.py` exposes a trained RollingWAM checkpoint through a binary
WebSocket request/reply API. The options below configure the G1 observation and
action schema.

Install the serving dependencies if they are unavailable:

```bash
pip install 'msgpack==1.1.2' 'websockets==16.0'
```

Start the server on one GPU:

```bash
CUDA_VISIBLE_DEVICES=7 \
DIFFSYNTH_MODEL_BASE_PATH=/workspace/RollingWAM/checkpoints \
DIFFSYNTH_SKIP_DOWNLOAD=true \
python scripts/serve.py \
  --checkpoint /workspace/RollingWAM/runs/g1_pnp_pour_rolling_1cam_320_1e-4/<run>/checkpoints/weights/step_000530.pt \
  --device cuda:0 \
  --num-steps 10 \
  --embodiment unitree_g1_sonic \
  --image-key ego_view \
  --state-key state \
  --action-key action \
  --fps 10 \
  --host 0.0.0.0 \
  --port 8000
```

The resolved `config.yaml` and `dataset_stats.json` are loaded automatically
from the checkpoint's run directory. Use `--config` or `--dataset-stats` only
for a checkpoint stored outside its original run.

The server sends metadata immediately after connecting. Requests and responses
use MessagePack with NumPy array support:

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
all four rows at 10 Hz before requesting the next chunk. The model server
returns predictions only; the robot controller owns safety checks, rate limits,
emergency stop, and actuation.

RollingWAM is stateful. Opening a connection resets its rolling window, closing
the connection resets it again, and changing `text` resets it before inference.
Use a fresh websocket connection for every new episode. Only one client may be
connected at a time.
