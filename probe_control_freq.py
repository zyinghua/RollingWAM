"""Control frequency / chunk duration, from ground truth. Standalone — no repo files
modified; delete to revert.

The control frequency is NOT a computed constant anywhere in the code: the loader reads
it verbatim from the dataset's meta/info.json ("fps") and uses it to space the actions
(delta_timestamps = t / fps, global_sample_stride=1). So:
    one N-action chunk  ==  N / fps  seconds.

Run on the server:
    python probe_control_freq.py <dataset_root> [eval_video.mp4]
      <dataset_root> is the dir that contains meta/info.json
      (e.g. /datasets/robotwin2.0-fastwam/robotwin2.0)

The optional eval video is only a STEP-COUNT cross-check: RoboTwin writes exactly one
video frame per executed action step (eval_video_freq=1). Its encoded framerate is a
hardcoded display value (10, eval_policy.py) and is NOT the control frequency — never
divide frames by the video fps to get frequency.
"""
import json
import subprocess
import sys
from pathlib import Path

ACTIONS_PER_CHUNK = 16

if len(sys.argv) < 2:
    sys.exit("usage: python probe_control_freq.py <dataset_root> [eval_video.mp4]")
dataset_root = Path(sys.argv[1])

info_path = dataset_root / "meta" / "info.json"
if not info_path.exists():
    sys.exit(f"info.json not found at {info_path}")
info = json.loads(info_path.read_text())
if "fps" not in info:
    sys.exit(f"'fps' key missing from {info_path}; keys present: {sorted(info)}")
fps = info["fps"]

print(f"[GROUND TRUTH]  {info_path}")
print(f"[GROUND TRUTH]  fps = {fps} Hz   <-- the control frequency the model was trained/consumed at")
print(f"[GROUND TRUTH]  one {ACTIONS_PER_CHUNK}-action chunk = {ACTIONS_PER_CHUNK} / {fps} = {ACTIONS_PER_CHUNK / fps:.4f} s")
print(f"[GROUND TRUTH]  full W-window (5 chunks = 80 actions) = {80 / fps:.4f} s")

if len(sys.argv) > 2:
    video = sys.argv[2]
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames,r_frame_rate",
        "-of", "json", video,
    ])
    st = json.loads(out)["streams"][0]
    nframes = int(st["nb_read_frames"])
    vfps = st.get("r_frame_rate")
    print()
    print(f"[cross-check]   {video}: {nframes} frames, encoded r_frame_rate={vfps}")
    print(f"[cross-check]   RoboTwin writes 1 frame per executed action step, so")
    print(f"[cross-check]   frames == total action steps executed in this episode.")
    print(f"[cross-check]   steps/chunk = {nframes} / (chunks executed); one chunk -> expect {ACTIONS_PER_CHUNK}.")
    print(f"[warning]       encoded fps {vfps} is a HARDCODED display value (10), NOT the control frequency.")
