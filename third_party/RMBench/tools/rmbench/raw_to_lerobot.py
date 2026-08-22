"""Convert one raw RMBench task (demo_clean) into the LeRobot v2.1 layout RollingWAM trains on.

Adapted from the LingBot-VA RMBench converter; every format decision below is
derived from RollingWAM's own consumers:

  * the vendored LeRobot v2.1 reader (src/rollingwam/datasets/lerobot/lerobot/)
    needs meta/{info.json,episodes.jsonl,episodes_stats.jsonl,tasks.jsonl},
    per-episode parquet with timestamp/frame_index/episode_index/index/
    task_index plus float32 'observation.state' and 'action', and one mp4 per
    (camera, episode) whose frame pts match the parquet timestamps.
  * the deploy policy (experiments/robotwin/rollingwam_policy/deploy_policy.py,
    reused for RMBench through a symlink) observes head/left/right camera RGB
    plus the live 14-dim ``joint_action/vector`` and executes predictions as
    absolute joint positions (``take_action(action_type="qpos")``) -> training
    state/action MUST come from ``joint_action/vector``, NOT ``endpose``.
  * configs/data/rmbench.yaml declares 3 cameras at [3, 240, 320], action 14,
    state 14, z-score normalization; normalization statistics are computed by
    RollingWAM itself on the first training run (dataset_stats.json), and video
    is VAE-encoded on the fly, so no latent or norm-stat step exists here.

Raw RMBench episodes (data/<task>/demo_clean/data/episodeN.hdf5) carry
joint_action/{left_arm,left_gripper,right_arm,right_gripper,vector(14)} and
per-camera JPEG streams under observation/{head,left,right}_camera/rgb at the
native 240x320 resolution, which is stored as-is (RollingWAM resizes to
240x320 for training anyway, so upscaling would only waste storage). The
natural-language condition comes from ``instructions/episodeN.json`` (field
``seen``); a task-level RMBench instruction JSON or a literal instruction can
be passed as an explicit fallback.

Camera name mapping (RoboTwin convention, matching configs/data/rmbench.yaml):
    observation.images.cam_high        <- observation/head_camera
    observation.images.cam_left_wrist  <- observation/left_camera
    observation.images.cam_right_wrist <- observation/right_camera

As in the released RoboTwin LeRobot data (and RMBench's own ACT/GO1 reference
converters), exported row ``t`` contains the image and joint state from raw
frame ``t`` and the action target from raw frame ``t+1``: each raw episode
becomes ``raw_length - 1`` aligned rows and the final raw frame is dropped
from the videos.

Usage
-----
    python tools/rmbench/raw_to_lerobot.py \
        --raw-root /datasets/RMBench-data/data \
        --task put_back_block \
        --out  /datasets/RMBench-data/lerobot/put_back_block
"""

import argparse
import json
from pathlib import Path

import av
import cv2
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CAM_MAP = {
    "observation.images.cam_high": "head_camera",
    "observation.images.cam_left_wrist": "left_camera",
    "observation.images.cam_right_wrist": "right_camera",
}
# RMBench LargeView cameras record at 320x240; stored without rescaling.
VIDEO_HEIGHT = 240
VIDEO_WIDTH = 320
VIDEO_CODEC = "libsvtav1"
VIDEO_PIX_FMT = "yuv420p"
VIDEO_OPTIONS = {"g": "2", "crf": "30"}
AV1_CODEC_ID = av.codec.Codec("av1", "r").id
ACTION_NAMES = (
    [f"left_joint{i}" for i in range(1, 7)] + ["left_gripper"]
    + [f"right_joint{i}" for i in range(1, 7)] + ["right_gripper"]
)
ACTION_DIM = len(ACTION_NAMES)  # 14


def _first_instruction(path, instruction_type):
    payload = json.loads(path.read_text(encoding="utf-8"))
    choices = payload.get(instruction_type)
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], str):
        raise ValueError(
            f"{path}: expected a non-empty string list at key {instruction_type!r}"
        )
    return choices[0].strip()


def episode_instruction(src, raw_idx, args, eval_reference=None):
    """Resolve the full-episode prompt, preferring episode-specific RMBench text."""
    if args.instruction:
        return args.instruction.strip()

    episode_path = src / "instructions" / f"episode{raw_idx}.json"
    if episode_path.is_file():
        instruction = _first_instruction(episode_path, args.instruction_type)
        # Eval regenerates prompts from the vendored task_instruction JSON, so a
        # divergent raw-data instruction would silently mismatch train and eval.
        if eval_reference is not None and instruction != eval_reference:
            raise ValueError(
                f"episode {raw_idx}: instruction differs from third_party/RMBench/"
                f"description/task_instruction/{args.task}.json "
                f"({args.instruction_type!r}): {instruction!r} vs {eval_reference!r}. "
                "Pass --instruction explicitly if this is intended."
            )
        return instruction

    if args.instruction_file:
        return _first_instruction(Path(args.instruction_file), args.instruction_type)

    raise FileNotFoundError(
        f"missing {episode_path}. Pass --instruction-file pointing to "
        f"third_party/RMBench/description/task_instruction/{args.task}.json, or pass "
        "--instruction. language_annotation.json contains low-level subtask "
        "durations, not the full instruction used at evaluation."
    )


def decode_rgb(encoded, source):
    """Decode one RMBench byte payload, preserving its legacy RGB channel order.

    RMBench passes simulator RGB arrays directly to ``cv2.imencode``. Although
    OpenCV calls the decoded array BGR, its numeric channel order here is still
    the original RGB. Treating it as ordinary BGR would swap red and blue.
    """
    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode RGB bytes from {source}")
    return image


def read_joint_vector(f, h5_path):
    """Read the 14-dim absolute joint vector and cross-check it against its parts."""
    vector = np.asarray(f["joint_action/vector"], dtype=np.float32)  # (raw_T, 14)
    if vector.ndim != 2 or vector.shape[1] != ACTION_DIM:
        raise ValueError(f"{h5_path}: joint_action/vector has shape {vector.shape}")

    parts = np.concatenate(
        [
            np.asarray(f["joint_action/left_arm"], dtype=np.float32),
            np.asarray(f["joint_action/left_gripper"], dtype=np.float32)[:, None],
            np.asarray(f["joint_action/right_arm"], dtype=np.float32),
            np.asarray(f["joint_action/right_gripper"], dtype=np.float32)[:, None],
        ],
        axis=1,
    )
    if parts.shape != vector.shape or not np.allclose(parts, vector, atol=1e-6):
        raise ValueError(
            f"{h5_path}: joint_action/vector does not equal "
            "[left_arm, left_gripper, right_arm, right_gripper]"
        )
    return vector


def feat_stats(arr):
    """min/max/mean/std/count in the shape episodes_stats.jsonl uses."""
    a = np.asarray(arr, dtype=np.float64).reshape(len(arr), -1)
    return {
        "min": a.min(0).tolist(), "max": a.max(0).tolist(),
        "mean": a.mean(0).tolist(), "std": a.std(0).tolist(), "count": [len(a)],
    }


def img_stats(frames):
    """Per-channel stats over a subsample of decoded frames, values in [0,1], shape [3,1,1]."""
    a = np.stack(frames).astype(np.float64) / 255.0          # (n, H, W, 3)
    def per(function):
        return [[[float(value)]] for value in function(a, axis=(0, 1, 2))]

    return {"min": per(np.min), "max": per(np.max), "mean": per(np.mean),
            "std": per(np.std), "count": [len(frames)]}


def image_stat_indices(length):
    """Mirror LeRobot v2.1 compute_stats.sample_indices."""
    min_samples = min(100, length)
    num_samples = max(min_samples, min(int(length ** 0.75), 10_000))
    return set(np.round(np.linspace(0, length - 1, num_samples)).astype(int).tolist())


def downsample_stat_image(image, target_size=150, max_size_threshold=300):
    """Mirror LeRobot's inexpensive spatial subsampling for image statistics."""
    height, width = image.shape[:2]
    if max(width, height) < max_size_threshold:
        return image
    factor = int(width / target_size) if width > height else int(height / target_size)
    return image[::factor, ::factor]


def encode_av1_video(video_path, rgb_frames, fps):
    """Encode RGB frames with LeRobot's default AV1 settings (libsvtav1, GOP 2, CRF 30)."""
    tmp_path = video_path.with_name(f"{video_path.stem}.tmp{video_path.suffix}")
    with av.open(str(tmp_path), "w") as output:
        stream = output.add_stream(
            VIDEO_CODEC,
            fps,
            options=VIDEO_OPTIONS,
        )
        stream.pix_fmt = VIDEO_PIX_FMT
        stream.width = VIDEO_WIDTH
        stream.height = VIDEO_HEIGHT

        for rgb in rgb_frames:
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)

    if not tmp_path.is_file() or tmp_path.stat().st_size == 0:
        raise RuntimeError(f"AV1 encoder produced no data at {tmp_path}")
    tmp_path.replace(video_path)


def inspect_video(video_path):
    """Return decoded stream properties and frame count using the PyAV stack."""
    with av.open(str(video_path), "r") as container:
        stream = container.streams.video[0]
        codec_context = stream.codec_context
        codec = codec_context.codec
        properties = {
            "codec_id": codec.id,
            "codec_name": codec.name,
            "pix_fmt": codec_context.pix_fmt,
            "fps": float(stream.average_rate),
            "height": codec_context.height,
            "width": codec_context.width,
        }
        keyframes = []
        frame_count = 0
        for frame_count, frame in enumerate(container.decode(video=0), start=1):
            if frame.key_frame:
                keyframes.append(frame_count - 1)
        properties["frames"] = frame_count
        properties["keyframes"] = keyframes
    return properties


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", required=True,
                    help="RMBench data root holding <task>/<config>/data/episodeN.hdf5")
    ap.add_argument("--task", required=True)
    ap.add_argument("--config", default="demo_clean")
    ap.add_argument("--out", required=True, help="output LeRobot dataset root")
    ap.add_argument("--fps", type=int, default=15,
                    help="nominal LeRobot metadata rate; RMBench's save_freq=15 "
                         "(task_config/demo_clean.yml) subsamples a 250 Hz control "
                         "loop, so wall-clock capture is ~16.7 Hz and non-uniform — "
                         "15 only keeps parquet timestamps and video pts consistent")
    ap.add_argument("--instruction-type", default="seen", choices=("seen", "unseen"),
                    help="which list to use in RMBench instruction JSON files")
    prompt = ap.add_mutually_exclusive_group()
    prompt.add_argument("--instruction-file",
                        help="task-level RMBench JSON used only when per-episode instructions are absent")
    prompt.add_argument("--instruction",
                        help="literal full-episode instruction used for every episode")
    ap.add_argument("--limit", type=int, default=None, help="first N episodes only (smoke test)")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow replacing files in an already exported dataset")
    args = ap.parse_args()

    src = Path(args.raw_root) / args.task / args.config
    out = Path(args.out)
    eps = sorted((src / "data").glob("episode*.hdf5"),
                 key=lambda p: int(p.stem.replace("episode", "")))
    if args.limit:
        eps = eps[: args.limit]
    if not eps:
        raise SystemExit(f"no episodes under {src}/data")
    known_output_files = (
        list((out / "meta").glob("*"))
        + list((out / "data").rglob("*.parquet"))
        + list((out / "videos").rglob("*.mp4"))
    )
    if known_output_files and not args.overwrite:
        raise SystemExit(
            f"{out} already contains dataset files; pass --overwrite to replace "
            "its known export files, or use a fresh --out directory"
        )
    if args.overwrite:
        expected = {f"episode_{i:06d}" for i in range(len(eps))}
        stale = [
            p for p in (out / "data" / "chunk-000").glob("episode_*.parquet")
            if p.stem not in expected
        ]
        for cam in CAM_MAP:
            stale.extend(
                p for p in (out / "videos" / "chunk-000" / cam).glob("episode_*.mp4")
                if p.stem not in expected
            )
        if stale:
            raise SystemExit(
                "refusing to mix a new export with stale episode files. "
                f"Use a fresh --out directory. Examples: {stale[:6]}"
            )
    eval_reference = None
    reference_path = (
        Path(__file__).resolve().parents[2]
        / "third_party" / "RMBench" / "description" / "task_instruction"
        / f"{args.task}.json"
    )
    if reference_path.is_file():
        eval_reference = _first_instruction(reference_path, args.instruction_type)
    else:
        print(f"[raw2lerobot] WARNING: no eval reference at {reference_path}; "
              "train/eval instruction identity will not be checked", flush=True)

    print(
        f"[raw2lerobot] {args.task}: {len(eps)} episodes from {src}",
        flush=True,
    )

    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    for cam in CAM_MAP:
        (out / "videos" / "chunk-000" / cam).mkdir(parents=True, exist_ok=True)

    episodes, stats_rows, instructions = [], [], {}
    running_index = total_frames = 0

    for new_idx, h5_path in enumerate(eps):
        raw_idx = int(h5_path.stem.replace("episode", ""))
        instruction = episode_instruction(src, raw_idx, args, eval_reference)
        if not instruction:
            raise ValueError(f"episode {raw_idx}: empty instruction")
        if "@" in instruction:
            # RollingWAMProcessor splits task strings on '@' (zh@eng convention),
            # which would break the text-embed cache filename hash.
            raise ValueError(f"episode {raw_idx}: instruction contains '@': {instruction!r}")
        task_index = instructions.setdefault(instruction, len(instructions))

        with h5py.File(h5_path, "r") as f:
            joints = read_joint_vector(f, h5_path)  # (raw_T, 14)
            raw_T = joints.shape[0]
            if raw_T < 2:
                raise ValueError(f"{h5_path}: need at least 2 raw frames, got {raw_T}")
            if not np.isfinite(joints).all():
                raise ValueError(f"{h5_path}: joint_action contains NaN or infinity")

            # Match the released RoboTwin LeRobot alignment: row t observes
            # raw frame t and supervises the joint target at raw frame t+1.
            state = joints[:-1]
            action = joints[1:]
            T = raw_T - 1

            ep_img_stats = {}
            for cam, h5cam in CAM_MAP.items():
                jpegs = f[f"observation/{h5cam}/rgb"]
                if len(jpegs) != raw_T:
                    raise ValueError(
                        f"{h5_path}: {h5cam} has {len(jpegs)} frames, expected {raw_T}"
                    )
                vp = out / "videos" / "chunk-000" / cam / f"episode_{new_idx:06d}.mp4"
                sampled = []
                stat_ids = image_stat_indices(T)
                def rgb_frames():
                    for i in range(T):
                        img = decode_rgb(jpegs[i], f"{h5_path}:{h5cam}:{i}")
                        if img.shape[:2] != (VIDEO_HEIGHT, VIDEO_WIDTH):
                            raise ValueError(
                                f"{h5_path}:{h5cam}:{i} has resolution {img.shape[:2]}, "
                                f"expected {(VIDEO_HEIGHT, VIDEO_WIDTH)}"
                            )
                        # ``img`` is already legacy RGB (see decode_rgb), and
                        # PyAV receives it explicitly as rgb24: do not BGR-swap.
                        if i in stat_ids:
                            sampled.append(downsample_stat_image(img))
                        yield img

                print(
                    f"  ep{raw_idx:>3} -> {new_idx:06d}  {cam}: "
                    f"encoding {T} AV1 frames",
                    flush=True,
                )
                encode_av1_video(vp, rgb_frames(), args.fps)
                actual = inspect_video(vp)
                expected = {
                    "codec_id": AV1_CODEC_ID,
                    "pix_fmt": VIDEO_PIX_FMT,
                    "fps": float(args.fps),
                    "height": VIDEO_HEIGHT,
                    "width": VIDEO_WIDTH,
                    "frames": T,
                }
                comparable = {key: actual[key] for key in expected}
                if comparable != expected:
                    raise RuntimeError(
                        f"{vp}: encoded stream properties {actual}, expected {expected}"
                    )
                expected_keyframes = list(range(0, T, 2))
                if actual["keyframes"] != expected_keyframes:
                    raise RuntimeError(
                        f"{vp}: AV1 keyframes do not follow GOP 2; first actual="
                        f"{actual['keyframes'][:10]}, expected={expected_keyframes[:10]}"
                    )
                ep_img_stats[cam] = img_stats(sampled)

        timestamp = (np.arange(T) / args.fps).astype(np.float32)
        frame_index = np.arange(T, dtype=np.int64)

        table = pa.table({
            "observation.state": pa.FixedSizeListArray.from_arrays(
                pa.array(state.reshape(-1), type=pa.float32()), ACTION_DIM),
            "action": pa.FixedSizeListArray.from_arrays(
                pa.array(action.reshape(-1), type=pa.float32()), ACTION_DIM),
            "timestamp": pa.array(timestamp, type=pa.float32()),
            "frame_index": pa.array(frame_index),
            "episode_index": pa.array(np.full(T, new_idx, dtype=np.int64)),
            "index": pa.array(np.arange(running_index, running_index + T, dtype=np.int64)),
            "task_index": pa.array(np.full(T, task_index, dtype=np.int64)),
        })
        pq.write_table(table, out / "data" / "chunk-000" / f"episode_{new_idx:06d}.parquet")

        episodes.append({
            "episode_index": new_idx,
            "tasks": [instruction],
            "length": T,
            "source_episode": h5_path.name,
        })
        st = {"observation.state": feat_stats(state), "action": feat_stats(action),
              "timestamp": feat_stats(timestamp[:, None]),
              "frame_index": feat_stats(frame_index[:, None]),
              "episode_index": feat_stats(np.full((T, 1), new_idx)),
              "index": feat_stats(np.arange(running_index, running_index + T)[:, None]),
              "task_index": feat_stats(np.full((T, 1), task_index))}
        st.update(ep_img_stats)
        stats_rows.append({"episode_index": new_idx, "stats": st})

        running_index += T
        total_frames += T
        print(f"  ep{raw_idx:>3} -> {new_idx:06d}  complete frames={T}", flush=True)

    vids = {cam: {
        "dtype": "video", "shape": [3, VIDEO_HEIGHT, VIDEO_WIDTH],
        "names": ["channels", "height", "width"],
        "info": {"video.fps": args.fps, "video.height": VIDEO_HEIGHT,
                 "video.width": VIDEO_WIDTH, "video.channels": 3,
                 "video.codec": "av1", "video.pix_fmt": VIDEO_PIX_FMT,
                 "video.is_depth_map": False, "has_audio": False},
    } for cam in CAM_MAP}
    info = {
        "codebase_version": "v2.1", "robot_type": "aloha", "fps": args.fps,
        "total_episodes": len(episodes), "total_frames": total_frames,
        "total_tasks": len(instructions), "total_videos": len(episodes) * len(CAM_MAP),
        "total_chunks": 1, "chunks_size": 1000,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [ACTION_DIM], "names": [ACTION_NAMES]},
            "action": {"dtype": "float32", "shape": [ACTION_DIM], "names": [ACTION_NAMES]},
            **vids,
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    with open(out / "meta" / "episodes.jsonl", "w") as fh:
        for e in episodes:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(out / "meta" / "episodes_stats.jsonl", "w") as fh:
        for s in stats_rows:
            fh.write(json.dumps(s) + "\n")
    with open(out / "meta" / "tasks.jsonl", "w") as fh:
        for text, idx in sorted(instructions.items(), key=lambda kv: kv[1]):
            fh.write(json.dumps({"task_index": idx, "task": text}, ensure_ascii=False) + "\n")

    print(f"[raw2lerobot] wrote {len(episodes)} episodes / {total_frames} frames -> {out}")


if __name__ == "__main__":
    main()
