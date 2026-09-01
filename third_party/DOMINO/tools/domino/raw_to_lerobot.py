"""Convert one raw DOMINO task/level into RollingWAM's LeRobot v2.1 layout.

The conversion preserves DOMINO's native three D435 cameras and 14-D absolute
joint positions. Row ``t`` contains observation/state from raw frame ``t`` and
the action target from raw frame ``t+1``; the final raw frame is dropped.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

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
VIDEO_HEIGHT = 240
VIDEO_WIDTH = 320
VIDEO_CODEC = "libsvtav1"
VIDEO_PIX_FMT = "yuv420p"
VIDEO_OPTIONS = {"g": "2", "crf": "30"}
AV1_CODEC_ID = av.codec.Codec("av1", "r").id
ACTION_NAMES = (
    [f"left_joint{i}" for i in range(1, 7)]
    + ["left_gripper"]
    + [f"right_joint{i}" for i in range(1, 7)]
    + ["right_gripper"]
)
ACTION_DIM = len(ACTION_NAMES)


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _first_instruction(path: Path, instruction_type: str) -> str:
    choices = _read_json(path).get(instruction_type)
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], str):
        raise ValueError(
            f"{path}: expected a non-empty string list at {instruction_type!r}"
        )
    instruction = choices[0].strip()
    if not instruction:
        raise ValueError(f"{path}: first {instruction_type!r} instruction is empty")
    return instruction


def _instruction_patterns(path: Path, instruction_type: str) -> list[re.Pattern[str]]:
    templates = _read_json(path).get(instruction_type, [])
    patterns: list[re.Pattern[str]] = []
    for template in templates:
        escaped = re.escape(str(template).strip())
        wildcarded = re.sub(r"\\\{[A-Za-z]\\\}", ".+?", escaped)
        patterns.append(re.compile(f"^{wildcarded}$", re.IGNORECASE))
    if not patterns:
        raise ValueError(f"{path}: no {instruction_type!r} instruction templates")
    return patterns


def _episode_instruction(
    source: Path,
    raw_index: int,
    instruction_type: str,
    literal_instruction: str | None,
) -> str:
    if literal_instruction is not None:
        instruction = literal_instruction.strip()
    else:
        instruction_path = source / "instructions" / f"episode{raw_index}.json"
        if not instruction_path.is_file():
            raise FileNotFoundError(
                f"missing {instruction_path}; DOMINO collection should generate "
                "episode-specific instructions after rendering"
            )
        instruction = _first_instruction(instruction_path, instruction_type)
    if "@" in instruction:
        raise ValueError(
            f"episode {raw_index}: instruction contains '@', which RollingWAM reserves "
            f"for bilingual prompt splitting: {instruction!r}"
        )
    return instruction


def _decode_rgb(encoded: np.ndarray, source: str) -> np.ndarray:
    """Decode DOMINO's JPEG bytes without swapping its legacy RGB channel order."""
    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode RGB bytes from {source}")
    return image


def _read_joint_vector(file: h5py.File, h5_path: Path) -> np.ndarray:
    vector = np.asarray(file["joint_action/vector"], dtype=np.float32)
    if vector.ndim != 2 or vector.shape[1] != ACTION_DIM:
        raise ValueError(f"{h5_path}: joint_action/vector has shape {vector.shape}")

    parts = np.concatenate(
        [
            np.asarray(file["joint_action/left_arm"], dtype=np.float32),
            np.asarray(file["joint_action/left_gripper"], dtype=np.float32)[:, None],
            np.asarray(file["joint_action/right_arm"], dtype=np.float32),
            np.asarray(file["joint_action/right_gripper"], dtype=np.float32)[:, None],
        ],
        axis=1,
    )
    if parts.shape != vector.shape or not np.allclose(parts, vector, atol=1e-6):
        raise ValueError(
            f"{h5_path}: joint_action/vector does not equal "
            "[left_arm, left_gripper, right_arm, right_gripper]"
        )
    if not np.isfinite(vector).all():
        raise ValueError(f"{h5_path}: joint_action contains NaN or infinity")
    return vector


def _feature_stats(array: np.ndarray) -> dict:
    values = np.asarray(array, dtype=np.float64).reshape(len(array), -1)
    return {
        "min": values.min(0).tolist(),
        "max": values.max(0).tolist(),
        "mean": values.mean(0).tolist(),
        "std": values.std(0).tolist(),
        "count": [len(values)],
    }


def _image_stats(frames: list[np.ndarray]) -> dict:
    values = np.stack(frames).astype(np.float64) / 255.0

    def per_channel(function) -> list[list[list[float]]]:
        return [[[float(value)]] for value in function(values, axis=(0, 1, 2))]

    return {
        "min": per_channel(np.min),
        "max": per_channel(np.max),
        "mean": per_channel(np.mean),
        "std": per_channel(np.std),
        "count": [len(frames)],
    }


def _image_stat_indices(length: int) -> set[int]:
    minimum = min(100, length)
    count = max(minimum, min(int(length**0.75), 10_000))
    return set(np.round(np.linspace(0, length - 1, count)).astype(int).tolist())


def _downsample_stat_image(
    image: np.ndarray,
    target_size: int = 150,
    max_size_threshold: int = 300,
) -> np.ndarray:
    height, width = image.shape[:2]
    if max(width, height) < max_size_threshold:
        return image
    factor = int(width / target_size) if width > height else int(height / target_size)
    return image[::factor, ::factor]


def _encode_video(video_path: Path, frames: Iterable[np.ndarray], fps: int) -> None:
    temporary = video_path.with_name(f"{video_path.stem}.tmp{video_path.suffix}")
    with av.open(str(temporary), "w") as output:
        stream = output.add_stream(VIDEO_CODEC, fps, options=VIDEO_OPTIONS)
        stream.pix_fmt = VIDEO_PIX_FMT
        stream.width = VIDEO_WIDTH
        stream.height = VIDEO_HEIGHT
        for rgb in frames:
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise RuntimeError(f"AV1 encoder produced no data at {temporary}")
    temporary.replace(video_path)


def _inspect_video(video_path: Path) -> dict:
    with av.open(str(video_path), "r") as container:
        stream = container.streams.video[0]
        codec_context = stream.codec_context
        properties = {
            "codec_id": codec_context.codec.id,
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


def _episode_files(source: Path, episode_count: int) -> list[Path]:
    episodes = sorted(
        (source / "data").glob("episode*.hdf5"),
        key=lambda path: int(path.stem.removeprefix("episode")),
    )
    if len(episodes) < episode_count:
        raise ValueError(
            f"{source}: requested {episode_count} episodes but found only {len(episodes)}"
        )
    selected = episodes[:episode_count]
    expected_indices = list(range(episode_count))
    actual_indices = [int(path.stem.removeprefix("episode")) for path in selected]
    if actual_indices != expected_indices:
        raise ValueError(
            f"{source}: first {episode_count} episode indices must be contiguous 0.."
            f"{episode_count - 1}, got {actual_indices[:10]}"
        )
    return selected


def _prepare_output(output: Path, episode_count: int, overwrite: bool) -> None:
    known_files = (
        list((output / "meta").glob("*"))
        + list((output / "data").rglob("*.parquet"))
        + list((output / "videos").rglob("*.mp4"))
    )
    if known_files and not overwrite:
        raise FileExistsError(
            f"{output} already contains a converted dataset; pass --overwrite to replace it"
        )
    if overwrite:
        expected = {f"episode_{index:06d}" for index in range(episode_count)}
        stale = [
            path
            for path in (output / "data" / "chunk-000").glob("episode_*.parquet")
            if path.stem not in expected
        ]
        for camera in CAM_MAP:
            stale.extend(
                path
                for path in (output / "videos" / "chunk-000" / camera).glob(
                    "episode_*.mp4"
                )
                if path.stem not in expected
            )
        if stale:
            raise RuntimeError(
                "refusing to mix the selected export with stale episodes; use a fresh "
                f"output directory. Examples: {stale[:5]}"
            )

    (output / "meta").mkdir(parents=True, exist_ok=True)
    (output / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    for camera in CAM_MAP:
        (output / "videos" / "chunk-000" / camera).mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dynamic-level", required=True, type=int, choices=(1, 2, 3))
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument(
        "--instruction-type", default="seen", choices=("seen", "unseen")
    )
    parser.add_argument("--instruction")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.episodes <= 0:
        raise ValueError(f"--episodes must be positive, got {args.episodes}")
    if args.fps <= 0:
        raise ValueError(f"--fps must be positive, got {args.fps}")

    level_match = re.search(r"level([123])", args.config)
    if level_match is not None and int(level_match.group(1)) != args.dynamic_level:
        raise ValueError(
            f"source config {args.config!r} names level {level_match.group(1)}, but "
            f"--dynamic-level={args.dynamic_level}"
        )

    source = Path(args.raw_root).expanduser() / args.task / args.config
    output = Path(args.out).expanduser()
    episodes = _episode_files(source, args.episodes)
    _prepare_output(output, args.episodes, args.overwrite)

    domino_root = Path(__file__).resolve().parents[2]
    instruction_patterns: list[re.Pattern[str]] | None = None
    if args.instruction is None:
        task_instruction_path = (
            domino_root / "description" / "task_instruction" / f"{args.task}.json"
        )
        if not task_instruction_path.is_file():
            raise FileNotFoundError(
                f"DOMINO task instruction file not found: {task_instruction_path}"
            )
        instruction_patterns = _instruction_patterns(
            task_instruction_path, args.instruction_type
        )

    episode_rows: list[dict] = []
    episode_stats: list[dict] = []
    instructions: dict[str, int] = {}
    running_index = 0
    total_frames = 0

    print(
        f"[domino2lerobot] task={args.task} level={args.dynamic_level} "
        f"episodes={len(episodes)} source={source}",
        flush=True,
    )

    for new_index, h5_path in enumerate(episodes):
        raw_index = int(h5_path.stem.removeprefix("episode"))
        instruction = _episode_instruction(
            source,
            raw_index,
            args.instruction_type,
            args.instruction,
        )
        if instruction_patterns is not None and not any(
            pattern.fullmatch(instruction) for pattern in instruction_patterns
        ):
            raise ValueError(
                f"episode {raw_index}: instruction does not match any DOMINO "
                f"{args.instruction_type!r} template for {args.task}: {instruction!r}"
            )
        task_index = instructions.setdefault(instruction, len(instructions))

        with h5py.File(h5_path, "r") as file:
            joints = _read_joint_vector(file, h5_path)
            raw_length = joints.shape[0]
            if raw_length < 2:
                raise ValueError(f"{h5_path}: need at least two frames")
            state = joints[:-1]
            action = joints[1:]
            length = raw_length - 1

            image_stats: dict[str, dict] = {}
            for camera, raw_camera in CAM_MAP.items():
                encoded_frames = file[f"observation/{raw_camera}/rgb"]
                if len(encoded_frames) != raw_length:
                    raise ValueError(
                        f"{h5_path}: {raw_camera} has {len(encoded_frames)} frames, "
                        f"expected {raw_length}"
                    )
                video_path = (
                    output
                    / "videos"
                    / "chunk-000"
                    / camera
                    / f"episode_{new_index:06d}.mp4"
                )
                sampled: list[np.ndarray] = []
                sample_indices = _image_stat_indices(length)

                def frames() -> Iterable[np.ndarray]:
                    for frame_index in range(length):
                        image = _decode_rgb(
                            encoded_frames[frame_index],
                            f"{h5_path}:{raw_camera}:{frame_index}",
                        )
                        if image.shape[:2] != (VIDEO_HEIGHT, VIDEO_WIDTH):
                            raise ValueError(
                                f"{h5_path}:{raw_camera}:{frame_index} has resolution "
                                f"{image.shape[:2]}, expected {(VIDEO_HEIGHT, VIDEO_WIDTH)}"
                            )
                        if frame_index in sample_indices:
                            sampled.append(_downsample_stat_image(image))
                        yield image

                print(
                    f"  episode {raw_index:>3} -> {new_index:06d} {camera}: "
                    f"encoding {length} frames",
                    flush=True,
                )
                _encode_video(video_path, frames(), args.fps)
                actual = _inspect_video(video_path)
                expected = {
                    "codec_id": AV1_CODEC_ID,
                    "pix_fmt": VIDEO_PIX_FMT,
                    "fps": float(args.fps),
                    "height": VIDEO_HEIGHT,
                    "width": VIDEO_WIDTH,
                    "frames": length,
                }
                comparable = {key: actual[key] for key in expected}
                if comparable != expected:
                    raise RuntimeError(
                        f"{video_path}: encoded stream properties {actual}, "
                        f"expected {expected}"
                    )
                expected_keyframes = list(range(0, length, 2))
                if actual["keyframes"] != expected_keyframes:
                    raise RuntimeError(
                        f"{video_path}: AV1 keyframes do not follow GOP 2; first "
                        f"actual={actual['keyframes'][:10]}, "
                        f"expected={expected_keyframes[:10]}"
                    )
                image_stats[camera] = _image_stats(sampled)

        timestamp = (np.arange(length) / args.fps).astype(np.float32)
        frame_index = np.arange(length, dtype=np.int64)
        parquet = pa.table(
            {
                "observation.state": pa.FixedSizeListArray.from_arrays(
                    pa.array(state.reshape(-1), type=pa.float32()), ACTION_DIM
                ),
                "action": pa.FixedSizeListArray.from_arrays(
                    pa.array(action.reshape(-1), type=pa.float32()), ACTION_DIM
                ),
                "timestamp": pa.array(timestamp, type=pa.float32()),
                "frame_index": pa.array(frame_index),
                "episode_index": pa.array(
                    np.full(length, new_index, dtype=np.int64)
                ),
                "index": pa.array(
                    np.arange(running_index, running_index + length, dtype=np.int64)
                ),
                "task_index": pa.array(
                    np.full(length, task_index, dtype=np.int64)
                ),
            }
        )
        pq.write_table(
            parquet,
            output / "data" / "chunk-000" / f"episode_{new_index:06d}.parquet",
        )

        episode_rows.append(
            {
                "episode_index": new_index,
                "tasks": [instruction],
                "length": length,
                "source_episode": h5_path.name,
                "source_task": args.task,
                "source_config": args.config,
                "dynamic_level": args.dynamic_level,
            }
        )
        stats = {
            "observation.state": _feature_stats(state),
            "action": _feature_stats(action),
            "timestamp": _feature_stats(timestamp[:, None]),
            "frame_index": _feature_stats(frame_index[:, None]),
            "episode_index": _feature_stats(np.full((length, 1), new_index)),
            "index": _feature_stats(
                np.arange(running_index, running_index + length)[:, None]
            ),
            "task_index": _feature_stats(np.full((length, 1), task_index)),
        }
        stats.update(image_stats)
        episode_stats.append({"episode_index": new_index, "stats": stats})
        running_index += length
        total_frames += length

    video_features = {
        camera: {
            "dtype": "video",
            "shape": [3, VIDEO_HEIGHT, VIDEO_WIDTH],
            "names": ["channels", "height", "width"],
            "info": {
                "video.fps": args.fps,
                "video.height": VIDEO_HEIGHT,
                "video.width": VIDEO_WIDTH,
                "video.channels": 3,
                "video.codec": "av1",
                "video.pix_fmt": VIDEO_PIX_FMT,
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
        for camera in CAM_MAP
    }
    info = {
        "codebase_version": "v2.1",
        "robot_type": "aloha",
        "fps": args.fps,
        "total_episodes": len(episode_rows),
        "total_frames": total_frames,
        "total_tasks": len(instructions),
        "total_videos": len(episode_rows) * len(CAM_MAP),
        "total_chunks": 1,
        "chunks_size": 1000,
        "splits": {"train": f"0:{len(episode_rows)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [ACTION_DIM],
                "names": [ACTION_NAMES],
            },
            "action": {
                "dtype": "float32",
                "shape": [ACTION_DIM],
                "names": [ACTION_NAMES],
            },
            **video_features,
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }

    (output / "meta" / "info.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )
    with (output / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as file:
        for row in episode_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output / "meta" / "episodes_stats.jsonl").open(
        "w", encoding="utf-8"
    ) as file:
        for row in episode_stats:
            file.write(json.dumps(row) + "\n")
    with (output / "meta" / "tasks.jsonl").open("w", encoding="utf-8") as file:
        for instruction, task_index in sorted(
            instructions.items(), key=lambda item: item[1]
        ):
            file.write(
                json.dumps(
                    {"task_index": task_index, "task": instruction},
                    ensure_ascii=False,
                )
                + "\n"
            )

    if len(episode_rows) != args.episodes:
        raise RuntimeError(
            f"converted {len(episode_rows)} episodes, expected {args.episodes}"
        )
    print(
        f"[domino2lerobot] wrote {len(episode_rows)} episodes / {total_frames} "
        f"frames -> {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
