#!/usr/bin/env bash
# Convert selected raw DOMINO demonstrations into RollingWAM LeRobot datasets.
#
# Defaults to 50 demonstrations for each of:
#   beat_block_hammer:2 place_bread_basket:1
# (100 demonstrations total).
#
# Usage:
#   bash scripts/domino/prepare_domino_data.sh
#   bash scripts/domino/prepare_domino_data.sh beat_block_hammer:2
#
# Environment overrides:
#   DOMINO_TASK_SPECS          space/comma-separated task:level entries
#   DOMINO_RAW_ROOT            default: /datasets/DOMINO/dataset
#   DOMINO_DATA_ROOT           default: /datasets/DOMINO/rollingwam-domino
#   DOMINO_RAW_CONFIG_TEMPLATE default: aloha-agilex_clean_level{level}
#   DOMINO_EXPECT_EPISODES     default: 50 per entry
#   DOMINO_OVERWRITE           true/false; default: false

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DEFAULT_SPECS="beat_block_hammer:2 place_bread_basket:1"
RAW_ROOT="${DOMINO_RAW_ROOT:-/datasets/DOMINO/dataset}"
DATA_ROOT="${DOMINO_DATA_ROOT:-/datasets/DOMINO/rollingwam-domino}"
RAW_CONFIG_TEMPLATE="${DOMINO_RAW_CONFIG_TEMPLATE:-aloha-agilex_clean_level{level}}"
EXPECT_EPISODES="${DOMINO_EXPECT_EPISODES:-50}"

if [[ ! "$EXPECT_EPISODES" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: DOMINO_EXPECT_EPISODES must be a positive integer, got: $EXPECT_EPISODES" >&2
  exit 2
fi

if [[ $# -gt 0 ]]; then
  TASK_SPECS=("$@")
else
  RAW_SPECS="${DOMINO_TASK_SPECS:-$DEFAULT_SPECS}"
  RAW_SPECS="${RAW_SPECS//,/ }"
  read -r -a TASK_SPECS <<< "$RAW_SPECS"
fi

if [[ ${#TASK_SPECS[@]} -eq 0 ]]; then
  echo "Error: no DOMINO task:level entries were selected." >&2
  exit 2
fi

OVERWRITE=false
case "${DOMINO_OVERWRITE:-false}" in
  true|TRUE|True|1|yes|YES) OVERWRITE=true ;;
  false|FALSE|False|0|no|NO) ;;
  *)
    echo "Error: DOMINO_OVERWRITE must be true or false, got: ${DOMINO_OVERWRITE}" >&2
    exit 2
    ;;
esac

validate_spec() {
  local spec="$1"
  if [[ ! "$spec" =~ ^([A-Za-z0-9_]+):([123])$ ]]; then
    echo "Error: invalid task spec '$spec'; expected task_name:1, task_name:2, or task_name:3." >&2
    exit 2
  fi
  TASK_NAME="${BASH_REMATCH[1]}"
  DYNAMIC_LEVEL="${BASH_REMATCH[2]}"
}

read_episode_count() {
  local info_path="$1"
  PYTHONDONTWRITEBYTECODE=1 python -B - "$info_path" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as file:
    value = json.load(file).get("total_episodes")
if not isinstance(value, int):
    raise SystemExit(f"missing integer total_episodes in {sys.argv[1]}")
print(value)
PY
}

SEEN_SPECS=" "
for spec in "${TASK_SPECS[@]}"; do
  validate_spec "$spec"
  if [[ "$SEEN_SPECS" == *" $spec "* ]]; then
    echo "Error: duplicate DOMINO task spec: $spec" >&2
    exit 2
  fi
  SEEN_SPECS+="$spec "

  if [[ ! -f "third_party/DOMINO/envs/${TASK_NAME}.py" ]]; then
    echo "Error: DOMINO task does not exist: ${TASK_NAME}" >&2
    exit 2
  fi
done

TOTAL_EPISODES=0
for spec in "${TASK_SPECS[@]}"; do
  validate_spec "$spec"
  RAW_CONFIG="${RAW_CONFIG_TEMPLATE//\{level\}/${DYNAMIC_LEVEL}}"
  SOURCE_DIR="${RAW_ROOT}/${TASK_NAME}/${RAW_CONFIG}"

  # Official released data uses aloha-agilex_clean_levelN. Locally collected
  # data produced by this integration uses demo_clean_dynamic_levelN.
  if [[ ! -d "$SOURCE_DIR" && -z "${DOMINO_RAW_CONFIG_TEMPLATE+x}" ]]; then
    FALLBACK_CONFIG="demo_clean_dynamic_level${DYNAMIC_LEVEL}"
    FALLBACK_DIR="${RAW_ROOT}/${TASK_NAME}/${FALLBACK_CONFIG}"
    if [[ -d "$FALLBACK_DIR" ]]; then
      echo "Using local collection config ${FALLBACK_CONFIG} for ${TASK_NAME}."
      RAW_CONFIG="$FALLBACK_CONFIG"
      SOURCE_DIR="$FALLBACK_DIR"
    fi
  fi

  if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "Error: raw DOMINO data not found: ${SOURCE_DIR}" >&2
    exit 2
  fi

  OUTPUT_DIR="${DATA_ROOT}/${TASK_NAME}_level${DYNAMIC_LEVEL}"
  echo "=== ${TASK_NAME}, dynamic level ${DYNAMIC_LEVEL} (${EXPECT_EPISODES} demos) ==="
  CONVERT_CMD=(python -B \
    third_party/DOMINO/tools/domino/raw_to_lerobot.py \
    --raw-root "$RAW_ROOT" \
    --task "$TASK_NAME" \
    --config "$RAW_CONFIG" \
    --dynamic-level "$DYNAMIC_LEVEL" \
    --out "$OUTPUT_DIR" \
    --episodes "$EXPECT_EPISODES")
  if [[ "$OVERWRITE" == true ]]; then
    CONVERT_CMD+=(--overwrite)
  fi
  PYTHONDONTWRITEBYTECODE=1 "${CONVERT_CMD[@]}"

  ACTUAL_EPISODES="$(read_episode_count "${OUTPUT_DIR}/meta/info.json")"
  if [[ "$ACTUAL_EPISODES" -ne "$EXPECT_EPISODES" ]]; then
    echo "Error: ${OUTPUT_DIR} has ${ACTUAL_EPISODES} episodes; expected ${EXPECT_EPISODES}." >&2
    exit 1
  fi
  TOTAL_EPISODES=$((TOTAL_EPISODES + ACTUAL_EPISODES))
done

EXPECTED_TOTAL=$((EXPECT_EPISODES * ${#TASK_SPECS[@]}))
if [[ "$TOTAL_EPISODES" -ne "$EXPECTED_TOTAL" ]]; then
  echo "Error: converted ${TOTAL_EPISODES} demonstrations; expected ${EXPECTED_TOTAL}." >&2
  exit 1
fi

echo "Prepared ${TOTAL_EPISODES} DOMINO demonstrations across ${#TASK_SPECS[@]} task-level datasets under ${DATA_ROOT}."
