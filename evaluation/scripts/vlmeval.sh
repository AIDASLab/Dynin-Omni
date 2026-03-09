#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLMKIT_DIR="${ROOT_DIR}/VLMEvalKit"

GPU_SPEC=""
WORK_DIR="./outputs"
MASTER_PORT="${MASTER_PORT:-54321}"
MODE="all"
API_NPROC="${API_NPROC:-4}"
JUDGE=""
RETRY=""
JUDGE_ARGS=""
USE_VLLM=0
VERBOSE=0
IGNORE_FAILED=0
REUSE=0
REUSE_AUX=1

MODEL_LIST=()
DATA_LIST=()
EXTRA_RUN_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/vlmeval.sh --gpu <ids> --data <dataset[,dataset...]> [--model <model[,model...]>] [options] [-- <extra run.py args>]

Examples:
  bash scripts/vlmeval.sh --gpu 0 --data MMStar --model DyninOmni
  bash scripts/vlmeval.sh --gpu 0,1,2,3,4,5,6,7 --data MMMU_DEV_VAL,POPE --model DyninOmni
  bash scripts/vlmeval.sh --gpu 0 --data all --model all

Required:
  --gpu        GPU ids, comma separated (e.g. 0 or 0,1,2,3)
  --data       VLMEvalKit dataset name(s), comma separated or repeated; supports "all"

Optional:
  --model      VLMEvalKit model name(s), comma separated or repeated; default: DyninOmni; supports "all"
  --work-dir   Output root passed to run.py (default: ./outputs)
  --mode       run.py mode: all|infer (default: all)
  --master-port  torchrun master port (default: 54321)
  --api-nproc  API parallelism for run.py (default: 4)
  --judge      Explicit judge model for run.py
  --judge-args JSON string for run.py --judge-args
  --retry      Retry count for run.py
  --verbose    Enable verbose logging
  --use-vllm   Forward to run.py --use-vllm
  --ignore     Forward to run.py --ignore
  --reuse      Forward to run.py --reuse
  --reuse-aux <0|1>  Forward to run.py --reuse-aux (default: 1)
  -h, --help   Show help
EOF
}

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

append_csv_items() {
  local raw="$1"
  local target="$2"
  local part item
  IFS=',' read -r -a part <<< "$raw"
  for item in "${part[@]}"; do
    item="$(trim "$item")"
    [[ -z "$item" ]] && continue
    if [[ "$target" == "model" ]]; then
      MODEL_LIST+=("$item")
    else
      DATA_LIST+=("$item")
    fi
  done
}

load_all_models() {
  mapfile -t MODEL_LIST < <(
    cd "${VLMKIT_DIR}" && python - <<'PY'
from vlmeval.config import supported_VLM
for name in sorted(supported_VLM.keys()):
    print(name)
PY
  )
}

load_all_datasets() {
  mapfile -t DATA_LIST < <(
    cd "${VLMKIT_DIR}" && python - <<'PY'
from vlmeval.dataset import SUPPORTED_DATASETS
from vlmeval.dataset.video_dataset_config import supported_video_datasets

names = set()
if isinstance(SUPPORTED_DATASETS, dict):
    names.update(SUPPORTED_DATASETS.keys())
else:
    names.update(SUPPORTED_DATASETS)
names.update(supported_video_datasets.keys())

for name in sorted(names):
    print(name)
PY
  )
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU_SPEC="$2"
      shift 2
      ;;
    --data)
      append_csv_items "$2" "data"
      shift 2
      ;;
    --model)
      append_csv_items "$2" "model"
      shift 2
      ;;
    --work-dir)
      WORK_DIR="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --master-port)
      MASTER_PORT="$2"
      shift 2
      ;;
    --api-nproc)
      API_NPROC="$2"
      shift 2
      ;;
    --judge)
      JUDGE="$2"
      shift 2
      ;;
    --judge-args)
      JUDGE_ARGS="$2"
      shift 2
      ;;
    --retry)
      RETRY="$2"
      shift 2
      ;;
    --verbose)
      VERBOSE=1
      shift 1
      ;;
    --use-vllm)
      USE_VLLM=1
      shift 1
      ;;
    --ignore)
      IGNORE_FAILED=1
      shift 1
      ;;
    --reuse)
      REUSE=1
      shift 1
      ;;
    --reuse-aux)
      REUSE_AUX="$2"
      shift 2
      ;;
    --)
      shift
      EXTRA_RUN_ARGS=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${GPU_SPEC}" ]]; then
  echo "--gpu is required." >&2
  usage
  exit 1
fi

if [[ ${#DATA_LIST[@]} -eq 0 ]]; then
  echo "--data is required." >&2
  usage
  exit 1
fi

if [[ ${#MODEL_LIST[@]} -eq 0 ]]; then
  MODEL_LIST=("DyninOmni")
fi

if [[ " ${MODEL_LIST[*]} " == *" all "* ]]; then
  load_all_models
fi

if [[ " ${DATA_LIST[*]} " == *" all "* ]]; then
  load_all_datasets
fi

if [[ ${#MODEL_LIST[@]} -eq 0 ]]; then
  echo "No models resolved." >&2
  exit 1
fi

if [[ ${#DATA_LIST[@]} -eq 0 ]]; then
  echo "No datasets resolved." >&2
  exit 1
fi

if [[ ! -d "${VLMKIT_DIR}" ]]; then
  echo "VLMEvalKit directory not found: ${VLMKIT_DIR}" >&2
  exit 1
fi

IFS=',' read -r -a GPU_IDS <<< "${GPU_SPEC}"
GPU_COUNT="${#GPU_IDS[@]}"
if [[ "${GPU_COUNT}" -lt 1 ]]; then
  echo "Invalid --gpu value: ${GPU_SPEC}" >&2
  exit 1
fi

RUN_ARGS=(
  run.py
  --data "${DATA_LIST[@]}"
  --model "${MODEL_LIST[@]}"
  --work-dir "${WORK_DIR}"
  --mode "${MODE}"
  --api-nproc "${API_NPROC}"
  --reuse-aux "${REUSE_AUX}"
)

if [[ -n "${JUDGE}" ]]; then
  RUN_ARGS+=(--judge "${JUDGE}")
fi
if [[ -n "${JUDGE_ARGS}" ]]; then
  RUN_ARGS+=(--judge-args "${JUDGE_ARGS}")
fi
if [[ -n "${RETRY}" ]]; then
  RUN_ARGS+=(--retry "${RETRY}")
fi
if [[ "${VERBOSE}" -eq 1 ]]; then
  RUN_ARGS+=(--verbose)
fi
if [[ "${USE_VLLM}" -eq 1 ]]; then
  RUN_ARGS+=(--use-vllm)
fi
if [[ "${IGNORE_FAILED}" -eq 1 ]]; then
  RUN_ARGS+=(--ignore)
fi
if [[ "${REUSE}" -eq 1 ]]; then
  RUN_ARGS+=(--reuse)
fi
if [[ ${#EXTRA_RUN_ARGS[@]} -gt 0 ]]; then
  RUN_ARGS+=("${EXTRA_RUN_ARGS[@]}")
fi

echo "[vlmeval] cwd=${VLMKIT_DIR}"
echo "[vlmeval] gpu=${GPU_SPEC} (nproc=${GPU_COUNT})"
echo "[vlmeval] models=${#MODEL_LIST[@]} datasets=${#DATA_LIST[@]}"

cd "${VLMKIT_DIR}"

if [[ "${GPU_COUNT}" -eq 1 ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_SPEC}" python "${RUN_ARGS[@]}"
else
  CUDA_VISIBLE_DEVICES="${GPU_SPEC}" torchrun \
    --nproc-per-node="${GPU_COUNT}" \
    --master-port="${MASTER_PORT}" \
    "${RUN_ARGS[@]}"
fi
