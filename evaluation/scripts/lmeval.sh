#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LM_DIR="${ROOT_DIR}/lm"

ACCELERATE_BIN="accelerate"
CONFIG_FILE=""

TASK="gsm8k"
MODEL_PATH=""
MODEL_CONFIG_PATH=""
NUM_FEWSHOT=0
GEN_LENGTH=1024
STEPS=""
BLOCK_LENGTH=16
SHOW_SPEED="True"

GPU_SPEC="${CUDA_VISIBLE_DEVICES:-0}"
NUM_PROCESSES=""

OUTPUT_DIR="${ROOT_DIR}/results/lm"
OUTPUT_PATH=""
LIMIT=0
LOG_SAMPLES=1

HF_HOME_PATH=""
HF_DATASETS_CACHE_PATH=""
TRANSFORMERS_CACHE_PATH=""
OFFLINE=0

EXTRA_MODEL_ARGS=()

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/lmeval.sh --model-path <path_or_hf_id> [options]

Description:
  Run Dynin-Omni text LM evaluation via lm-eval-harness (evaluation/lm/eval_dynin_omni.py).
  This script only targets Dynin-Omni LM evaluation and does not run LLaDA baseline scripts.

Required:
  --model-path <path_or_hf_id>   Model path or Hugging Face model id.

Main options:
  --model-config-path <path>      Optional explicit config.json path passed to eval_dynin_omni.py
  --task <name>                  lm-eval task name (default: gsm8k)
  --num-fewshot <n>              Number of few-shot examples (default: 0)
  --gen-length <n>               Generation length (default: 1024)
  --steps <n>                    Sampling steps (default: same as --gen-length)
  --block-length <n>             Block length (default: 16)
  --show-speed <True|False>      show_speed model arg (default: True)

Runtime options:
  --gpu <ids>                    CUDA_VISIBLE_DEVICES (default: from env or 0)
  --num-processes <n>            accelerate --num_processes (default: inferred)
  --accelerate <path>            accelerate executable (default: accelerate)
  --config-file <path>           accelerate config file

Output options:
  --output-dir <path>            Base output directory (default: evaluation/results/lm)
  --output-path <path>           Full lm-eval output path override
  --limit <n>                    lm-eval --limit (default: 0, disabled)
  --no-log-samples               Disable --log_samples

Cache/env options:
  --hf-home <path>               Export HF_HOME
  --hf-datasets-cache <path>     Export HF_DATASETS_CACHE
  --transformers-cache <path>    Export TRANSFORMERS_CACHE
  --offline                      Export HF_DATASETS_OFFLINE=1, TRANSFORMERS_OFFLINE=1,
                                 HF_HUB_DISABLE_TELEMETRY=1

Advanced model_args:
  --param <key=value>            Append extra --model_args entry (repeatable)
                                 Example: --param use_fastdllm_v1=True
                                 Example: --param factor=3.0
                                 Example: --param shard_id=0 --param num_shards=5 --param save_dir=/tmp/shard0

Examples:
  bash scripts/lmeval.sh \
    --model-path /path/to/unwrapped_model

  bash scripts/lmeval.sh \
    --model-path /path/to/unwrapped_model \
    --task hendrycks_math500 \
    --gen-length 1024 --block-length 16 \
    --output-dir ./results/lm

  bash scripts/lmeval.sh \
    --model-path /path/to/unwrapped_model \
    --param use_fastdllm_v1=True --param factor=3.0
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)
      TASK="$2"
      shift 2
      ;;
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --model-config-path)
      MODEL_CONFIG_PATH="$2"
      shift 2
      ;;
    --num-fewshot)
      NUM_FEWSHOT="$2"
      shift 2
      ;;
    --gen-length)
      GEN_LENGTH="$2"
      shift 2
      ;;
    --steps)
      STEPS="$2"
      shift 2
      ;;
    --block-length)
      BLOCK_LENGTH="$2"
      shift 2
      ;;
    --show-speed)
      SHOW_SPEED="$2"
      shift 2
      ;;
    --gpu)
      GPU_SPEC="$2"
      shift 2
      ;;
    --num-processes)
      NUM_PROCESSES="$2"
      shift 2
      ;;
    --accelerate)
      ACCELERATE_BIN="$2"
      shift 2
      ;;
    --config-file)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --output-path)
      OUTPUT_PATH="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --no-log-samples)
      LOG_SAMPLES=0
      shift 1
      ;;
    --hf-home)
      HF_HOME_PATH="$2"
      shift 2
      ;;
    --hf-datasets-cache)
      HF_DATASETS_CACHE_PATH="$2"
      shift 2
      ;;
    --transformers-cache)
      TRANSFORMERS_CACHE_PATH="$2"
      shift 2
      ;;
    --offline)
      OFFLINE=1
      shift 1
      ;;
    --param)
      EXTRA_MODEL_ARGS+=("$2")
      shift 2
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

if [[ -z "${MODEL_PATH}" ]]; then
  echo "--model-path is required." >&2
  usage
  exit 1
fi

if [[ -z "${STEPS}" ]]; then
  STEPS="${GEN_LENGTH}"
fi

if [[ -z "${OUTPUT_PATH}" ]]; then
  OUTPUT_PATH="${OUTPUT_DIR}/${TASK}-ns${NUM_FEWSHOT}-${GEN_LENGTH}"
fi

if [[ ! -d "${LM_DIR}" ]]; then
  echo "LM directory not found: ${LM_DIR}" >&2
  exit 1
fi

if [[ -n "${HF_HOME_PATH}" ]]; then
  export HF_HOME="${HF_HOME_PATH}"
fi
if [[ -n "${HF_DATASETS_CACHE_PATH}" ]]; then
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE_PATH}"
fi
if [[ -n "${TRANSFORMERS_CACHE_PATH}" ]]; then
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE_PATH}"
fi
if [[ "${OFFLINE}" -eq 1 ]]; then
  export HF_DATASETS_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_DISABLE_TELEMETRY=1
fi

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTHONNOUSERSITE=1

export CUDA_VISIBLE_DEVICES="${GPU_SPEC}"
if [[ -z "${NUM_PROCESSES}" ]]; then
  IFS=',' read -r -a _gpus <<< "${GPU_SPEC}"
  NUM_PROCESSES="${#_gpus[@]}"
fi

MODEL_ARGS=(
  "model_path=${MODEL_PATH}"
  "gen_length=${GEN_LENGTH}"
  "steps=${STEPS}"
  "block_length=${BLOCK_LENGTH}"
  "show_speed=${SHOW_SPEED}"
)

if [[ -n "${MODEL_CONFIG_PATH}" ]]; then
  MODEL_ARGS+=("model_config_path=${MODEL_CONFIG_PATH}")
fi

for arg in "${EXTRA_MODEL_ARGS[@]}"; do
  MODEL_ARGS+=("${arg}")
done

MODEL_ARGS_STR="$(IFS=,; echo "${MODEL_ARGS[*]}")"

RUN_CMD=(
  "${ACCELERATE_BIN}" launch
)

if [[ -n "${CONFIG_FILE}" ]]; then
  RUN_CMD+=(--config_file "${CONFIG_FILE}")
fi

RUN_CMD+=(
  --num_processes "${NUM_PROCESSES}"
  eval_dynin_omni.py
  --tasks "${TASK}"
  --num_fewshot "${NUM_FEWSHOT}"
  --confirm_run_unsafe_code
  --model dynin_omni_dist
  --model_args "${MODEL_ARGS_STR}"
  --output_path "${OUTPUT_PATH}"
)

if [[ "${LOG_SAMPLES}" -eq 1 ]]; then
  RUN_CMD+=(--log_samples)
fi

if [[ "${LIMIT}" -gt 0 ]]; then
  RUN_CMD+=(--limit "${LIMIT}")
fi

echo "[lmeval] lm_dir=${LM_DIR}"
echo "[lmeval] task=${TASK} fewshot=${NUM_FEWSHOT}"
echo "[lmeval] model_path=${MODEL_PATH}"
echo "[lmeval] gpus=${GPU_SPEC} num_processes=${NUM_PROCESSES}"
echo "[lmeval] output_path=${OUTPUT_PATH}"

mkdir -p "$(dirname "${OUTPUT_PATH}")"

cd "${LM_DIR}"
"${RUN_CMD[@]}"
