#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_PATH="configs/dynin_omni_demo.yaml"
PYTHON_BIN="python"
MODE=""
RESULT_DIR=""
EXTRA_ARGS=()
OVERWRITE_INIT_HF=0

# Validation data defaults (relative to repository root)
DEFAULT_I2I_EDIT_JSON="validation/data/text/i2i_edits.json"
DEFAULT_I2I_ORIGIN_IMG_ROOT="validation/data/image"
DEFAULT_T2I_METADATA_FILE="validation/data/text/t2i_metadata.jsonl"
DEFAULT_LM_QUESTIONS_FILE="validation/data/text/lm_questions.jsonl"

# I2I options
I2I_EDIT_JSON=""
I2I_ORIGIN_IMG_ROOT=""
I2I_MODEL_PATH=""

# MMU options
MMU_IMAGE_ROOT=""
VIDEO_IMAGE_ROOT=""

# Speech options
SPEECH_CKPT_PATH=""
LIBRISPEECH_ROOT=""
S2T_SAMPLES=""
T2S_SAMPLES=""

# T2I options
VALIDATION_PROMPTS_FILE=""
METADATA_FILE=""
N_SAMPLES=""
BATCH_SIZE=""
GUIDANCE_SCALE=""
GENERATION_TIMESTEPS=""

TEXT_RESULT_FILE="text_generation.txt"
MMU_RESULT_FILE="mmu_inference_results.json"
LM_QUESTIONS_FILE=""

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/inference.sh [mode] [options] [-- extra args]

Modes (choose one):
  --text                 Run validation/generate.py
  --i2i                  Run validation/i2i_generate.py
  --mmu                  Run validation/mmu_generate.py (i2t+v2t)
  --speech               Run validation/speech.py (s2t+t2s)
  --t2i                  Run validation/t2i_generate.py

Common options:
  --config <path>        Config path (default: configs/dynin_omni_demo.yaml)
  --python <bin>         Python executable (default: python)
  --result <dir>         Result directory
                         (default: results/<mode>)
  --overwrite            Overwrite HF cache root to ${PROJECT_ROOT_DIR}/datasets/huggingface
  -h, --help             Show this help message

Text options (optional for --text):
  --questions-file <path> (default: validation/data/text/lm_questions.jsonl)

I2I options (optional for --i2i; defaults shown):
  --edit-json <path>      (default: validation/data/text/i2i_edits.json)
  --origin-img-root <path> (default: validation/data/image)
  --model-path <path>    Optional model override

MMU options (optional for --mmu):
  --mmu-image-root <path>
  --video-image-root <path>

Speech options (optional for --speech):
  --ckpt-path <path>          Optional model override
  --librispeech-root <path>   Optional
  --s2t-samples <int>         Optional
  --t2s-samples <int>         Optional

T2I options (optional for --t2i):
  --validation-prompts-file <path>
  --metadata-file <path>
                         (default when both are omitted:
                          validation/data/text/t2i_metadata.jsonl)
  --n-samples <int>
  --batch-size <int>
  --guidance-scale <float>
  --generation-timesteps <int>

Examples:
  bash scripts/inference.sh --mmu --config configs/dynin_omni_demo.yaml
  bash scripts/inference.sh --i2i
  bash scripts/inference.sh --i2i --edit-json data/image/edit.json --origin-img-root data/image --result results/i2i
  bash scripts/inference.sh --speech --result results/speech
  bash scripts/inference.sh --t2i --result results/t2i --metadata-file validation/data/text/t2i_metadata.jsonl
USAGE
}

ensure_single_mode() {
  local candidate="$1"
  if [[ -n "${MODE}" && "${MODE}" != "${candidate}" ]]; then
    echo "Error: choose exactly one mode (--text|--i2i|--mmu|--speech|--t2i)." >&2
    exit 1
  fi
  MODE="${candidate}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --text)
      ensure_single_mode "text"
      shift
      ;;
    --i2i)
      ensure_single_mode "i2i"
      shift
      ;;
    --mmu)
      ensure_single_mode "mmu"
      shift
      ;;
    --speech)
      ensure_single_mode "speech"
      shift
      ;;
    --t2i)
      ensure_single_mode "t2i"
      shift
      ;;
    --config)
      if [[ $# -lt 2 ]]; then
        echo "Error: --config requires a value." >&2
        exit 1
      fi
      CONFIG_PATH="$2"
      shift 2
      ;;
    --python)
      if [[ $# -lt 2 ]]; then
        echo "Error: --python requires a value." >&2
        exit 1
      fi
      PYTHON_BIN="$2"
      shift 2
      ;;
    --result)
      if [[ $# -lt 2 ]]; then
        echo "Error: --result requires a value." >&2
        exit 1
      fi
      RESULT_DIR="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE_INIT_HF=1
      shift
      ;;
    --questions-file)
      LM_QUESTIONS_FILE="$2"
      shift 2
      ;;
    --edit-json)
      I2I_EDIT_JSON="$2"
      shift 2
      ;;
    --origin-img-root)
      I2I_ORIGIN_IMG_ROOT="$2"
      shift 2
      ;;
    --model-path)
      I2I_MODEL_PATH="$2"
      shift 2
      ;;
    --mmu-image-root)
      MMU_IMAGE_ROOT="$2"
      shift 2
      ;;
    --video-image-root)
      VIDEO_IMAGE_ROOT="$2"
      shift 2
      ;;
    --ckpt-path)
      SPEECH_CKPT_PATH="$2"
      shift 2
      ;;
    --librispeech-root)
      LIBRISPEECH_ROOT="$2"
      shift 2
      ;;
    --s2t-samples)
      S2T_SAMPLES="$2"
      shift 2
      ;;
    --t2s-samples)
      T2S_SAMPLES="$2"
      shift 2
      ;;
    --validation-prompts-file)
      VALIDATION_PROMPTS_FILE="$2"
      shift 2
      ;;
    --metadata-file)
      METADATA_FILE="$2"
      shift 2
      ;;
    --n-samples)
      N_SAMPLES="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --guidance-scale)
      GUIDANCE_SCALE="$2"
      shift 2
      ;;
    --generation-timesteps)
      GENERATION_TIMESTEPS="$2"
      shift 2
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown option '$1'" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${MODE}" ]]; then
  echo "Error: mode is required." >&2
  usage
  exit 1
fi

# Centralize HF cache setup after parsing so script args are not forwarded to
# init_hf.sh unintentionally.
export DYNIN_OMNI_INIT_HF_QUIET=1
if [[ "${OVERWRITE_INIT_HF}" -eq 1 ]]; then
  source "${REPO_ROOT}/scripts/init_hf.sh" --overwrite
else
  source "${REPO_ROOT}/scripts/init_hf.sh" --
fi
unset DYNIN_OMNI_INIT_HF_QUIET || true

# Keep runtime imports isolated from user-site packages and ensure conda libs
# are discoverable for ffmpeg/torchcodec dependent paths.
export PYTHONNOUSERSITE=1
export PYTHONPATH=
if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Error: config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ "${MODE}" == "i2i" ]]; then
  if [[ -z "${I2I_EDIT_JSON}" ]]; then
    I2I_EDIT_JSON="${DEFAULT_I2I_EDIT_JSON}"
  fi
  if [[ -z "${I2I_ORIGIN_IMG_ROOT}" ]]; then
    I2I_ORIGIN_IMG_ROOT="${DEFAULT_I2I_ORIGIN_IMG_ROOT}"
  fi
fi

if [[ "${MODE}" == "t2i" ]]; then
  if [[ -z "${VALIDATION_PROMPTS_FILE}" && -z "${METADATA_FILE}" ]]; then
    METADATA_FILE="${DEFAULT_T2I_METADATA_FILE}"
  fi
fi

if [[ "${MODE}" == "text" ]]; then
  if [[ -z "${LM_QUESTIONS_FILE}" ]]; then
    LM_QUESTIONS_FILE="${DEFAULT_LM_QUESTIONS_FILE}"
  fi
fi

if [[ -z "${RESULT_DIR}" ]]; then
  RESULT_DIR="results/${MODE}"
fi
mkdir -p "${RESULT_DIR}"

CMD=()

case "${MODE}" in
  text)
    CMD=(
      "${PYTHON_BIN}" validation/generate.py
      "config=${CONFIG_PATH}"
      "questions_file=${LM_QUESTIONS_FILE}"
    )
    ;;
  i2i)
    if [[ -z "${I2I_EDIT_JSON}" || -z "${I2I_ORIGIN_IMG_ROOT}" ]]; then
      echo "Error: i2i input paths are not resolved." >&2
      exit 1
    fi
    if [[ ! -f "${I2I_EDIT_JSON}" ]]; then
      echo "Error: i2i edit json not found: ${I2I_EDIT_JSON}" >&2
      exit 1
    fi
    if [[ ! -d "${I2I_ORIGIN_IMG_ROOT}" ]]; then
      echo "Error: i2i origin image root not found: ${I2I_ORIGIN_IMG_ROOT}" >&2
      exit 1
    fi
    CMD=(
      "${PYTHON_BIN}" validation/i2i_generate.py
      --config "${CONFIG_PATH}"
      --edit_json "${I2I_EDIT_JSON}"
      --origin_img_root "${I2I_ORIGIN_IMG_ROOT}"
      --outdir "${RESULT_DIR}"
    )
    if [[ -n "${I2I_MODEL_PATH}" ]]; then
      CMD+=(--model_path "${I2I_MODEL_PATH}")
    fi
    ;;
  mmu)
    CMD=("${PYTHON_BIN}" validation/mmu_generate.py "config=${CONFIG_PATH}")
    if [[ -n "${MMU_IMAGE_ROOT}" ]]; then
      CMD+=("mmu_image_root=${MMU_IMAGE_ROOT}")
    fi
    if [[ -n "${VIDEO_IMAGE_ROOT}" ]]; then
      CMD+=("video_image_root=${VIDEO_IMAGE_ROOT}")
    fi
    CMD+=("mmu_output_file=${RESULT_DIR}/${MMU_RESULT_FILE}")
    ;;
  speech)
    CMD=(
      "${PYTHON_BIN}" validation/speech.py
      --config "${CONFIG_PATH}"
      --out-dir "${RESULT_DIR}"
    )
    if [[ -n "${SPEECH_CKPT_PATH}" ]]; then
      CMD+=(--ckpt-path "${SPEECH_CKPT_PATH}")
    fi
    if [[ -n "${LIBRISPEECH_ROOT}" ]]; then
      CMD+=(--librispeech-root "${LIBRISPEECH_ROOT}")
    fi
    if [[ -n "${S2T_SAMPLES}" ]]; then
      CMD+=(--s2t-samples "${S2T_SAMPLES}")
    fi
    if [[ -n "${T2S_SAMPLES}" ]]; then
      CMD+=(--t2s-samples "${T2S_SAMPLES}")
    fi
    ;;
  t2i)
    CMD=("${PYTHON_BIN}" validation/t2i_generate.py "config=${CONFIG_PATH}")
    CMD+=("outdir=${RESULT_DIR}")
    if [[ -n "${VALIDATION_PROMPTS_FILE}" ]]; then
      CMD+=("validation_prompts_file=${VALIDATION_PROMPTS_FILE}")
    fi
    if [[ -n "${METADATA_FILE}" ]]; then
      CMD+=("metadata_file=${METADATA_FILE}")
    fi
    if [[ -n "${N_SAMPLES}" ]]; then
      CMD+=("n_samples=${N_SAMPLES}")
    fi
    if [[ -n "${BATCH_SIZE}" ]]; then
      CMD+=("batch_size=${BATCH_SIZE}")
    fi
    if [[ -n "${GUIDANCE_SCALE}" ]]; then
      CMD+=("guidance_scale=${GUIDANCE_SCALE}")
    fi
    if [[ -n "${GENERATION_TIMESTEPS}" ]]; then
      CMD+=("generation_timesteps=${GENERATION_TIMESTEPS}")
    fi
    ;;
  *)
    echo "Error: unsupported mode '${MODE}'." >&2
    exit 1
    ;;
esac

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Running mode: ${MODE}"
echo "Config: ${CONFIG_PATH}"
echo "Result: ${RESULT_DIR}"
echo "Command: ${CMD[*]}"
if [[ "${MODE}" == "text" ]]; then
  "${CMD[@]}" 2>&1 | tee "${RESULT_DIR}/${TEXT_RESULT_FILE}"
else
  "${CMD[@]}"
fi
