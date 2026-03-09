#!/bin/bash
set -euo pipefail

_dynin_init_hf_is_sourced=0
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  _dynin_init_hf_is_sourced=1
fi

_dynin_init_hf_usage() {
  cat <<'USAGE'
Usage: source scripts/init_hf.sh [--overwrite] [--]

Options:
  --overwrite   Always use ${PROJECT_ROOT_DIR}/datasets/huggingface as cache root.
  --        Ignore caller positional arguments.
USAGE
}

OVERWRITE_PROJECT_CACHE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --overwrite)
      OVERWRITE_PROJECT_CACHE=1
      shift
      ;;
    --)
      shift
      break
      ;;
    -h|--help)
      _dynin_init_hf_usage
      if [[ "${_dynin_init_hf_is_sourced}" -eq 1 ]]; then
        return 0
      fi
      exit 0
      ;;
    *)
      echo "Error: unknown option '$1'" >&2
      _dynin_init_hf_usage >&2
      if [[ "${_dynin_init_hf_is_sourced}" -eq 1 ]]; then
        return 1
      fi
      exit 1
      ;;
  esac
done

export PROJECT_ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# Resolve a single HF cache root.
# Default behavior: DYNIN_OMNI_HF_CACHE_DIR > HF_HOME > project default.
# --overwrite behavior: project default only.
if [[ "${OVERWRITE_PROJECT_CACHE}" -eq 1 ]]; then
  HF_CACHE_ROOT_RAW="${PROJECT_ROOT_DIR}/datasets/huggingface"
else
  HF_CACHE_ROOT_RAW="${DYNIN_OMNI_HF_CACHE_DIR:-${HF_HOME:-${PROJECT_ROOT_DIR}/datasets/huggingface}}"
fi
HF_CACHE_ROOT="$(python -c "from pathlib import Path; print(Path(r'''${HF_CACHE_ROOT_RAW}''').expanduser().resolve())")"

# Force HF cache roots to writable project-local (or user-selected) directories
# so global environment values (e.g. TRANSFORMERS_CACHE=/models) cannot break setup.
export HF_HOME="${HF_CACHE_ROOT}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}"
export DYNIN_OMNI_HF_CACHE_DIR="${HF_HOME}"
# Use HF_* cache variables (TRANSFORMERS_CACHE is deprecated and may be set to
# unwritable global paths in cluster environments).
unset TRANSFORMERS_CACHE || true
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}"

if [[ "${DYNIN_OMNI_INIT_HF_QUIET:-0}" != "1" ]]; then
  if [[ "${OVERWRITE_PROJECT_CACHE}" -eq 1 ]]; then
    echo "Initialized HuggingFace cache directories at ${HF_HOME} (overwritten to project cache)."
  else
    echo "Initialized HuggingFace cache directories at ${HF_HOME}."
  fi
fi
