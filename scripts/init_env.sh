#!/bin/bash
set -euo pipefail

export PROJECT_ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONNOUSERSITE=1
export PIP_USER=0
unset PYTHONPATH || true

OVERWRITE_INIT_HF=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --overwrite)
      OVERWRITE_INIT_HF=1
      shift
      ;;
    -h|--help)
      cat <<'USAGE'
Usage: bash scripts/init_env.sh [--overwrite]

Options:
  --overwrite   Force overwrite Huggingface cache root to ${PROJECT_ROOT_DIR}/datasets/huggingface.
USAGE
      exit 0
      ;;
    *)
      echo "Error: unknown option '$1'" >&2
      exit 1
      ;;
  esac
done

# Keep HF cache env in current shell context for all subsequent pip/python calls.
if [[ "${OVERWRITE_INIT_HF}" -eq 1 ]]; then
  source "$PROJECT_ROOT_DIR/scripts/init_hf.sh" --overwrite
else
  source "$PROJECT_ROOT_DIR/scripts/init_hf.sh" --
fi

# Install native runtime dependencies used by audio/video preprocessing.
if [[ -n "${CONDA_PREFIX:-}" ]] && command -v conda >/dev/null 2>&1; then
  conda install -y -c conda-forge \
    "ffmpeg>=7,<8" \
    sox \
    "libstdcxx-ng>=13" \
    "libgcc-ng>=13"
  conda install -y -c nvidia \
    "cuda-toolkit=12.8" \
    "cuda-nvcc=12.8"
  if [[ -d "${CONDA_PREFIX}/lib" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  fi
else
  echo "[WARN] conda environment not detected. Please install ffmpeg/sox/libstdc++ runtime deps manually."
fi

python -m pip install --no-user --upgrade pip wheel "setuptools<81"
# CLIP is only used as a Python package here. Avoid pulling unconstrained deps
# (torch/numpy/torchvision) before the main constrained installation.
python -m pip install --no-user --no-build-isolation --no-deps \
  "git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1"

cd "$PROJECT_ROOT_DIR"
python -m pip install --no-user -r requirements.txt -c tokenizers/constraints-emova.txt

# Initialize tokenizers
bash "$PROJECT_ROOT_DIR/tokenizers/init_tokenizers.sh"

# Sanity-check that critical HF libs resolve from the active environment.
python - <<'PY'
import importlib.metadata as importlib_metadata
import importlib.util
import pathlib
import shutil
import sys
import accelerate
import torch
import transformers
import tokenizers

prefix = pathlib.Path(sys.prefix).resolve()
expected = {
    "transformers": "4.47.1",
    "tokenizers": "0.21.2",
    "accelerate": "1.8.1",
    "deepspeed": "0.17.2",
}
expected_torch_prefix = "2.7.1"
expected_torchvision = "0.22.1"
loaded = {
    "transformers": (transformers.__version__, pathlib.Path(transformers.__file__).resolve()),
    "tokenizers": (tokenizers.__version__, pathlib.Path(tokenizers.__file__).resolve()),
    "accelerate": (accelerate.__version__, pathlib.Path(accelerate.__file__).resolve()),
    "torch": (torch.__version__, pathlib.Path(torch.__file__).resolve()),
}

try:
    deepspeed_dist = importlib_metadata.distribution("deepspeed")
except importlib_metadata.PackageNotFoundError as exc:
    raise RuntimeError("deepspeed is not installed in the active environment.") from exc

deepspeed_spec = importlib.util.find_spec("deepspeed")
if deepspeed_spec is None or deepspeed_spec.origin is None:
    raise RuntimeError("deepspeed module spec could not be resolved in the active environment.")

loaded["deepspeed"] = (
    deepspeed_dist.version,
    pathlib.Path(deepspeed_spec.origin).resolve(),
)

try:
    torchvision_dist = importlib_metadata.distribution("torchvision")
except importlib_metadata.PackageNotFoundError as exc:
    raise RuntimeError("torchvision is not installed in the active environment.") from exc

torchvision_spec = importlib.util.find_spec("torchvision")
if torchvision_spec is None or torchvision_spec.origin is None:
    raise RuntimeError("torchvision module spec could not be resolved in the active environment.")

loaded["torchvision"] = (
    torchvision_dist.version,
    pathlib.Path(torchvision_spec.origin).resolve(),
)

for name, (version, path) in loaded.items():
    if name in expected and version != expected[name]:
        raise RuntimeError(
            f"{name} version mismatch: expected {expected[name]}, got {version} ({path})"
        )
    if name == "torch" and not version.startswith(expected_torch_prefix):
        raise RuntimeError(
            f"torch version mismatch: expected prefix {expected_torch_prefix}, got {version} ({path})"
        )
    if name == "torchvision" and version != expected_torchvision:
        raise RuntimeError(
            f"torchvision version mismatch: expected {expected_torchvision}, got {version} ({path})"
        )
    if prefix not in path.parents:
        raise RuntimeError(
            f"{name} is imported from outside current env: {path} (env prefix: {prefix}). "
            "Remove conflicting user-site packages or run with PYTHONNOUSERSITE=1."
        )

for binary in ("ffmpeg", "sox"):
    if shutil.which(binary) is None:
        raise RuntimeError(f"Missing required binary in PATH: {binary}")

accelerate_bin = shutil.which("accelerate")
if accelerate_bin is not None:
    accelerate_bin_path = pathlib.Path(accelerate_bin).resolve()
    if prefix not in accelerate_bin_path.parents:
        print(
            "[WARN] `accelerate` CLI is outside the active env "
            f"({accelerate_bin_path}, env prefix: {prefix}). "
            "Use `python -m accelerate.commands.launch` to avoid env mismatch."
        )

print(f"Torch version: {torch.__version__}")
PY

echo "Environment initialization complete. Run \`bash scripts/inference.sh --text\` to test the setup."
