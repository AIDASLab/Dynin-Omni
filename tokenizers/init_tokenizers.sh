#!/bin/bash
set -euo pipefail

export PROJECT_ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONSTRAINTS_FILE="$PROJECT_ROOT_DIR/tokenizers/constraints-emova.txt"
export PYTHONNOUSERSITE=1
export PIP_USER=0
unset PYTHONPATH || true

# Keep HF cache env in current shell context for all subsequent pip/python calls.
source "$PROJECT_ROOT_DIR/scripts/init_hf.sh" --

cd "$PROJECT_ROOT_DIR/tokenizers/EMOVA_speech_tokenizer"
python -m pip install --no-user -r requirements/requirements.txt -c "$CONSTRAINTS_FILE"
python -m pip install --no-user -e . --no-deps

python - <<'PY'
import pathlib
import sys
import sentencepiece
import google.protobuf
import onnx
import emova_speech_tokenizer
import transformers
import tokenizers

if sentencepiece.__version__ != "0.1.99":
    raise RuntimeError(f"Unexpected sentencepiece version: {sentencepiece.__version__}")
if not google.protobuf.__version__.startswith("3.20"):
    raise RuntimeError(f"Unexpected protobuf version: {google.protobuf.__version__}")
if onnx.__version__ != "1.12.0":
    raise RuntimeError(f"Unexpected onnx version: {onnx.__version__}")
if transformers.__version__ != "4.47.1":
    raise RuntimeError(f"Unexpected transformers version: {transformers.__version__}")
if tokenizers.__version__ != "0.21.2":
    raise RuntimeError(f"Unexpected tokenizers version: {tokenizers.__version__}")

prefix = pathlib.Path(sys.prefix).resolve()
for name, module in (
    ("transformers", transformers),
    ("tokenizers", tokenizers),
    ("sentencepiece", sentencepiece),
    ("onnx", onnx),
):
    mod_path = pathlib.Path(module.__file__).resolve()
    if prefix not in mod_path.parents:
        raise RuntimeError(
            f"{name} imported outside current env: {mod_path} (env prefix: {prefix})"
        )
PY
