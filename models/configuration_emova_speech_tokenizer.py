# coding=utf-8
# Copyright 2024 The EMOVA team and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""EMOVASpeechTokenizer configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from transformers import PretrainedConfig

__all__ = ["EMOVASpeechTokenizerConfig"]


def _load_style2idx_from_package() -> Dict[str, int]:
    try:
        import emova_speech_tokenizer  # type: ignore
    except Exception:
        return {}

    root = Path(emova_speech_tokenizer.__file__).resolve().parent
    cond_file = (
        root
        / "speech_tokenization"
        / "condition_style_centroid"
        / "condition2style_centroid.txt"
    )
    if not cond_file.exists():
        return {}

    mapping: Dict[str, int] = {}
    for line in cond_file.read_text(encoding="utf-8").splitlines()[1:]:
        if not line:
            continue
        condition = line.split("|", 1)[0]
        if condition not in mapping:
            mapping[condition] = len(mapping)
    return mapping


def _infer_style_dim_from_package() -> Optional[int]:
    try:
        import emova_speech_tokenizer  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        return None

    root = Path(emova_speech_tokenizer.__file__).resolve().parent
    cond_file = (
        root
        / "speech_tokenization"
        / "condition_style_centroid"
        / "condition2style_centroid.txt"
    )
    if not cond_file.exists():
        return None

    lines = cond_file.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        return None
    rel_path = lines[1].split("|", 1)[1]
    emb_path = (root / rel_path).resolve()
    if not emb_path.exists():
        return None
    try:
        arr = np.load(emb_path)
        return int(arr.shape[0])
    except Exception:
        return None


class EMOVASpeechTokenizerConfig(PretrainedConfig):
    model_type = "emova_speech_tokenizer"

    def __init__(
        self,
        s2u_unit_type: str = "40ms_multilingual_8888",
        u2s_unit_type: str = "40ms_multilingual_8888_xujing_cosyvoice_FT",
        u2s_num_styles: Optional[int] = None,
        u2s_dim_styles: Optional[int] = None,
        u2s_style2idx: Optional[Dict[str, int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if u2s_style2idx is None:
            u2s_style2idx = _load_style2idx_from_package()

        if u2s_num_styles is None:
            u2s_num_styles = len(u2s_style2idx) if u2s_style2idx else 126

        if u2s_dim_styles is None:
            u2s_dim_styles = _infer_style_dim_from_package() or 256

        self.s2u_unit_type = s2u_unit_type
        self.u2s_unit_type = u2s_unit_type
        self.u2s_num_styles = u2s_num_styles
        self.u2s_dim_styles = u2s_dim_styles
        self.u2s_style2idx = u2s_style2idx or {}
