# coding=utf-8
# Copyright 2024 The EMOVA team and The HuggingFace Inc. team. All rights reserved.
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
"""EMOVASpeechTokenizer model"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel

try:
    from emova_speech_tokenizer.speech_utils import (
        get_S2U_ckpt_config_path,
        load_config,
        VQCTCFinetuneModel,
        s2u_extract_unit_demo,
    )
    from emova_speech_tokenizer.speech_utils import (
        get_U2S_config_checkpoint_file,
        load_U2S_config,
        SynthesizerTrn,
        synthesis,
    )
except:
    raise ImportError(
        "Dependencies of emova speech tokenizer are not installed properly. Check https://github.com/emova-ollm/EMOVA_speech_tokenizer#installation for detailed instructions."
    )

from .configuration_emova_speech_tokenizer import EMOVASpeechTokenizerConfig


class EMOVASpeechTokenizer(PreTrainedModel):
    config_class = EMOVASpeechTokenizerConfig
    base_model_prefix = "emova_speech_tokenizer"

    def __init__(self, config: EMOVASpeechTokenizerConfig):
        super().__init__(config)
        self.config = config

        # s2u encoder configs
        _, S2U_config_path = get_S2U_ckpt_config_path(config.s2u_unit_type)
        s2u_cfg = load_config(config=S2U_config_path)
        s2u_cfg.model.pretrain_chkpt_path = None

        # u2s decoder configs
        U2S_config_file, _ = get_U2S_config_checkpoint_file(config.u2s_unit_type)
        u2s_cfg = load_U2S_config(U2S_config_file)

        # construct models
        self.s2u_config = s2u_cfg.model
        self.u2s_config = u2s_cfg
        self.encoder = VQCTCFinetuneModel(s2u_cfg.model, trainer=None)
        self.decoder = SynthesizerTrn(
            u2s_cfg.num_symbols,
            u2s_cfg.data.filter_length // 2 + 1,
            u2s_cfg.train.segment_size // u2s_cfg.data.hop_length,
            n_speakers=u2s_cfg.data.n_speakers,
            **u2s_cfg.model
        )
        self.style_embedding = nn.Embedding(
            config.u2s_num_styles, config.u2s_dim_styles
        )

    @property
    def device(self):
        return next(self.encoder.parameters()).device

    @property
    def dtype(self):
        return next(self.encoder.parameters()).dtype

    def encode(self, wav_file):
        speech_unit = s2u_extract_unit_demo(
            self.encoder, wav_file, model_name="SPIRAL-FSQ-CTC", reduced=True
        )
        unit_numbers = speech_unit.replace("<|speech_", "").replace("|>", " ").strip()
        unit_ids = [int(unit) for unit in unit_numbers.split(" ")]
        return torch.LongTensor(unit_ids).unsqueeze(0)

    def decode(self, speech_unit, condition=None, output_wav_file="output.wav"):
        content_unit = speech_unit.replace("<|speech_", "").replace("|>", " ").strip()
        style_centroid_embedding = (
            self.style_embedding(
                torch.LongTensor([self.config.u2s_style2idx[condition]]).to(self.device)
            ).unsqueeze(-1)
            if condition
            else None
        )
        audio = synthesis(
            content_unit,
            style_centroid_embedding,
            self.u2s_config,
            self.decoder,
            output_wav_file,
        )
        return audio
