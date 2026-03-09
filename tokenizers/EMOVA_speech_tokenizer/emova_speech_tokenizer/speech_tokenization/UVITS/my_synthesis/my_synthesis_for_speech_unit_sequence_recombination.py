from pathlib import Path

import torch
from scipy.io.wavfile import write
from tqdm import tqdm

import uvits_utils
from emova_speech_tokenizer.speech_tokenization.UVITS.models import SynthesizerTrn
from text import text_to_sequence

import random
import shutil
import librosa
import numpy as np
import json

from data_utils import load_wav_to_torch, spectrogram_torch


def get_audio(filename, hps):
    audio, sampling_rate = librosa.load(filename, sr=hps.sampling_rate)
    audio_norm = torch.FloatTensor(audio.astype(np.float32)).unsqueeze(0)
    spec = spectrogram_torch(audio_norm, hps.filter_length,
                             hps.sampling_rate, hps.hop_length, hps.win_length,
                             center=False)
    spec = torch.squeeze(spec, 0)
    return spec, audio_norm


def get_text_unit_list(TTS_input_output_file):
    text_list, unit_seq_list = [], []
    with open(TTS_input_output_file, 'r') as f:
        for line in f:
            if line.startswith('input:'):
                text_list.append(line.lstrip('input:').rstrip('\n').strip())
            elif line.startswith('output: '):
                unit_seq_list.append(line.lstrip('output:').rstrip('\n').strip())
    return text_list, unit_seq_list


def get_U2S_config_checkpoint_file(unit_type, language='English'):
    assert language in ['English', 'Chinese']
    assert unit_type =='40ms_multilingual_8888_xujing_cosyvoice_FT'
    # English and Chinese using the same UVITS model and config!!
    config_file = "./speech_tokenization/UVITS/my_UVITS_model/ROMA_1n8g_u2s_40ms_multilingual_8888_xujing_cosyvoice_EN_CH_female_male_FT_20240812/config.json"
    checkpoint_file = "./speech_tokenization/UVITS/my_UVITS_model/ROMA_1n8g_u2s_40ms_multilingual_8888_xujing_cosyvoice_EN_CH_female_male_FT_20240812/saved_checkpoint/G_322000.pth"
    return config_file, checkpoint_file