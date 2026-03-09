# coding=utf-8
# Copyright 2026 Dynin-Omni Team, AIDAS Lab, Seoul National University
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

import os
import sys
import json
import pandas
import logging
import math
import shutil
import time
import cv2
import glob
import random
from pathlib import Path
from typing import Union
import csv
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import wandb
import torch
from torch.optim import AdamW
from lightning.pytorch.utilities import CombinedLoader

from transformers import AutoTokenizer, AutoConfig
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed
from tqdm.auto import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.modeling_emova_speech_tokenizer import EMOVASpeechTokenizer
from datasets import load_dataset as hf_load_dataset
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from training.data import (
    SpeechTextDataset,
    MixedSpeechTextDataset,
    load_video_mp4,
    VideoCaptionDataset,
    S2T_INSTRUCTION,
    T2S_INSTRUCTION,
)

from training.data import Text2ImageDataset
from training.config_resolver import (
    apply_dataset_sources,
    configure_hf_cache_env as common_configure_hf_cache_env,
    resolve_hf_cache_root as common_resolve_hf_cache_root,
    resolve_model_local_files_only,
)
from training.utils import get_config, flatten_omega_conf, image_transform

from models import MAGVITv2, get_mask_schedule, DyninOmniModelLM, DyninOmniConfig
from training.prompting_utils import UniversalPrompting
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error

import re
import editdistance
import soundfile as sf
from functools import partial
from transformers import pipeline

os.environ["TOKENIZERS_PARALLELISM"] = "true"

SYSTEM_PROMPT_LEN = 28

from training.utils import mask_or_random_replace_tokens, AverageMeter

try:
    import apex

    is_apex_available = True
except ImportError:
    is_apex_available = False

logger = get_logger(__name__, log_level="INFO")
# Use stdlib logging before Accelerator initialization.
bootstrap_logger = logging.getLogger(__name__)


def _is_env_truthy(name: str) -> bool:
    value = os.getenv(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _first_non_empty_text(*values) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        if text.upper() == "NONE":
            continue
        return text
    return None


def _resolve_stage1_model_sources(config):
    model_cfg = config.model.dynin_omni

    tokenizer_source = _first_non_empty_text(
        model_cfg.get("tokenizer_path", None),
        model_cfg.get("tokenizer_repo_id", None),
        model_cfg.get("repo_id", None),
    )
    model_source = _first_non_empty_text(
        model_cfg.get("pretrained_model_path", None),
        model_cfg.get("repo_id", None),
    )
    vq_image_source = _first_non_empty_text(
        config.model.vq_model_image.get("vq_model_name", None),
        config.model.vq_model_image.get("repo_id", None),
    )
    vq_audio_source = _first_non_empty_text(
        config.model.vq_model_audio.get("vq_model_name", None),
        config.model.vq_model_audio.get("repo_id", None),
    )

    missing = []
    if tokenizer_source is None:
        missing.append("model.dynin_omni.tokenizer_path")
    if model_source is None:
        missing.append("model.dynin_omni.pretrained_model_path")
    if vq_image_source is None:
        missing.append("model.vq_model_image.vq_model_name")
    if vq_audio_source is None:
        missing.append("model.vq_model_audio.vq_model_name")
    if missing:
        raise ValueError("Missing required source fields: " + ", ".join(missing))

    return tokenizer_source, model_source, vq_image_source, vq_audio_source


def _to_plain_dict(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if OmegaConf.is_config(value):
        container = OmegaConf.to_container(value, resolve=True)
        return container if isinstance(container, dict) else {}
    return {}


def _sanitize_experiment_intervals(config) -> None:
    interval_keys = ("save_every", "generate_every", "eval_every", "log_every")
    for key in interval_keys:
        value = config.experiment.get(key, None)
        if value is None:
            config.experiment[key] = None
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            bootstrap_logger.warning(
                "Invalid experiment.%s=%r. This trigger is disabled (set to null).",
                key,
                value,
            )
            config.experiment[key] = None
            continue
        config.experiment[key] = int(value)


def _resolve_hf_cache_root(config) -> str:
    return common_resolve_hf_cache_root(config, project_root=PROJECT_ROOT)


def _configure_hf_cache_env(config) -> str:
    return common_configure_hf_cache_env(config, project_root=PROJECT_ROOT)


def _load_dataset_with_cache(*args, **kwargs):
    kwargs.setdefault(
        "cache_dir",
        os.getenv(
            "DYNIN_OMNI_HF_CACHE_DIR", str(Path(PROJECT_ROOT) / "datasets" / "huggingface")
        ),
    )
    return hf_load_dataset(*args, **kwargs)


def pad_tensor(tensor, length, value):
    pad_size = length - tensor.shape[1]
    if pad_size <= 0:
        return tensor
    # Pad on the right side of the sequence (last dimension)
    return torch.nn.functional.pad(tensor, (0, pad_size), "constant", value)


def pad_answer_lengths(ans: torch.Tensor, length: int) -> torch.Tensor:
    b, l = ans.shape
    if l >= length:
        return ans
    pad_block = ans[:, :1].expand(b, length - l)
    return torch.cat([ans, pad_block], dim=1)


def resize_vocab(model, config):
    logger.info(f"Resizing token embeddings to {config.model.dynin_omni.new_vocab_size}")
    model.resize_token_embeddings(config.model.dynin_omni.new_vocab_size)


def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    elif model_type == "emova":
        return EMOVASpeechTokenizer
    else:
        raise ValueError(f"model_type {model_type} not supported.")


def collate_fn_audio(batch):
    # In this setup, the tokenizer handles batching of audio paths
    return {
        "audio_path": [item["audio_path"] for item in batch],
        "text": [item["text"] for item in batch],
    }


def collate_fn_video_caption(batch):

    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None

    frame_list = []
    input_ids_list = []
    for item in batch:
        frame_tensor = torch.stack(item["video"], dim=0)  # (T, C, H, W)
        frame_list.append(frame_tensor)
        input_ids_list.append(item["caption"])

    frames = torch.stack(frame_list, dim=0)  # (B, T, C, H, W)

    return {
        "video": frames,  # torch tensor (B, T, C, H, W)
        "captions": input_ids_list,  # input_ids (B, seq_len)
    }


def s2t_eval_collate_fn(batch, vq_model_audio, tokenizer, uni_prompting, config):

    audio_tokens_batch = []
    offset = len(uni_prompting.text_tokenizer) + config.model.dynin_omni.codebook_size
    for item in batch:
        path = item["audio_path"]
        tokens = vq_model_audio.encode(path)
        tokens_with_offset = tokens + offset
        audio_tokens_batch.append(tokens_with_offset)

    sptids_dict = uni_prompting.sptids_dict
    device = audio_tokens_batch[0].device
    batched_input_ids = []

    for audio_tokens in audio_tokens_batch:
        task_tensor = sptids_dict["<|s2t|>"].to(device).unsqueeze(0)
        soa_tensor = sptids_dict["<|soa|>"].to(device).unsqueeze(0)
        eoa_tensor = sptids_dict["<|eoa|>"].to(device).unsqueeze(0)
        audio_block = torch.cat(
            [task_tensor, soa_tensor, audio_tokens, eoa_tensor], dim=1
        )

        prompt_text = random.choice(S2T_INSTRUCTION)
        full_prompt_text = f"<|start_header_id|>user<|end_header_id|>\n{prompt_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        prompt_tensor = tokenizer(full_prompt_text, return_tensors="pt").input_ids.to(
            device
        )

        final_sequence = torch.cat([audio_block, prompt_tensor], dim=1)
        batched_input_ids.append(final_sequence.squeeze(0))

    max_len = max(seq.size(0) for seq in batched_input_ids)
    pad_token_id = 126093

    final_batch_input_ids = torch.full(
        (len(batched_input_ids), max_len), pad_token_id, dtype=torch.long, device=device
    )

    for i, seq in enumerate(batched_input_ids):
        final_batch_input_ids[i, -len(seq) :] = seq

    return {
        "input_ids": final_batch_input_ids,
        "gt_texts": [item["gt_text"] for item in batch],
        "sample_ids": [item["sample_id"] for item in batch],
    }


# Evaluation helper functions.
def normalize_text(text):
    """A simple normalizer for WER calculation."""
    text = text.lower()
    text = re.sub(r"[^\w\s']", "", text)
    return text


def calculate_wer(predictions, references):
    """Calculates the Word Error Rate (WER) between predicted and ground truth texts."""
    predictions = [normalize_text(p) for p in predictions]
    references = [normalize_text(r) for r in references]

    total_errors = 0
    total_words = 0
    for pred, ref in zip(predictions, references):
        pred_words = pred.split()
        ref_words = ref.split()
        total_errors += editdistance.eval(pred_words, ref_words)
        total_words += len(ref_words)

    wer = total_errors / total_words if total_words > 0 else 0.0
    return wer, total_errors, total_words


class S2TEvalDataset(Dataset):
    def __init__(self, hf_dataset, root_path):
        self.hf_dataset = hf_dataset
        self.root_path = root_path

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        example = self.hf_dataset[idx]
        sample_id = example["id"]
        speaker_id, chapter_id, _ = sample_id.split("-")
        audio_path = os.path.join(
            self.root_path, speaker_id, chapter_id, f"{sample_id}.flac"
        )

        return {
            "audio_path": audio_path,
            "gt_text": example["text"],
            "sample_id": sample_id,
        }


# T2S evaluation dataset.
class T2SEvalDataset(Dataset):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        example = self.hf_dataset[idx]
        return {"gt_text": example["text"], "sample_id": example["id"]}


# Evaluation functions.
@torch.no_grad()
def evaluate_s2t(
    model, vq_model_audio, uni_prompting, config, accelerator, global_step
):
    logger.info("***** Running S2T Evaluation (WER on Librispeech test-clean) *****")
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.eval()
    eval_cfg = _to_plain_dict(getattr(config, "evaluation", {}))
    s2t_librispeech_root = eval_cfg.get("s2t_librispeech_root")
    if not s2t_librispeech_root:
        logger.warning(
            "evaluation.s2t_librispeech_root is not set. Skipping S2T evaluation."
        )
        return

    # Load the evaluation dataset.
    try:
        s2t_eval_dataset_raw = _load_dataset_with_cache(
            "librispeech_asr", "clean", split="test", streaming=False
        ).select(range(128))
        s2t_eval_dataset = S2TEvalDataset(
            s2t_eval_dataset_raw, root_path=s2t_librispeech_root
        )
    except Exception as e:
        logger.error(f"Failed to load S2T evaluation dataset: {e}")
        return

    collate_with_args = partial(
        s2t_eval_collate_fn,
        vq_model_audio=vq_model_audio,
        tokenizer=uni_prompting.text_tokenizer,
        uni_prompting=uni_prompting,
        config=config,
    )

    s2t_eval_dataloader = DataLoader(
        s2t_eval_dataset,
        batch_size=config.training.batch_size_s2t,
        shuffle=False,
        collate_fn=collate_with_args,
    )
    s2t_eval_dataloader = accelerator.prepare(s2t_eval_dataloader)

    local_results = []

    for batch in tqdm(
        s2t_eval_dataloader,
        desc="S2T Evaluation",
        disable=not accelerator.is_main_process,
    ):
        input_ids = batch["input_ids"]
        gt_texts = batch["gt_texts"]
        sample_ids = batch["sample_ids"]

        output_ids = unwrapped_model.mmu_generate(
            input_ids,
            max_new_tokens=256,
            steps=128,
            block_length=64,
            remasking="low_confidence",
        )

        decoded_texts = uni_prompting.text_tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1] :], skip_special_tokens=True
        )

        for i in range(len(decoded_texts)):
            local_results.append(
                {"gt_text": gt_texts[i], "decoded_text": decoded_texts[i]}
            )

    # Gather and log results.
    all_results = accelerator.gather_for_metrics(local_results)

    if accelerator.is_main_process:
        if not all_results:
            logger.warning("S2T evaluation produced no results.")
            return
        gt_list = [res["gt_text"] for res in all_results]
        pred_list = [res["decoded_text"] for res in all_results]

        wer, errors, words = calculate_wer(pred_list, gt_list)
        logger.info(
            f"S2T Final WER (Librispeech test-clean): {wer:.4f} | Word Errors: {errors} | Total Words: {words}"
        )

        accelerator.log(
            {
                "eval/s2t_wer": wer,
                "eval/s2t_word_errors": errors,
                "eval/s2t_total_words": words,
            },
            step=global_step,
        )


# T2S evaluation.
@torch.no_grad()
def evaluate_t2s(
    model, vq_model_audio, uni_prompting, config, accelerator, global_step
):
    logger.info("***** Running T2S Evaluation (WER via Whisper on Librispeech) *****")
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.eval()
    eval_cfg = _to_plain_dict(getattr(config, "evaluation", {}))
    eval_audio_dirname = str(eval_cfg.get("audio_output_dirname", "eval_audio"))

    # Load dataset and Whisper model on the main process first.
    if accelerator.is_main_process:
        try:
            t2s_eval_dataset_raw = _load_dataset_with_cache(
                "librispeech_asr", "clean", split="test"
            ).select(range(128))
            whisper_pipe = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-large-v3",
                device=accelerator.device,
            )
            os.makedirs(
                Path(config.experiment.output_dir) / eval_audio_dirname, exist_ok=True
            )
        except Exception as e:
            logger.error(f"Failed to load T2S dataset or Whisper model: {e}")
            whisper_pipe = None

    accelerator.wait_for_everyone()
    # Initialize on non-main processes after synchronization.
    if not accelerator.is_main_process:
        try:
            t2s_eval_dataset_raw = _load_dataset_with_cache(
                "librispeech_asr", "clean", split="test"
            ).select(range(128))
            whisper_pipe = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-large-v3",
                device=accelerator.device,
            )
        except Exception as e:
            whisper_pipe = None

    if whisper_pipe is None:
        logger.warning("Skipping T2S evaluation as Whisper or dataset failed to load.")
        return

    output_dir_per_step = os.path.join(
        config.experiment.output_dir, eval_audio_dirname, f"step_{global_step}"
    )
    os.makedirs(output_dir_per_step, exist_ok=True)

    t2s_eval_dataset = T2SEvalDataset(t2s_eval_dataset_raw)
    t2s_dataloader = DataLoader(
        t2s_eval_dataset, batch_size=config.training.batch_size_t2s
    )
    t2s_dataloader = accelerator.prepare(t2s_dataloader)

    local_results = []
    mask_token_id = unwrapped_model.config.mask_token_id
    mask_schedule = get_mask_schedule(config.training.get("mask_schedule", "cosine"))
    offset = len(uni_prompting.text_tokenizer) + config.model.dynin_omni.codebook_size

    # Evaluation loop.
    for batch in tqdm(
        t2s_dataloader, desc="T2S Evaluation", disable=not accelerator.is_main_process
    ):
        gt_texts = batch["gt_text"]
        sample_ids = batch["sample_id"]

        prompts = [f"{text}\n{random.choice(T2S_INSTRUCTION)}" for text in gt_texts]
        batch_size = len(prompts)

        speech_token_length = (
            config.dataset.preprocessing.max_aud_length - 1
        )  # Reserve one token for SOA.
        audio_tokens = (
            torch.ones(
                (batch_size, speech_token_length),
                dtype=torch.long,
                device=accelerator.device,
            )
            * mask_token_id
        )
        input_ids, attention_mask = uni_prompting((prompts, audio_tokens), "t2s_gen")

        if config.training.guidance_scale > 0:
            uncond_input_ids, uncond_attention_mask = uni_prompting(
                ([""] * batch_size, audio_tokens), "t2s_gen"
            )
        else:
            uncond_input_ids, uncond_attention_mask = None, None

        output_ids = unwrapped_model.t2s_generate(
            input_ids=input_ids,
            uncond_input_ids=uncond_input_ids,
            attention_mask=attention_mask,
            uncond_attention_mask=uncond_attention_mask,
            guidance_scale=config.training.guidance_scale,
            temperature=1.0,
            timesteps=24,
            noise_schedule=mask_schedule,
            noise_type="mask",
            seq_len=100,
            uni_prompting=uni_prompting,
            config=config,
        )

        # Decode generated speech and run Whisper.
        for i in range(batch_size):
            gt = gt_texts[i].rsplit("\n", 1)[-1].strip()

            gen_speech_tokens = output_ids[i]

            id_list = gen_speech_tokens.cpu().tolist()

            if not id_list:
                logger.warning(
                    f"Generated token list is empty for sample {sample_ids[i]}. Skipping."
                )
                continue

            speech_unit_str = " ".join(map(str, id_list))
            speech_unit_for_decode = "".join(
                [f"<|speech_{unit}|>" for unit in speech_unit_str.split(" ")]
            )

            filename = f"process_{accelerator.process_index}_{sample_ids[i]}.wav"
            output_wav_path = os.path.join(output_dir_per_step, filename)
            condition = "gender-female_emotion-neutral_speed-normal_pitch-normal"

            _ = vq_model_audio.decode(
                speech_unit_for_decode,
                condition=condition,
                output_wav_file=output_wav_path,
            )

            whisper_result = whisper_pipe(
                output_wav_path, generate_kwargs={"language": "english"}
            )
            whisper_text = whisper_result.get("text", "")

            local_results.append(
                {
                    "sample_id": sample_ids[i],
                    "gt_text": gt,
                    "whisper_text": whisper_text,
                    "audio_path": output_wav_path,
                }
            )

    # Gather and log results.
    all_results = accelerator.gather_for_metrics(local_results)

    if accelerator.is_main_process:
        gt_list = [res["gt_text"] for res in all_results]
        pred_list = [res["whisper_text"] for res in all_results]

        wer, errors, words = calculate_wer(pred_list, gt_list)
        logger.info(
            f"T2S Final WER (via Whisper): {wer:.4f} | Word Errors: {errors} | Total Words: {words}"
        )

        accelerator.log(
            {
                "eval/t2s_wer": wer,
                "eval/t2s_word_errors": errors,
                "eval/t2s_total_words": words,
            },
            step=global_step,
        )

        # Log a subset of generated samples to W&B.
        results_table = wandb.Table(
            columns=["ID", "Ground Truth", "Whisper Transcription", "Generated Audio"]
        )
        for res in all_results[:8]:
            audio = wandb.Audio(res["audio_path"], caption=res["whisper_text"])
            results_table.add_data(
                res["sample_id"], res["gt_text"], res["whisper_text"], audio
            )

        accelerator.log({"eval/t2s_samples": results_table}, step=global_step)


@torch.no_grad()
def evaluate_t2s_fixed(
    model, vq_model_audio, uni_prompting, config, accelerator, global_step
):
    """
    Text-to-Speech (fixed-length) evaluation:
      - Input prompt contains SOA + [MASK]*L + EOA (EOA is injected, not predicted)
      - The model only fills VQ codes for exactly L positions (no EOA/EOS prediction)
      - Generated audio is transcribed by Whisper; we report WER
    """
    logger.info("***** Running T2S (fixed-length) Evaluation *****")
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.eval()
    eval_cfg = _to_plain_dict(getattr(config, "evaluation", {}))
    eval_audio_dirname = str(eval_cfg.get("audio_output_dirname", "eval_audio"))

    # Load evaluation dataset and Whisper on the main process.
    if accelerator.is_main_process:
        try:
            ds_raw = _load_dataset_with_cache(
                "librispeech_asr", "clean", split="test"
            ).select(range(128))
            whisper_pipe = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-large-v3",
                device=accelerator.device,
            )
            os.makedirs(
                Path(config.experiment.output_dir) / eval_audio_dirname, exist_ok=True
            )
        except Exception as e:
            logger.error(f"Failed to load dataset or Whisper model: {e}")
            whisper_pipe = None

    accelerator.wait_for_everyone()

    # Initialize on non-main processes.
    if not accelerator.is_main_process:
        try:
            ds_raw = _load_dataset_with_cache(
                "librispeech_asr", "clean", split="test"
            ).select(range(128))
            whisper_pipe = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-large-v3",
                device=accelerator.device,
            )
        except Exception:
            whisper_pipe = None

    if whisper_pipe is None:
        logger.warning("Skipping T2S fixed evaluation due to missing Whisper/dataset.")
        return

    # Save generated audio under a per-step output directory.
    out_dir = os.path.join(
        config.experiment.output_dir, eval_audio_dirname, f"step_{global_step}_fixed"
    )
    os.makedirs(out_dir, exist_ok=True)

    eval_ds = T2SEvalDataset(ds_raw)
    loader = DataLoader(eval_ds, batch_size=config.training.batch_size_t2s)
    loader = accelerator.prepare(loader)

    local_results = []
    mask_token_id = unwrapped.config.mask_token_id
    mask_schedule = get_mask_schedule(config.training.get("mask_schedule", "cosine"))

    for batch in tqdm(
        loader, desc="T2S Fixed Evaluation", disable=not accelerator.is_main_process
    ):
        gt_texts = batch["gt_text"]
        sample_ids = batch["sample_id"]

        prompts = [f"{text}\n{random.choice(T2S_INSTRUCTION)}" for text in gt_texts]
        batch_size = len(prompts)

        speech_token_length = (
            config.dataset.preprocessing.max_aud_length - 2
        )  # Reserve tokens for SOA and EOA.
        audio_tokens = (
            torch.ones(
                (batch_size, speech_token_length),
                dtype=torch.long,
                device=accelerator.device,
            )
            * mask_token_id
        )
        input_ids, attention_mask = uni_prompting(
            (prompts, audio_tokens), "t2s_fixed_gen"
        )

        if config.training.guidance_scale > 0:
            uncond_input_ids, uncond_attention_mask = uni_prompting(
                ([""] * batch_size, audio_tokens), "t2s_fixed_gen"
            )
        else:
            uncond_input_ids, uncond_attention_mask = None, None

        # Generate fixed-length speech tokens.
        outputs = unwrapped.t2s_fixed_generate(
            input_ids=input_ids,
            uncond_input_ids=uncond_input_ids,
            attention_mask=attention_mask,
            uncond_attention_mask=uncond_attention_mask,
            guidance_scale=1.5,
            temperature=1.0,
            timesteps=64,
            noise_schedule=mask_schedule,
            noise_type="mask",
            seq_len=150,
            uni_prompting=uni_prompting,
            config=config,
        )

        # Decode generated tokens and run Whisper.
        for i in range(batch_size):
            gt = gt_texts[i].rsplit("\n", 1)[-1].strip()
            gen_rel = outputs[i]
            id_list = gen_rel.tolist()

            if not id_list:
                logger.warning(f"[fixed] Empty tokens for {sample_ids[i]}; skipping.")
                continue

            # Convert to the format expected by the speech tokenizer decoder.
            unit_str = " ".join(map(str, id_list))
            speech_unit_for_decode = "".join(
                [f"<|speech_{u}|>" for u in unit_str.split(" ")]
            )

            # Synthesize audio and transcribe with Whisper.
            fname = f"process_{accelerator.process_index}_{sample_ids[i]}_fixed.wav"
            wav_path = os.path.join(out_dir, fname)
            condition = "gender-female_emotion-neutral_speed-normal_pitch-normal"

            _ = vq_model_audio.decode(
                speech_unit_for_decode, condition=condition, output_wav_file=wav_path
            )
            asr = whisper_pipe(wav_path, generate_kwargs={"language": "english"})
            whisper_text = asr.get("text", "")

            local_results.append(
                {
                    "sample_id": sample_ids[i],
                    "gt_text": gt,
                    "whisper_text": whisper_text,
                    "audio_path": wav_path,
                }
            )

    # Gather results across processes and compute WER on the main process.
    all_res = accelerator.gather_for_metrics(local_results)
    if accelerator.is_main_process and all_res:
        gt_list = [r["gt_text"] for r in all_res]
        pred_list = [r["whisper_text"] for r in all_res]
        wer, errors, words = calculate_wer(pred_list, gt_list)
        logger.info(f"T2S Fixed WER: {wer:.4f} | Errors: {errors} | Words: {words}")

        accelerator.log(
            {
                "eval/t2s_fixed_wer": wer,
                "eval/t2s_fixed_errors": errors,
                "eval/t2s_fixed_words": words,
            },
            step=global_step,
        )

        # Log a small subset of samples to Weights & Biases.
        table = wandb.Table(columns=["ID", "GT", "ASR", "Audio"])
        for r in all_res[:8]:
            table.add_data(
                r["sample_id"],
                r["gt_text"],
                r["whisper_text"],
                wandb.Audio(r["audio_path"], caption=r["whisper_text"]),
            )
        accelerator.log({"eval/t2s_fixed_samples": table}, step=global_step)


# V2T evaluation.
@torch.no_grad()
def evaluate_v2t(
    model, vq_model_image, uni_prompting, config, accelerator, global_step
):
    # This qualitative evaluation runs only on the main process.
    if not accelerator.is_main_process:
        return

    logger.info("***** Running V2T Qualitative Evaluation *****")
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.eval()
    eval_cfg = _to_plain_dict(getattr(config, "evaluation", {}))
    video_root = eval_cfg.get("v2t_video_root")
    if not video_root or not os.path.exists(video_root):
        logger.warning(
            f"V2T eval root '{video_root}' not found. Skipping V2T evaluation."
        )
        return

    file_list = [f for f in os.listdir(video_root) if f.lower().endswith(".mp4")]
    if not file_list:
        logger.warning(
            f"No .mp4 files found in '{video_root}'. Skipping V2T evaluation."
        )
        return

    question = "Please provide a detailed description of the video."
    results_table = wandb.Table(columns=["Video ID", "Question", "Generated Caption"])

    for file_name in tqdm(
        file_list[:], desc="V2T Evaluation", disable=not accelerator.is_main_process
    ):
        video_path = os.path.join(video_root, file_name)

        # Load and process video.
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_frames - 1, 8, dtype=int)
        frames = []
        for i in range(total_frames):
            ret, frame = cap.read()
            if i in indices:
                if not ret:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame)
                frames.append(
                    image_transform(
                        pil_img, resolution=config.dataset.preprocessing.resolution
                    )
                )
        cap.release()

        if len(frames) < 8:
            continue

        video_tensor = torch.stack(frames).to(accelerator.device)
        video_tokens = vq_model_image.get_code(video_tensor) + len(
            uni_prompting.text_tokenizer
        )
        video_tokens = video_tokens.view(1, -1)

        sptids = uni_prompting.sptids_dict
        device = unwrapped_model.device

        prompt_text = f"<|start_header_id|>user<|end_header_id|>\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        prompt_tensor = uni_prompting.text_tokenizer(
            prompt_text, return_tensors="pt"
        ).input_ids.to(device)

        input_ids = torch.cat(
            [
                sptids["<|v2t|>"].to(device).unsqueeze(0),
                sptids["<|soi|>"].to(device).unsqueeze(0),
                video_tokens,
                sptids["<|eoi|>"].to(device).unsqueeze(0),
                sptids["<|sot|>"].to(device).unsqueeze(0),
                prompt_tensor,
            ],
            dim=1,
        ).long()

        output_ids = unwrapped_model.mmu_generate(
            input_ids, max_new_tokens=256, steps=256, block_length=128
        )
        text = uni_prompting.text_tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1] :], skip_special_tokens=True
        )[0]
        results_table.add_data(file_name, question, text)

    accelerator.log({"eval/v2t_qualitative_samples": results_table}, step=global_step)


# Evaluation orchestration.


def run_evaluation(
    model,
    vq_model_image,
    vq_model_audio,
    uni_prompting,
    config,
    accelerator,
    global_step,
):
    """
    Orchestrates the S2T, T2S, and V2T evaluations.
    """
    logger.info(f"--- Starting evaluation at step {global_step} ---")
    model.eval()

    evaluate_s2t(model, vq_model_audio, uni_prompting, config, accelerator, global_step)

    evaluate_t2s(model, vq_model_audio, uni_prompting, config, accelerator, global_step)
    evaluate_t2s_fixed(
        model, vq_model_audio, uni_prompting, config, accelerator, global_step
    )

    evaluate_v2t(model, vq_model_image, uni_prompting, config, accelerator, global_step)

    accelerator.wait_for_everyone()
    logger.info(
        f"--- Finished evaluation at step {global_step}. Returning to training. ---"
    )
    model.train()


def main():
    # Set up accelerator.
    config = get_config()
    _sanitize_experiment_intervals(config)
    apply_dataset_sources(config)
    hf_cache_root = _configure_hf_cache_env(config)
    bootstrap_logger.info("HF cache root: %s", hf_cache_root)

    # Enable TF32 on Ampere GPUs.
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.output_dir) / "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
        split_batches=True,
    )

    total_batch_size_per_gpu = (
        config.training.batch_size_t2i
        + config.training.batch_size_lm
        + config.training.batch_size_mmu
        + config.training.batch_size_v2t
        + config.training.batch_size_s2t
        + config.training.batch_size_t2s
    )
    total_batch_size = (
        (
            config.training.batch_size_t2i
            + config.training.batch_size_lm
            + config.training.batch_size_mmu
            + config.training.batch_size_v2t
            + config.training.batch_size_s2t
            + config.training.batch_size_t2s
        )
        * accelerator.num_processes
        * config.training.gradient_accumulation_steps
    )

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config[
            "train_micro_batch_size_per_gpu"
        ] = total_batch_size_per_gpu

    # Set up logging, seed, and config.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    # Initialize trackers and store config.
    if accelerator.is_main_process:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id

        wandb_init_kwargs = dict(
            name=config.experiment.name,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
            dir=config.experiment.logging_dir,
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint")

        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    if accelerator.is_main_process:
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        config_path = Path(config.experiment.output_dir) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    # Set the training seed.
    if config.training.seed is not None:
        set_seed(config.training.seed)

    # Load models and optimizer.
    logger.info("Loading models and optimizer")

    model_local_files_only = resolve_model_local_files_only(config, default=False)
    if _is_env_truthy("HF_HUB_OFFLINE") or _is_env_truthy("TRANSFORMERS_OFFLINE"):
        model_local_files_only = True

    tokenizer_source, model_source, vq_image_source, vq_audio_source = (
        _resolve_stage1_model_sources(config)
    )
    logger.info(
        "Model sources | tokenizer=%s | model=%s | vq_image=%s | vq_audio=%s | local_files_only=%s",
        tokenizer_source,
        model_source,
        vq_image_source,
        vq_audio_source,
        model_local_files_only,
    )
    tokenizer_kwargs = {"padding_side": "left"}
    tokenizer_kwargs["trust_remote_code"] = True
    if model_local_files_only:
        tokenizer_kwargs["local_files_only"] = True
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        **tokenizer_kwargs,
    )

    uni_prompting = UniversalPrompting(
        tokenizer,
        max_text_len=config.dataset.preprocessing.max_seq_length,
        max_audio_len=config.dataset.preprocessing.max_aud_length,
        special_tokens=(
            "<|soi|>",
            "<|eoi|>",
            "<|sov|>",
            "<|eov|>",
            "<|t2i|>",
            "<|mmu|>",
            "<|t2v|>",
            "<|v2v|>",
            "<|lvg|>",
            # Dynin-Omni special tokens.
            "<|v2t|>",
            "<|s2t|>",
            "<|t2s|>",
            "<|soa|>",
            "<|eoa|>",
        ),
        ignore_id=-100,
        cond_dropout_prob=config.training.cond_dropout_prob,
        use_reserved_token=True,
    )

    logger.info("Loaded %d special tokens.", len(uni_prompting.sptids_dict))

    speech_vocab_start = len(uni_prompting.text_tokenizer) + int(
        config.model.dynin_omni.codebook_size
    )
    audio_codebook_size = max(
        int(config.model.dynin_omni.new_vocab_size) - speech_vocab_start, 0
    )
    t2s_special_token_ids = {
        "eoa": int(uni_prompting.sptids_dict["<|eoa|>"][0].item()),
        "eos": int(uni_prompting.text_tokenizer.eos_token_id),
    }

    # Load VQ models.
    vq_model_image = get_vq_model_class(config.model.vq_model_image.type)
    if config.model.vq_model_image.get("pretrained_model_path", None):
        vq_model_image = vq_model_image()
        state_dict = torch.load(config.model.vq_model_image.pretrained_model_path)[
            "model"
        ]
        vq_model_image.load_state_dict(state_dict)
    else:
        vq_model_image = vq_model_image.from_pretrained(
            vq_image_source,
            local_files_only=model_local_files_only,
        )

    vq_model_audio = get_vq_model_class(config.model.vq_model_audio.type)
    vq_model_audio = vq_model_audio.from_pretrained(
        vq_audio_source,
        local_files_only=model_local_files_only,
    )

    vq_model_image.eval()
    vq_model_image.requires_grad_(False)

    vq_model_audio.eval()
    vq_model_audio.requires_grad_(False)

    model_load_kwargs = {"torch_dtype": torch.bfloat16, "trust_remote_code": True}
    if model_local_files_only:
        model_load_kwargs["local_files_only"] = True
    model = DyninOmniModelLM.from_pretrained(
        model_source,
        **model_load_kwargs,
    )

    # Resize vocabulary for audio modality.
    unwrapped_model = accelerator.unwrap_model(model)
    original_vocab_size = unwrapped_model.get_input_embeddings().weight.shape[0]
    logger.info("Calling resize_vocab.")
    logger.info("Vocab size before resizing: %d", original_vocab_size)

    resize_vocab(unwrapped_model, config)

    resized_vocab_size = unwrapped_model.get_input_embeddings().weight.shape[0]
    logger.info("Vocab size after resizing: %d", resized_vocab_size)
    logger.info("Configured new_vocab_size: %d", config.model.dynin_omni.new_vocab_size)

    if resized_vocab_size == config.model.dynin_omni.new_vocab_size:
        logger.info("Vocab resize successful.")
    else:
        logger.info("Vocab resize failed or did not match config.")
    mask_id = model.config.mask_token_id

    # Create optimizer and scheduler.
    optimizer_config = config.optimizer.params

    # Apply no weight decay to bias, layer norm, and embedding weights.
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    # Create mask scheduler.
    if config.get("mask_schedule", None) is not None:
        schedule = config.mask_schedule.schedule
        args = config.mask_schedule.get("params", {})
        mask_schedule = get_mask_schedule(schedule, **args)
    else:
        mask_schedule = get_mask_schedule(
            config.training.get("mask_schedule", "cosine")
        )

    # Create dataloaders.
    logger.info("Creating dataloaders and lr_scheduler")

    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params
    batch_size_v2t_cfg = int(config.training.batch_size_v2t)
    batch_size_s2t_cfg = int(config.training.batch_size_s2t)
    batch_size_t2s_cfg = int(config.training.batch_size_t2s)

    if (
        batch_size_v2t_cfg < 0
        or batch_size_s2t_cfg < 0
        or batch_size_t2s_cfg < 0
    ):
        raise ValueError("Batch sizes for v2t/s2t/t2s must be >= 0.")
    if batch_size_v2t_cfg + batch_size_s2t_cfg + batch_size_t2s_cfg == 0:
        raise ValueError("At least one of batch_size_v2t/s2t/t2s must be > 0.")

    video_captioning_dataset = None
    sampler_v2t = None
    train_dataloader_v2t = None
    if batch_size_v2t_cfg > 0:
        video_cfg = _to_plain_dict(dataset_config.get("video_caption_dataset", {}))
        video_dataset_kwargs = {
            "transform": image_transform,
            "tokenizer": uni_prompting.text_tokenizer,
            "max_seq_length": preproc_config.max_seq_length,
            "resolution": int(video_cfg.get("resolution", preproc_config.resolution)),
            "sample_method": video_cfg.get("sample_method", "uniform"),
            "num_frames": int(video_cfg.get("num_frames", 8)),
            "dataset_name": video_cfg.get("dataset_name", "webvid10m"),
        }

        for optional_key in (
            "openvid1m_path",
            "webvid10m_path",
            "llavavid_path",
            "llavavid_local_files_only",
            "llavavid_skip_configs",
            "llavavid_skip_video_patterns",
            "llavavid_max_samples",
            "llavavid_sample_seed",
            "max_video_seconds",
        ):
            if optional_key in video_cfg and video_cfg[optional_key] is not None:
                video_dataset_kwargs[optional_key] = video_cfg[optional_key]

        video_captioning_dataset = VideoCaptionDataset(**video_dataset_kwargs)
        sampler_v2t = (
            DistributedSampler(
                video_captioning_dataset,
                num_replicas=accelerator.num_processes,
                rank=accelerator.process_index,
                shuffle=True,
                drop_last=True,
            )
            if accelerator.num_processes > 1
            else None
        )

        train_dataloader_v2t = DataLoader(
            video_captioning_dataset,
            batch_size=batch_size_v2t_cfg,
            num_workers=dataset_config.num_workers,
            collate_fn=collate_fn_video_caption,
            sampler=sampler_v2t,
            shuffle=sampler_v2t is None,
        )

    dataset_sm = None
    sampler_s2t = None
    sampler_t2s = None
    train_dataloader_s2t = None
    train_dataloader_t2s = None
    speech_enabled = batch_size_s2t_cfg > 0 or batch_size_t2s_cfg > 0
    if speech_enabled:
        dataset_sm = MixedSpeechTextDataset(config.dataset.params.audio_data)

        if batch_size_s2t_cfg > 0:
            sampler_s2t = (
                DistributedSampler(
                    dataset_sm,
                    num_replicas=accelerator.num_processes,
                    rank=accelerator.process_index,
                    shuffle=True,
                )
                if accelerator.num_processes > 1
                else None
            )
            train_dataloader_s2t = DataLoader(
                dataset_sm,
                batch_size=batch_size_s2t_cfg,
                shuffle=False,
                sampler=sampler_s2t,
                collate_fn=collate_fn_audio,
                num_workers=config.dataset.params.num_workers,
            )

        if batch_size_t2s_cfg > 0:
            sampler_t2s = (
                DistributedSampler(
                    dataset_sm,
                    num_replicas=accelerator.num_processes,
                    rank=accelerator.process_index,
                    shuffle=True,
                )
                if accelerator.num_processes > 1
                else None
            )
            train_dataloader_t2s = DataLoader(
                dataset_sm,
                batch_size=batch_size_t2s_cfg,
                shuffle=False,
                sampler=sampler_t2s,
                collate_fn=collate_fn_audio,
                num_workers=config.dataset.params.num_workers,
            )

    logger.info("Dataset prepared.")

    # Combine dataloaders into one iterable.
    iterables = {}
    if train_dataloader_v2t is not None:
        iterables["v2t_flow"] = train_dataloader_v2t
    if train_dataloader_t2s is not None:
        iterables["t2s_flow"] = train_dataloader_t2s
    if train_dataloader_s2t is not None:
        iterables["s2t_flow"] = train_dataloader_s2t
    if not iterables:
        raise ValueError("No active dataloaders were created for stage1 training.")

    combined_dataloader = CombinedLoader(
        iterables, mode=config.dataset.combined_loader_mode
    )

    def _num_steps(dataset_obj, batch_size_cfg: int) -> int:
        if dataset_obj is None or batch_size_cfg <= 0:
            return 0
        total_bs = (
            batch_size_cfg
            * accelerator.num_processes
            * config.training.gradient_accumulation_steps
        )
        if total_bs <= 0:
            return 0
        return math.ceil(len(dataset_obj) / total_bs)

    num_update_steps_per_epoch_v2t = _num_steps(video_captioning_dataset, batch_size_v2t_cfg)
    num_update_steps_per_epoch_s2t = _num_steps(dataset_sm, batch_size_s2t_cfg)
    num_update_steps_per_epoch_t2s = _num_steps(dataset_sm, batch_size_t2s_cfg)
    active_num_steps = [
        step
        for step in (
            num_update_steps_per_epoch_v2t,
            num_update_steps_per_epoch_s2t,
            num_update_steps_per_epoch_t2s,
        )
        if step > 0
    ]
    if not active_num_steps:
        raise ValueError("No train steps available; verify dataset paths and batch sizes.")

    # Calculate training epochs.
    num_update_steps_per_epoch = max(active_num_steps)
    num_train_epochs = (
        math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)
        if num_update_steps_per_epoch > 0
        else 1
    )

    if dataset_sm is not None:
        logger.info(f"len of speech: {len(dataset_sm)}")
    if video_captioning_dataset is not None:
        logger.info(f"len of video: {len(video_captioning_dataset)}")
    logger.info(f"Train steps: {config.training.max_train_steps}")
    logger.info(f"Num train epochs: {num_train_epochs}")

    # Restore model from checkpoint if configured.
    global_step = 0
    first_epoch = 0
    start_step = 0

    if config.experiment.resume_from_checkpoint:
        dirs = os.listdir(config.experiment.output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None
        if path is not None:
            path = os.path.join(config.experiment.output_dir, path)
            logger.info(f"Resuming from checkpoint: {path}")
            global_step = start_step = int(os.path.basename(path).split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
            if os.path.exists(f"{path}/unwrapped_model/pytorch_model.bin"):
                state_dict = torch.load(
                    f"{path}/unwrapped_model/pytorch_model.bin", map_location="cpu"
                )
                model.load_state_dict(state_dict, strict=True)
                del state_dict
            elif os.path.exists(f"{path}/unwrapped_model/pytorch_model.bin.index.json"):
                from safetensors.torch import load_file
                from transformers.modeling_utils import load_sharded_checkpoint

                load_sharded_checkpoint(model, f"{path}/unwrapped_model/")
            elif os.path.exists(f"{path}/unwrapped_model/model.safetensors.index.json"):
                from transformers.modeling_utils import load_sharded_checkpoint

                load_sharded_checkpoint(
                    model,
                    f"{path}/unwrapped_model/",
                )
            else:
                raise FileNotFoundError(
                    f"Checkpoint {path}/unwrapped_model/pytorch_model.bin or safetensors not found"
                )
    else:
        logger.info("Not resuming from checkpoint")

    # Prepare accelerator objects.
    logger.info("Preparing model, optimizer and dataloaders")

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
        min_lr_scale=config.lr_scheduler.params.min_lr_scale,
    )

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    vq_model_image.to(device=accelerator.device)
    vq_model_audio.to(device=accelerator.device)

    mask_dtype = model.get_input_embeddings().weight.dtype

    # Start training.
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {total_batch_size_per_gpu}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}"
    )
    log_every = config.experiment.get("log_every", None)
    save_every = config.experiment.get("save_every", None)
    eval_every = config.experiment.get("eval_every", None)

    @torch.no_grad()
    def prepare_inputs_and_labels(
        pixel_values_or_image_ids: Union[torch.FloatTensor, torch.LongTensor],
        texts: Union[str, str],
        min_masking_rate: float = 0.0,
        is_train: bool = True,
        seed: int = None,
    ):

        image_tokens = vq_model_image.get_code(pixel_values_or_image_ids)
        image_tokens = image_tokens + len(uni_prompting.text_tokenizer)
        # Create MLM mask and labels.
        input_ids, labels, loss_weight, mask_prob = mask_or_random_replace_tokens(
            image_tokens,
            mask_id,
            config,
            mask_schedule=mask_schedule,
            is_train=is_train,
        )
        input_ids, masks, labels = uni_prompting((texts, input_ids, labels), "t2i")
        return input_ids, labels, mask_prob, image_tokens, masks

    # Video also uses this.
    @torch.no_grad()
    def prepare_inputs_and_labels_for_mmu(
        input_ids_mmu, prompt_masks, labels_mmu, eps=1e-3
    ):
        b, l = input_ids_mmu.shape
        t = torch.rand(b, device=input_ids_mmu.device)
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, l)

        masked_indices = torch.rand((b, l), device=input_ids_mmu.device) < p_mask
        # 126336 is used for [MASK] token
        noisy_batch = torch.where(masked_indices, mask_id, input_ids_mmu)
        masked_indices = noisy_batch == mask_id
        noisy_batch[prompt_masks.bool()] = input_ids_mmu[prompt_masks.bool()]
        masked_indices = noisy_batch == mask_id

        prompt_masks = prompt_masks.to(torch.int64)
        answer_lengths = torch.sum((1 - prompt_masks), dim=-1, keepdim=True)
        answer_lengths = answer_lengths.repeat(1, noisy_batch.shape[1])

        return noisy_batch, labels_mmu, p_mask, answer_lengths

    @torch.no_grad()
    def prepare_inputs_and_labels_for_t2s(
        input_ids_t2s, prompt_masks, labels_t2s, mask_id=126336, eps=1e-3
    ):
        b, l = input_ids_t2s.shape
        t = torch.rand(b, device=input_ids_t2s.device)
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, l)

        masked_indices = torch.rand((b, l), device=input_ids_t2s.device) < p_mask
        noisy_batch = torch.where(masked_indices, mask_id, input_ids_t2s)
        masked_indices = noisy_batch == mask_id

        noisy_batch[prompt_masks.bool()] = input_ids_t2s[prompt_masks.bool()]
        masked_indices = noisy_batch == mask_id

        prompt_masks = prompt_masks.to(torch.int64)
        answer_lengths = torch.sum((1 - prompt_masks), dim=-1, keepdim=True)
        answer_lengths = answer_lengths.repeat(1, noisy_batch.shape[1])

        return noisy_batch, labels_t2s, p_mask, answer_lengths

    @torch.no_grad()
    def prepare_inputs_and_labels_for_s2t(
        input_ids_mmu, prompt_masks, labels_mmu, eps=1e-3
    ):
        b, l = input_ids_mmu.shape
        t = torch.rand(b, device=input_ids_mmu.device)
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, l)

        masked_indices = torch.rand((b, l), device=input_ids_mmu.device) < p_mask
        # 126336 is used for [MASK] token
        noisy_batch = torch.where(masked_indices, mask_id, input_ids_mmu)
        masked_indices = noisy_batch == mask_id
        noisy_batch[prompt_masks.bool()] = input_ids_mmu[prompt_masks.bool()]
        masked_indices = noisy_batch == mask_id

        prompt_masks = prompt_masks.to(torch.int64)
        answer_lengths = torch.sum((1 - prompt_masks), dim=-1, keepdim=True)
        answer_lengths = answer_lengths.repeat(1, noisy_batch.shape[1])

        return noisy_batch, labels_mmu, p_mask, answer_lengths

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for epoch in tqdm(
        range(first_epoch, num_train_epochs),
        desc="Epochs",
        disable=not accelerator.is_main_process,
        position=0,
    ):
        model.train()
        if sampler_v2t is not None:
            sampler_v2t.set_epoch(epoch)
        if sampler_s2t is not None:
            sampler_s2t.set_epoch(epoch)
        if sampler_t2s is not None:
            sampler_t2s.set_epoch(epoch)

        for batch, batch_idx, dataloader_idx in combined_dataloader:
            batch_size_t2i = 0
            batch_size_lm = 0
            batch_size_mmu = 0

            if batch is None:
                logger.warning(f"Skipping step {global_step} (batch is None)")
                continue

            v2t_flow = batch.get("v2t_flow")
            s2t_flow = batch.get("s2t_flow")
            t2s_flow = batch.get("t2s_flow")

            video_tensor = v2t_flow.get("video") if isinstance(v2t_flow, dict) else None
            texts_vid = v2t_flow.get("captions", []) if isinstance(v2t_flow, dict) else []
            batch_size_v2t = (
                int(video_tensor.shape[0]) if isinstance(video_tensor, torch.Tensor) else 0
            )

            audio_paths_s2t = (
                s2t_flow.get("audio_path", []) if isinstance(s2t_flow, dict) else []
            )
            texts_s2t = s2t_flow.get("text", []) if isinstance(s2t_flow, dict) else []
            batch_size_s2t = len(audio_paths_s2t)

            audio_paths_t2s = (
                t2s_flow.get("audio_path", []) if isinstance(t2s_flow, dict) else []
            )
            texts_t2s = t2s_flow.get("text", []) if isinstance(t2s_flow, dict) else []
            batch_size_t2s = len(audio_paths_t2s)

            # Avoid peak-memory spikes by alternating speech tasks when both are present.
            if batch_size_s2t > 0 and batch_size_t2s > 0:
                drop_t2s = (global_step % 2) == 0
                if drop_t2s:
                    audio_paths_t2s = []
                    texts_t2s = []
                    batch_size_t2s = 0
                else:
                    audio_paths_s2t = []
                    texts_s2t = []
                    batch_size_s2t = 0

            if batch_size_v2t + batch_size_s2t + batch_size_t2s == 0:
                logger.warning(f"Skipping step {global_step} (no active flow in batch)")
                continue

            logger.info(
                "Batch sizes v2t=%d, t2s=%d, s2t=%d",
                batch_size_v2t,
                batch_size_t2s,
                batch_size_s2t,
            )

            offset = speech_vocab_start
            device = accelerator.device

            data_time_m.update(time.time() - end)

            input_ids_vid = torch.empty((0, 1), dtype=torch.long, device=device)
            labels_vid = torch.empty((0, 1), dtype=torch.long, device=device)
            p_mask_vid = torch.empty((0, 1), dtype=torch.float32, device=device)
            answer_lengths_vid = torch.empty((0, 1), dtype=torch.long, device=device)

            if batch_size_v2t > 0:
                video_token_list = []
                for video in video_tensor:
                    # Keep full video batch on CPU and move one sample at a time.
                    video_sample = video.to(device, non_blocking=True)
                    video_token = vq_model_image.get_code(video_sample)
                    video_token = video_token + len(uni_prompting.text_tokenizer)
                    video_token_list.append(video_token.view(-1).cpu())

                if video_token_list:
                    video_tokens = torch.stack(video_token_list, dim=0)
                    input_ids_vid, prompt_masks_vid, labels_vid = uni_prompting(
                        (video_tokens, texts_vid), "v2t"
                    )
                    (input_ids_vid, labels_vid, p_mask_vid, answer_lengths_vid) = (
                        prepare_inputs_and_labels_for_mmu(
                            input_ids_vid, prompt_masks_vid, labels_vid
                        )
                    )
                    input_ids_vid = input_ids_vid.to(device, non_blocking=True)
                    labels_vid = labels_vid.to(device, non_blocking=True)
                    p_mask_vid = p_mask_vid.to(device, non_blocking=True)
                    answer_lengths_vid = answer_lengths_vid.to(device, non_blocking=True)
                else:
                    batch_size_v2t = 0

            input_ids_s2t = torch.empty((0, 1), dtype=torch.long, device=device)
            labels_s2t = torch.empty((0, 1), dtype=torch.long, device=device)
            p_mask_s2t = torch.empty((0, 1), dtype=torch.float32, device=device)
            answer_lengths_s2t = torch.empty((0, 1), dtype=torch.long, device=device)

            if batch_size_s2t > 0:
                prompt_s2t = [
                    "<|start_header_id|>user<|end_header_id|>\n"
                    + prompt
                    + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
                    for prompt in S2T_INSTRUCTION
                ]

                all_audio_tokens = []
                for path in audio_paths_s2t:
                    tokens = vq_model_audio.encode(path).to(device, non_blocking=True)
                    all_audio_tokens.append(tokens + offset)

                prompt = random.choice(prompt_s2t)
                texts_with_prompt = [f"{prompt}{text}" for text in texts_s2t]

                input_ids_s2t, prompt_masks_s2t, labels_s2t = uni_prompting(
                    (all_audio_tokens, texts_with_prompt), "s2t"
                )
                input_ids_s2t, labels_s2t, p_mask_s2t, answer_lengths_s2t = (
                    prepare_inputs_and_labels_for_s2t(
                        input_ids_s2t, prompt_masks_s2t, labels_s2t
                    )
                )
                input_ids_s2t = input_ids_s2t.to(device, non_blocking=True)
                labels_s2t = labels_s2t.to(device, non_blocking=True)
                p_mask_s2t = p_mask_s2t.to(device, non_blocking=True)
                answer_lengths_s2t = answer_lengths_s2t.to(device, non_blocking=True)

            input_ids_t2s = torch.empty((0, 1), dtype=torch.long, device=device)
            labels_t2s = torch.empty((0, 1), dtype=torch.long, device=device)
            p_mask_t2s = torch.empty((0, 1), dtype=torch.float32, device=device)
            answer_lengths_t2s = torch.empty((0, 1), dtype=torch.long, device=device)

            if batch_size_t2s > 0:
                prompt_t2s = [prompt for prompt in T2S_INSTRUCTION]

                all_audio_tokens = []
                for path in audio_paths_t2s:
                    tokens = vq_model_audio.encode(path).to(device, non_blocking=True)
                    all_audio_tokens.append(tokens + offset)

                prompt = random.choice(prompt_t2s)
                texts_with_prompt = [f"{text}\n{prompt}" for text in texts_t2s]

                input_ids_t2s, prompt_masks_t2s, labels_t2s = uni_prompting(
                    (texts_with_prompt, all_audio_tokens), "t2s"
                )
                input_ids_t2s, labels_t2s, p_mask_t2s, answer_lengths_t2s = (
                    prepare_inputs_and_labels_for_t2s(
                        input_ids_t2s, prompt_masks_t2s, labels_t2s
                    )
                )
                input_ids_t2s = input_ids_t2s.to(device, non_blocking=True)
                labels_t2s = labels_t2s.to(device, non_blocking=True)
                p_mask_t2s = p_mask_t2s.to(device, non_blocking=True)
                answer_lengths_t2s = answer_lengths_t2s.to(device, non_blocking=True)

            active_inputs = []
            if batch_size_v2t > 0:
                active_inputs.append(input_ids_vid)
            if batch_size_s2t > 0:
                active_inputs.append(input_ids_s2t)
            if batch_size_t2s > 0:
                active_inputs.append(input_ids_t2s)
            if not active_inputs:
                logger.warning(f"Skipping step {global_step} (all active flows empty)")
                continue

            # Pad all task tensors to the same sequence length.
            pad_token_id = uni_prompting.text_tokenizer.eos_token_id
            max_len = max(tensor.shape[1] for tensor in active_inputs)

            if batch_size_v2t > 0:
                input_ids_vid = pad_tensor(input_ids_vid, max_len, pad_token_id)
                labels_vid = pad_tensor(labels_vid, max_len, -100)
                p_mask_vid = pad_tensor(p_mask_vid, max_len, 1.0)
                answer_lengths_vid = pad_answer_lengths(answer_lengths_vid, max_len)
            else:
                input_ids_vid = torch.empty((0, max_len), dtype=torch.long, device=device)
                labels_vid = torch.empty((0, max_len), dtype=torch.long, device=device)
                p_mask_vid = torch.empty((0, max_len), dtype=torch.float32, device=device)
                answer_lengths_vid = torch.empty((0, max_len), dtype=torch.long, device=device)

            if batch_size_s2t > 0:
                input_ids_s2t = pad_tensor(input_ids_s2t, max_len, pad_token_id)
                labels_s2t = pad_tensor(labels_s2t, max_len, -100)
                p_mask_s2t = pad_tensor(p_mask_s2t, max_len, 1.0)
                answer_lengths_s2t = pad_answer_lengths(answer_lengths_s2t, max_len)
            else:
                input_ids_s2t = torch.empty((0, max_len), dtype=torch.long, device=device)
                labels_s2t = torch.empty((0, max_len), dtype=torch.long, device=device)
                p_mask_s2t = torch.empty((0, max_len), dtype=torch.float32, device=device)
                answer_lengths_s2t = torch.empty((0, max_len), dtype=torch.long, device=device)

            if batch_size_t2s > 0:
                input_ids_t2s = pad_tensor(input_ids_t2s, max_len, pad_token_id)
                labels_t2s = pad_tensor(labels_t2s, max_len, -100)
                p_mask_t2s = pad_tensor(p_mask_t2s, max_len, 1.0)
                answer_lengths_t2s = pad_answer_lengths(answer_lengths_t2s, max_len)
            else:
                input_ids_t2s = torch.empty((0, max_len), dtype=torch.long, device=device)
                labels_t2s = torch.empty((0, max_len), dtype=torch.long, device=device)
                p_mask_t2s = torch.empty((0, max_len), dtype=torch.float32, device=device)
                answer_lengths_t2s = torch.empty((0, max_len), dtype=torch.long, device=device)

            input_ids = torch.cat((input_ids_vid, input_ids_s2t, input_ids_t2s), dim=0)
            labels = torch.cat((labels_vid, labels_s2t, labels_t2s), dim=0)

            p_mask_lm = None
            p_mask_mmu = None
            answer_lengths_mmu = None
            t2i_masks = None

            if global_step == 0 and epoch == 0:
                logger.info("Input ids: {}".format(input_ids))
                logger.info("Input ids shape: {}".format(input_ids.shape))
                logger.info("Labels: {}".format(labels))

            (
                logits,
                _loss_t2i,
                _loss_i2i,
                _loss_ti2ti,
                _loss_lm,
                _loss_mmu,
                loss_vid,
                _loss_v2s,
                loss_s2t,
                _loss_s2s,
                loss_t2s,
            ) = accelerator.unwrap_model(model).forward_process(
                input_ids=input_ids,
                labels=labels,
                batch_size_t2i=batch_size_t2i,
                batch_size_lm=batch_size_lm,
                batch_size_mmu=batch_size_mmu,
                batch_size_v2t=batch_size_v2t,
                batch_size_s2t=batch_size_s2t,
                batch_size_t2s=batch_size_t2s,
                max_seq_length=config.dataset.preprocessing.max_seq_length,
                p_mask_lm=p_mask_lm,
                p_mask_mmu=p_mask_mmu,
                p_mask_vid=p_mask_vid,
                p_mask_s2t=p_mask_s2t,
                p_mask_t2s=p_mask_t2s,
                answer_lengths_mmu=answer_lengths_mmu,
                answer_lengths_vid=answer_lengths_vid,
                answer_lengths_s2t=answer_lengths_s2t,
                answer_lengths_t2s=answer_lengths_t2s,
                t2i_masks=t2i_masks,
                t2s_vocab_start=speech_vocab_start,
                t2s_codebook_size=audio_codebook_size,
                t2s_special_token_ids=t2s_special_token_ids,
                text_vocab_size_override=len(uni_prompting.text_tokenizer),
            )

            avg_loss_vid = (
                accelerator.gather(loss_vid.repeat(batch_size_v2t)).mean()
                if batch_size_v2t > 0
                else loss_vid.detach().new_zeros(())
            )
            avg_loss_s2t = (
                accelerator.gather(loss_s2t.repeat(batch_size_s2t)).mean()
                if batch_size_s2t > 0
                else loss_s2t.detach().new_zeros(())
            )
            avg_loss_t2s = (
                accelerator.gather(loss_t2s.repeat(batch_size_t2s)).mean()
                if batch_size_t2s > 0
                else loss_t2s.detach().new_zeros(())
            )

            loss = (
                config.training.v2t_coeff * loss_vid
                + config.training.s2t_coeff * loss_s2t
                + config.training.t2s_coeff * loss_t2s
            )

            accelerator.backward(loss)

            if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    model.parameters(), config.training.max_grad_norm
                )

            optimizer.step()
            lr_scheduler.step()

            if (
                accelerator.sync_gradients
                and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                and accelerator.is_main_process
            ):
                log_grad_norm(model, accelerator, global_step + 1)

            optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                batch_time_m.update(time.time() - end)
                end = time.time()

                if log_every is not None and (global_step + 1) % log_every == 0:
                    samples_per_second_per_gpu = (
                        config.training.gradient_accumulation_steps
                        * total_batch_size_per_gpu
                        / batch_time_m.val
                    )
                    logs = {
                        "step_loss_vid": avg_loss_vid.item(),
                        "step_loss_s2t": avg_loss_s2t.item(),
                        "step_loss_t2s": avg_loss_t2s.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "samples/sec/gpu": samples_per_second_per_gpu,
                        "data_time": data_time_m.val,
                        "batch_time": batch_time_m.val,
                    }
                    accelerator.log(logs, step=global_step + 1)

                    logger.info(
                        f"Step: {global_step + 1} "
                        f"Loss_vid: {avg_loss_vid.item():0.4f} "
                        f"Loss_s2t: {avg_loss_s2t.item():0.4f} "
                        f"Loss_t2s: {avg_loss_t2s.item():0.4f} "
                        f"Data (t): {data_time_m.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                        f"Batch (t): {batch_time_m.val:0.4f} "
                        f"LR: {lr_scheduler.get_last_lr()[0]:0.6f}"
                    )

                    batch_time_m.reset()
                    data_time_m.reset()

                if save_every is not None and (global_step + 1) % save_every == 0:
                    save_checkpoint(
                        model, config, accelerator, global_step + 1, uni_prompting
                    )

                if eval_every is not None and (
                    global_step == 0 or (global_step + 1) % eval_every == 0
                ):
                    run_evaluation(
                        model=accelerator.unwrap_model(model),
                        vq_model_image=vq_model_image,
                        vq_model_audio=vq_model_audio,
                        uni_prompting=uni_prompting,
                        config=config,
                        accelerator=accelerator,
                        global_step=global_step + 1,
                    )

                global_step += 1

            if global_step >= config.training.max_train_steps:
                break

        if global_step >= config.training.max_train_steps:
            break

    accelerator.wait_for_everyone()

    save_checkpoint(model, config, accelerator, global_step, uni_prompting)

    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        model.save_pretrained(config.experiment.output_dir, safe_serialization=True)

    accelerator.end_training()


@torch.no_grad()
def visualize_predictions(*args, **kwargs):
    # This function is kept for compatibility.
    pass


@torch.no_grad()
def generate_images(*args, **kwargs):
    # This function is kept for compatibility.
    pass


@torch.no_grad()
def understanding_images(*args, **kwargs):
    # This function is kept for compatibility.
    pass


def save_checkpoint(model, config, accelerator, global_step, uni_prompting):
    output_dir = config.experiment.output_dir
    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    # Check the total number of checkpoints before saving.
    if accelerator.is_main_process and checkpoints_total_limit is not None:
        checkpoints = os.listdir(output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        # Keep at most checkpoints_total_limit checkpoints after the new save.
        if len(checkpoints) >= checkpoints_total_limit:
            num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

    save_path = Path(output_dir) / f"checkpoint-{global_step}"

    # Collect state dict across processes, then save on the main process.
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            save_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=True,
        )
        json.dump(
            {"global_step": global_step}, (save_path / "metadata.json").open("w+")
        )
        logger.info(f"Saved state to {save_path}")

        # Save tokenizer together with model checkpoint.
        uni_prompting.text_tokenizer.save_pretrained(save_path / "unwrapped_model")


def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)


if __name__ == "__main__":
    main()
