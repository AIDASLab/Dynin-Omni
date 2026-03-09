# coding=utf-8
# Copyright 2026 Dynin-Omni Team, AIDAS Lab, Seoul National University
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

import hashlib
import os
import sys
import warnings
import subprocess
import tempfile

os.environ["FFMPEG_LOG_LEVEL"] = "error"
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import json
import logging
import math
import torch.nn.functional as F
import shutil
import time
import cv2
import glob
import random
from itertools import zip_longest
import contextlib
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Union, Dict, Any, List, Iterator
from collections.abc import Sequence
import csv
import numpy as np
from PIL import Image
from omegaconf import OmegaConf, DictConfig, ListConfig
import wandb
import torch
from torch.optim import AdamW
from lightning.pytorch.utilities import CombinedLoader
import multiprocessing as py_mp
import torch.multiprocessing as mp

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    warnings.filterwarnings("ignore", category=FutureWarning)

from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed

# I2I-specific imports.
from datasets import load_dataset as hf_load_dataset
from torch.utils.data import Dataset, DataLoader, DistributedSampler, ConcatDataset

# Omni-modal-specific imports.
from models.modeling_emova_speech_tokenizer import EMOVASpeechTokenizer
from training.data import (
    SpeechTextDataset,
    MixedSpeechTextDataset,
    TextImageInterleavedDataset,
    load_video_mp4,
    VideoCaptionDataset,
    ShareGPTVideoSFTDataset,
    S2T_INSTRUCTION,
    T2S_INSTRUCTION,
    ReasoningSFTCSVDataset,
)

from training.data import (
    Text2ImageDataset,
    HQEditX2IDataset,
    CombinedX2IDataset,
    HFInstructionTextDataset,
    PickAPicV2Dataset,
    FluxReasonDataset,
    UltraEditDataset,
    JourneyDBDataset,
    CambrianInterleavedDataset,
    PromptImageJsonlDataset,
    BasicEditJsonlDataset,
)
from training.config_resolver import (
    apply_dataset_sources,
    configure_hf_cache_env as common_configure_hf_cache_env,
    resolve_hf_cache_root as common_resolve_hf_cache_root,
    resolve_model_cfg_block,
    resolve_model_local_files_only,
    resolve_vq_repo_source,
)
from training.utils import (
    get_config,
    flatten_omega_conf,
    image_transform,
    mask_or_random_replace_tokens,
    AverageMeter,
)

from models import MAGVITv2, get_mask_schedule, DyninOmniModelLM, DyninOmniConfig
from training.prompting_utils import UniversalPrompting
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error

# Evaluation imports.
import re
import editdistance
import soundfile as sf
from functools import partial
from transformers import pipeline

SYSTEM_PROMPT_LEN = 28

cv2.setNumThreads(0)
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"

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


def _first_non_empty_text(*values: Any) -> Optional[str]:
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


def _resolve_stage3_model_sources(
    config,
    model_cfg: Optional[Any] = None,
) -> tuple[str, str, str, str]:
    if model_cfg is None:
        model_cfg = resolve_model_cfg_block(config)

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


def _configure_multiprocessing() -> None:
    try:
        mp.set_sharing_strategy("file_descriptor")
    except RuntimeError as exc:
        bootstrap_logger.warning(
            "Failed to set multiprocessing sharing strategy to 'file_descriptor': %s",
            exc,
        )


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
        "audio_tokens": [item.get("audio_tokens") for item in batch],
    }


def _empty_audio_batch() -> dict[str, list[Any]]:
    """Utility to create an empty speech batch placeholder."""
    return {
        "audio_path": [],
        "text": [],
        "audio_tokens": [],
    }


def collate_fn_mmu_mult(batch):
    return {
        "images": [item["images"] for item in batch],
        "text": [item["text"] for item in batch],
    }


def collate_fn_x2i(batch):
    t2i_texts: list[str] = []
    t2i_images: list[torch.Tensor] = []

    i2i_prompts: list[str] = []
    i2i_source_images: list[torch.Tensor] = []
    i2i_target_images: list[torch.Tensor] = []
    i2i_target_texts: list[str] = []

    ref_image: Optional[torch.Tensor] = None

    has_i2i_sample = False

    def _clean_prompt(prompt: Optional[str]) -> Optional[str]:
        if not isinstance(prompt, str):
            return None
        cleaned = prompt.strip()
        return cleaned or None

    for sample in batch:
        input_prompt = _clean_prompt(sample.get("input_prompt"))
        output_prompt = _clean_prompt(sample.get("output_prompt"))
        edit_prompt = _clean_prompt(sample.get("edit_prompt"))
        inverse_prompt = _clean_prompt(sample.get("inverse_prompt"))
        input_image = sample.get("input_image")
        output_image = sample.get("output_image")
        output_text = sample.get("output_text")

        if isinstance(input_image, torch.Tensor) and ref_image is None:
            ref_image = input_image
        if isinstance(output_image, torch.Tensor) and ref_image is None:
            ref_image = output_image

        has_edit_pair = (
            isinstance(input_image, torch.Tensor)
            and isinstance(output_image, torch.Tensor)
            and (
                (edit_prompt and edit_prompt.strip())
                or (inverse_prompt and inverse_prompt.strip())
            )
        )

        if has_edit_pair:
            has_i2i_sample = True
            edit_candidates: list[tuple[str, torch.Tensor, torch.Tensor]] = []
            if edit_prompt:
                edit_candidates.append((edit_prompt, input_image, output_image))
            if inverse_prompt:
                edit_candidates.append((inverse_prompt, output_image, input_image))

            if edit_candidates:
                chosen_prompt, chosen_src, chosen_tgt = random.choice(edit_candidates)
                i2i_prompts.append(chosen_prompt)
                i2i_source_images.append(chosen_src)
                i2i_target_images.append(chosen_tgt)
                if isinstance(output_text, str):
                    i2i_target_texts.append(output_text)
                else:
                    i2i_target_texts.append("")
            continue
        else:
            if input_prompt and isinstance(input_image, torch.Tensor):
                t2i_texts.append(input_prompt)
                t2i_images.append(input_image)
            elif output_prompt and isinstance(output_image, torch.Tensor):
                t2i_texts.append(output_prompt)
                t2i_images.append(output_image)

    if has_i2i_sample:
        # If any i2i sample exists, use this batch for i2i only and clear t2i.
        t2i_texts = []
        t2i_images = []

    def stack_images(images: list[torch.Tensor]) -> torch.Tensor:
        if images:
            return torch.stack(images, dim=0)
        if ref_image is not None:
            c, h, w = ref_image.shape[-3:]
            return torch.empty((0, c, h, w), dtype=ref_image.dtype)
        return torch.empty((0, 3, 0, 0), dtype=torch.float32)

    return {
        "t2i": {
            "texts": t2i_texts,
            "images": stack_images(t2i_images),
        },
        "i2i": {
            "prompts": i2i_prompts,
            "source_images": stack_images(i2i_source_images),
            "target_images": stack_images(i2i_target_images),
            "target_texts": i2i_target_texts,
        },
    }


def collate_fn_v2t(batch: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    filtered = [sample for sample in batch if sample is not None]
    if not filtered:
        return None
    video_tensors: list[torch.Tensor] = []
    captions: list[Any] = []
    for sample in filtered:
        frames = sample.get("video")
        if frames is None:
            continue
        frame_tensor = torch.stack(frames, dim=0)
        video_tensors.append(frame_tensor)
        captions.append(sample.get("caption"))
    if not video_tensors:
        return None
    return {
        "video": torch.stack(video_tensors, dim=0),
        "captions": captions,
    }


def s2t_eval_collate_fn(batch, vq_model_audio, tokenizer, uni_prompting, config):

    audio_tokens_batch = []
    offset = len(uni_prompting.text_tokenizer) + int(config.model.dynin_omni.codebook_size)
    for item in batch:
        audio_entry = item["audio_path"]
        if isinstance(audio_entry, torch.Tensor):
            tokens = audio_entry.cpu()
        else:
            tokens = vq_model_audio.encode(audio_entry).cpu()
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


# Evaluation helpers.


def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = (
        torch.zeros(
            mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
        )
        + base
    )

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1

    return num_transfer_tokens


@torch.no_grad()
def generate(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    attention_mask=None,
):
    """
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (B, L), where B is batch size.
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    """
    if attention_mask is not None and 0.0 in attention_mask:
        attention_bias = (
            (attention_mask[:, :, None] & attention_mask[:, None, :])
            .bool()
            .unsqueeze(1)
        )
    else:
        attention_bias = None
    batch_size = prompt.shape[0]
    x = torch.full(
        (batch_size, prompt.shape[1] + gen_length), mask_id, dtype=torch.long
    ).to(model.device)
    x[:, : prompt.shape[1]] = prompt.clone()

    prompt_index = x != mask_id

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (
            x[
                :,
                prompt.shape[1]
                + num_block * block_length : prompt.shape[1]
                + (num_block + 1) * block_length :,
            ]
            == mask_id
        )
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = x == mask_id
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_bias=attention_bias).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

            if remasking == "low_confidence":
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )  # b, l
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length :] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


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


def _resolve_mask_schedule(config):
    schedule_cfg = getattr(config, "mask_schedule", None)
    if isinstance(schedule_cfg, DictConfig):
        schedule_name = getattr(schedule_cfg, "schedule", None)
        params_cfg = getattr(schedule_cfg, "params", None)
    elif isinstance(schedule_cfg, dict):
        schedule_name = schedule_cfg.get("schedule")
        params_cfg = schedule_cfg.get("params")
    else:
        schedule_name = None
        params_cfg = None
    if schedule_name is None:
        schedule_name = config.training.get("mask_schedule", "cosine")
    params = {}
    if params_cfg is not None:
        if isinstance(params_cfg, DictConfig):
            params = OmegaConf.to_container(params_cfg, resolve=True) or {}
        elif isinstance(params_cfg, dict):
            params = dict(params_cfg)
        else:
            params = params_cfg
    if not isinstance(params, dict):
        params = {}
    return get_mask_schedule(schedule_name, **params)


def _tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    image = torch.clamp(
        (image_tensor.detach().cpu().float() + 1.0) / 2.0, min=0.0, max=1.0
    )
    array = (image.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(array)


# T2I evaluation logic.


@torch.no_grad()
def evaluate_t2i(
    model, vq_model_image, uni_prompting, config, accelerator, global_step
):
    if not accelerator.is_main_process:
        return
    logger.info("***** Running T2I Evaluation *****")
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.eval()
    dataset_cfg = getattr(config.dataset, "params", {})
    prompts_file = getattr(dataset_cfg, "t2i_eval_prompts_file", None)
    if prompts_file is None:
        prompts_file = getattr(dataset_cfg, "validation_prompts_file", None)
    if not prompts_file:
        logger.warning(
            "No validation prompts file configured. Skipping T2I evaluation."
        )
        return
    prompts_path = Path(prompts_file)
    if not prompts_path.is_absolute():
        prompts_path = Path.cwd() / prompts_path
        if not prompts_path.exists():
            repo_root = Path(__file__).resolve().parents[2]
            alt_path = repo_root / prompts_file
            if alt_path.exists():
                prompts_path = alt_path
    try:
        with open(prompts_path, "r", encoding="utf-8") as handle:
            prompts = [line.strip() for line in handle if line.strip()]
    except OSError as exc:
        logger.warning(
            f"Failed to read validation prompts from '{prompts_path}': {exc}. Skipping T2I evaluation."
        )
        return
    if not prompts:
        logger.warning("Validation prompts file is empty. Skipping T2I evaluation.")
        return
    max_samples = getattr(config.experiment, "eval_num_t2i_samples", 8)
    if not isinstance(max_samples, int) or max_samples <= 0:
        max_samples = 8
    prompts = prompts[:max_samples]
    mask_schedule = _resolve_mask_schedule(config)
    mask_token_id = unwrapped_model.config.mask_token_id
    seq_len = getattr(getattr(config.model, "dynin_omni", None), "num_vq_tokens", None)
    if seq_len is None:
        seq_len = getattr(unwrapped_model.config, "num_vq_tokens", None)
    if seq_len is None:
        logger.warning(
            "Unable to determine image token sequence length. Skipping T2I evaluation."
        )
        return
    seq_len = int(seq_len)
    device = accelerator.device
    image_tokens = torch.full(
        (len(prompts), seq_len), mask_token_id, dtype=torch.long, device=device
    )
    input_ids, attention_mask = uni_prompting((prompts, image_tokens), "t2i_gen")
    if config.training.guidance_scale > 0:
        uncond_input_ids, uncond_attention_mask = uni_prompting(
            ([""] * len(prompts), image_tokens), "t2i_gen"
        )
        cfg_scale = config.training.guidance_scale
    else:
        uncond_input_ids, uncond_attention_mask = None, None
        cfg_scale = 0.0
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32
    use_autocast = (
        accelerator.device.type == "cuda" and accelerator.mixed_precision != "no"
    )
    autocast_ctx = (
        torch.autocast("cuda", dtype=weight_dtype)
        if use_autocast
        else contextlib.nullcontext()
    )
    with autocast_ctx:
        gen_token_ids = unwrapped_model.t2i_generate(
            input_ids=input_ids,
            uncond_input_ids=uncond_input_ids,
            attention_mask=attention_mask,
            uncond_attention_mask=uncond_attention_mask,
            guidance_scale=3.5,
            temperature=config.training.get("generation_temperature", 1.0),
            timesteps=20,
            noise_schedule=mask_schedule,
            noise_type=config.training.get("noise_type", "mask"),
            predict_all_tokens=config.training.get("predict_all_tokens", False),
            seq_len=seq_len,
            uni_prompting=uni_prompting,
            config=config,
        )
    gen_token_ids = torch.clamp(
        gen_token_ids, min=0, max=unwrapped_model.config.codebook_size - 1
    )
    images = vq_model_image.decode_code(gen_token_ids)
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    images = images.permute(0, 2, 3, 1).cpu().numpy() * 255.0
    pil_images = [Image.fromarray(img.astype(np.uint8)) for img in images]
    wandb_images = [
        wandb.Image(img, caption=prompt) for img, prompt in zip(pil_images, prompts)
    ]
    accelerator.log({"eval/t2i_samples": wandb_images}, step=global_step)


# I2I evaluation logic.


@torch.no_grad()
def evaluate_i2i(
    model, vq_model_image, uni_prompting, config, accelerator, global_step
):
    if not accelerator.is_main_process:
        return
    logger.info("***** Running I2I Evaluation *****")
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.eval()
    dataset_cfg_raw = getattr(config.dataset, "params", {})
    resolution = 256

    def _cfg_to_dict(cfg):
        if cfg is None:
            return None
        if isinstance(cfg, dict):
            return cfg
        if isinstance(cfg, DictConfig):
            return OmegaConf.to_container(cfg, resolve=True)
        return cfg

    dataset_cfg = _cfg_to_dict(dataset_cfg_raw) or {}

    eval_datasets: list[Dataset] = []
    eval_source_names: list[str] = []

    # HQ-Edit evaluation dataset (always attempt; mirrors training)
    try:
        hqedit_split = dataset_cfg.get("hqedit_split", "train")
        hqedit_eval = HQEditX2IDataset(split=hqedit_split, resolution=resolution)
        if len(hqedit_eval) > 0:
            eval_datasets.append(hqedit_eval)
            eval_source_names.append(f"HQ-Edit[{hqedit_split}]")
        else:
            logger.warning(
                "HQ-Edit evaluation split '%s' is empty; skipping.", hqedit_split
            )
    except Exception as exc:
        logger.warning("Failed to build HQ-Edit evaluation dataset: %s", exc)

    if not eval_datasets:
        logger.warning("No i2i evaluation dataset available. Skipping.")
        return

    eval_dataset = (
        eval_datasets[0]
        if len(eval_datasets) == 1
        else CombinedX2IDataset(eval_datasets)
    )
    logger.info("Using I2I evaluation datasets: %s", ", ".join(eval_source_names))

    max_samples = getattr(config.experiment, "eval_num_i2i_samples", 8)

    if not isinstance(max_samples, int) or max_samples <= 0:
        max_samples = 8
    num_samples = min(max_samples, len(eval_dataset))
    if len(eval_dataset) <= num_samples:
        sample_indices = list(range(len(eval_dataset)))
    else:
        sample_indices = random.sample(range(len(eval_dataset)), num_samples)
    samples = [eval_dataset[i] for i in sample_indices]
    prompts = []
    original_tensors = []
    target_tensors = []
    for sample in samples:
        prompts.append(sample.get("edit_prompt") or sample.get("output_prompt") or "")
        original_tensors.append(sample["input_image"])
        target_tensors.append(sample["output_image"])
    original_images = torch.stack(original_tensors, dim=0).to(accelerator.device)
    original_tokens = vq_model_image.get_code(original_images) + len(
        uni_prompting.text_tokenizer
    )
    seq_len = original_tokens.shape[-1]
    mask_token_id = unwrapped_model.config.mask_token_id
    placeholder = torch.full(
        (num_samples, seq_len),
        mask_token_id,
        dtype=torch.long,
        device=accelerator.device,
    )
    input_ids, attention_mask = uni_prompting(
        (prompts, original_tokens, placeholder), "i2i_gen"
    )
    if config.training.guidance_scale > 0:
        uncond_input_ids, uncond_attention_mask = uni_prompting(
            ([""] * num_samples, original_tokens, placeholder), "i2i_gen"
        )
        cfg_scale = config.training.guidance_scale
    else:
        uncond_input_ids, uncond_attention_mask = None, None
        cfg_scale = 0.0
    mask_schedule = _resolve_mask_schedule(config)
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32
    use_autocast = (
        accelerator.device.type == "cuda" and accelerator.mixed_precision != "no"
    )
    autocast_ctx = (
        torch.autocast("cuda", dtype=weight_dtype)
        if use_autocast
        else contextlib.nullcontext()
    )
    with autocast_ctx:
        gen_token_ids = unwrapped_model.i2i_generate(
            input_ids=input_ids,
            uncond_input_ids=uncond_input_ids,
            attention_mask=attention_mask,
            uncond_attention_mask=uncond_attention_mask,
            guidance_scale=3.5,
            temperature=config.training.get("generation_temperature", 1.0),
            timesteps=24,
            noise_schedule=mask_schedule,
            noise_type=config.training.get("noise_type", "mask"),
            seq_len=seq_len,
            uni_prompting=uni_prompting,
            config=config,
        )
    gen_token_ids = torch.clamp(
        gen_token_ids, min=0, max=unwrapped_model.config.codebook_size - 1
    )
    generated_images = vq_model_image.decode_code(gen_token_ids)
    generated_images = torch.clamp((generated_images + 1.0) / 2.0, min=0.0, max=1.0)
    gen_images_pil = [
        Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8))
        for img in generated_images
    ]
    source_pil = [_tensor_to_pil(tensor) for tensor in original_tensors]
    target_pil = [_tensor_to_pil(tensor) for tensor in target_tensors]
    log_resolution = getattr(config.experiment, "eval_image_log_resolution", 512)
    wandb_images = []
    for prompt, src, pred, tgt in zip(prompts, source_pil, gen_images_pil, target_pil):
        composite = Image.new("RGB", (log_resolution * 3, log_resolution))
        src_resized = src.resize(
            (log_resolution, log_resolution), Image.Resampling.LANCZOS
        )
        pred_resized = pred.resize(
            (log_resolution, log_resolution), Image.Resampling.LANCZOS
        )
        tgt_resized = tgt.resize(
            (log_resolution, log_resolution), Image.Resampling.LANCZOS
        )
        composite.paste(src_resized, (0, 0))
        composite.paste(pred_resized, (log_resolution, 0))
        composite.paste(tgt_resized, (log_resolution * 2, 0))
        wandb_images.append(wandb.Image(composite, caption=f"Prompt: {prompt}"))
    accelerator.log({"eval/i2i_samples": wandb_images}, step=global_step)


# Text evaluation logic.
@torch.no_grad()
def evaluate_text(model, uni_prompting, config, accelerator, global_step):
    if not accelerator.is_main_process:
        return

    logger.info("***** Running Text Evaluation *****")
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.eval()

    dataset_cfg = getattr(config.dataset, "params", {})
    prompts_file = getattr(dataset_cfg, "text_eval_prompts_file", None)
    if prompts_file is None:
        prompts_file = getattr(dataset_cfg, "validation_prompts_file", None)
    if not prompts_file:
        logger.warning(
            "No text evaluation prompts file configured. Skipping text evaluation."
        )
        return

    prompts_path = Path(prompts_file)
    if not prompts_path.is_absolute():
        prompts_path = Path.cwd() / prompts_path
        if not prompts_path.exists():
            repo_root = Path(__file__).resolve().parents[2]
            alt_path = repo_root / prompts_file
            if alt_path.exists():
                prompts_path = alt_path

    if not prompts_path.exists():
        logger.warning(
            f"Text evaluation prompts file '{prompts_file}' not found. Skipping text evaluation."
        )
        return

    try:
        with open(prompts_path, "r", encoding="utf-8") as handle:
            raw_prompts = [line.strip() for line in handle if line.strip()]
    except OSError as exc:
        logger.warning(
            f"Failed to read text evaluation prompts from '{prompts_path}': {exc}. Skipping text evaluation."
        )
        return

    if not raw_prompts:
        logger.warning(
            "Text evaluation prompt list is empty. Skipping text evaluation."
        )
        return

    max_samples = getattr(config.experiment, "eval_num_text_samples", 4)
    if not isinstance(max_samples, int) or max_samples <= 0:
        max_samples = 4
    questions = raw_prompts[:max_samples]

    chat_prompts = [
        f"<|start_header_id|>user<|end_header_id|>\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        for question in questions
    ]

    tokenizer = uni_prompting.text_tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    answers: list[str] = []

    preproc_cfg = config.dataset.preprocessing
    max_text_len = getattr(
        preproc_cfg, "max_seq_length_text", preproc_cfg.max_seq_length
    )
    max_lm_input = getattr(preproc_cfg, "max_seq_length_lm_input", max_text_len)
    gen_len = max_text_len
    block_len = 128 if gen_len % 128 == 0 else gen_len

    for chat_prompt in chat_prompts:
        tokens = tokenizer(
            chat_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_lm_input,
        )

        input_ids = tokens["input_ids"].to(accelerator.device)
        out = generate(
            unwrapped_model,
            input_ids,
            steps=256,
            gen_length=512,
            block_length=256,
            temperature=1,
            cfg_scale=0.0,
            remasking="low_confidence",
        )
        answer = tokenizer.batch_decode(
            out[:, input_ids.shape[1] :], skip_special_tokens=True
        )

        answers.append(answer)

    table = wandb.Table(columns=["Index", "Question", "Answer"])
    for idx, (question, answer) in enumerate(zip(questions, answers)):
        table.add_data(idx, question, answer)

    accelerator.log({"eval/text_samples": table}, step=global_step)


# S2T evaluation logic.
@torch.no_grad()
def evaluate_s2t(
    model, vq_model_audio, uni_prompting, config, accelerator, global_step
):
    if not accelerator.is_main_process:
        return
    logger.info("***** Running S2T Evaluation (WER on Librispeech test-clean) *****")
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.eval()
    try:
        s2t_eval_batch_size = int(getattr(config.training, "batch_size_s2t", 0) or 0)
    except (TypeError, ValueError):
        s2t_eval_batch_size = 0
    if s2t_eval_batch_size <= 0:
        logger.warning(
            "Skipping S2T evaluation because training.batch_size_s2t is <= 0."
        )
        return

    dataset_cfg = getattr(config.dataset, "params", {})
    s2t_eval_root_path = getattr(dataset_cfg, "s2t_eval_root_path", None)
    if not s2t_eval_root_path:
        logger.warning("No s2t_eval_root_path configured. Skipping S2T evaluation.")
        return

    # 1. Load Dataset
    try:
        s2t_eval_dataset_raw = _load_dataset_with_cache(
            "librispeech_asr", "clean", split="test", streaming=False
        ).select(range(32))
        s2t_eval_dataset = S2TEvalDataset(
            s2t_eval_dataset_raw, root_path=s2t_eval_root_path
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
        batch_size=s2t_eval_batch_size,
        shuffle=False,
        collate_fn=collate_with_args,
    )

    local_results = []

    for batch in tqdm(s2t_eval_dataloader, desc="S2T Evaluation"):
        input_ids = batch["input_ids"]
        gt_texts = batch["gt_texts"]
        sample_ids = batch["sample_ids"]

        output_ids = unwrapped_model.mmu_generate(
            input_ids,
            max_new_tokens=256,
            steps=256,
            block_length=128,
            remasking="low_confidence",
        )

        decoded_texts = uni_prompting.text_tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1] :], skip_special_tokens=True
        )

        eos_token = uni_prompting.text_tokenizer.eos_token
        eos_marker = eos_token if eos_token is not None else "</s>"
        for i in range(len(decoded_texts)):
            full_text = decoded_texts[i]
            eos_idx = full_text.find(eos_marker)
            cleaned_text = full_text[:eos_idx] if eos_idx != -1 else full_text
            cleaned_text = cleaned_text.replace(eos_marker, "").strip()
            local_results.append(
                {
                    "sample_id": sample_ids[i],
                    "gt_text": gt_texts[i],
                    "decoded_text": cleaned_text,
                }
            )

    if not local_results:
        logger.warning("S2T evaluation produced no results.")
        return

    gt_list = [res["gt_text"] for res in local_results]
    pred_list = [res["decoded_text"] for res in local_results]

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

    samples_table = wandb.Table(columns=["ID", "Ground Truth", "Prediction"])
    for idx, res in enumerate(local_results):
        sample_id = res.get("sample_id", idx)
        samples_table.add_data(sample_id, res["gt_text"], res["decoded_text"])

    accelerator.log({"eval/s2t_samples": samples_table}, step=global_step)


# T2S evaluation logic.
@torch.no_grad()
def evaluate_t2s(
    model, vq_model_audio, uni_prompting, config, accelerator, global_step
):
    if not accelerator.is_main_process:
        return
    logger.info("***** Running T2S Evaluation (WER via Whisper on Librispeech) *****")
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.eval()
    try:
        t2s_eval_batch_size = int(getattr(config.training, "batch_size_t2s", 0) or 0)
    except (TypeError, ValueError):
        t2s_eval_batch_size = 0
    if t2s_eval_batch_size <= 0:
        logger.warning(
            "Skipping T2S evaluation because training.batch_size_t2s is <= 0."
        )
        return
    dataset_cfg = getattr(config.dataset, "params", {})
    eval_output_root = getattr(dataset_cfg, "eval_output_root", None)

    # 1. Load Dataset & Whisper Model
    try:
        t2s_eval_dataset_raw = _load_dataset_with_cache(
            "librispeech_asr", "clean", split="test"
        ).select(range(8))
        whisper_pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3",
            device=accelerator.device,
        )
    except Exception as e:
        logger.error(f"Failed to load T2S dataset or Whisper model: {e}")
        return

    if eval_output_root:
        output_dir_per_step = (
            Path(eval_output_root)
            / config.experiment.output_dir
            / "eval_audio"
            / f"step_{global_step}"
        )
    else:
        output_dir_per_step = (
            Path(config.experiment.output_dir) / "eval_audio" / f"step_{global_step}"
        )
    output_dir_per_step.mkdir(parents=True, exist_ok=True)

    t2s_eval_dataset = T2SEvalDataset(t2s_eval_dataset_raw)
    t2s_dataloader = DataLoader(t2s_eval_dataset, batch_size=t2s_eval_batch_size)

    local_results = []
    mask_token_id = unwrapped_model.config.mask_token_id
    mask_schedule = get_mask_schedule(config.training.get("mask_schedule", "cosine"))

    # 2. Evaluation Loop
    for batch in tqdm(t2s_dataloader, desc="T2S Evaluation"):
        gt_texts = batch["gt_text"]
        sample_ids = batch["sample_id"]

        # Chat-style instruction formatting for T2S: user prompt + text
        prompts = [
            f"<|start_header_id|>user<|end_header_id|>\n{random.choice(T2S_INSTRUCTION)}\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            for text in gt_texts
        ]
        batch_size = len(prompts)

        # We need a reasonable length for generated audio tokens
        speech_token_length = 384 - 1  # -1 for soa token
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
            guidance_scale=5.0,
            temperature=1.0,
            timesteps=50,
            noise_schedule=mask_schedule,
            noise_type="mask",
            seq_len=383,
            uni_prompting=uni_prompting,
            config=config,
        )

        # Decode and run Whisper
        for i in range(batch_size):
            gt = gt_texts[i].rsplit("\n", 1)[-1].strip()

            gen_speech_tokens = output_ids[i]

            # Remove padding/eos if necessary and keep valid token range.
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
            output_wav_path = str(output_dir_per_step / filename)
            condition = "gender-female_emotion-neutral_speed-normal_pitch-normal"

            audio_array = vq_model_audio.decode(
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

    if not local_results:
        logger.warning(
            "Skipping T2S evaluation logging because no samples were generated."
        )
        return

    gt_list = [res["gt_text"] for res in local_results]
    pred_list = [res["whisper_text"] for res in local_results]

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

    results_table = wandb.Table(
        columns=["ID", "Ground Truth", "Whisper Transcription", "Generated Audio"]
    )
    for res in local_results[:8]:
        audio = wandb.Audio(res["audio_path"], caption=res["whisper_text"])
        results_table.add_data(
            res["sample_id"], res["gt_text"], res["whisper_text"], audio
        )

    accelerator.log({"eval/t2s_samples": results_table}, step=global_step)


@torch.no_grad()
def evaluate_t2s_mmu_like(
    model, vq_model_audio, uni_prompting, config, accelerator, global_step
):
    """Text-to-speech evaluation using the MMU-style block refinement decoder."""

    if not accelerator.is_main_process:
        return

    logger.info("***** Running T2S Evaluation (MMU-style decoder) *****")
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.eval()
    try:
        t2s_eval_batch_size = int(getattr(config.training, "batch_size_t2s", 0) or 0)
    except (TypeError, ValueError):
        t2s_eval_batch_size = 0
    if t2s_eval_batch_size <= 0:
        logger.warning(
            "Skipping T2S MMU-style evaluation because training.batch_size_t2s is <= 0."
        )
        return
    dataset_cfg = getattr(config.dataset, "params", {})
    eval_output_root = getattr(dataset_cfg, "eval_output_root", None)

    try:
        t2s_eval_dataset_raw = _load_dataset_with_cache(
            "librispeech_asr", "clean", split="test"
        ).select(range(8))
        whisper_pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3",
            device=accelerator.device,
        )
    except Exception as exc:
        logger.error(
            f"Failed to load T2S dataset or Whisper model for MMU-style eval: {exc}"
        )
        return

    if eval_output_root:
        output_dir_per_step = (
            Path(eval_output_root)
            / config.experiment.output_dir
            / "eval_audio"
            / f"step_{global_step}_mmu"
        )
    else:
        output_dir_per_step = (
            Path(config.experiment.output_dir)
            / "eval_audio"
            / f"step_{global_step}_mmu"
        )
    output_dir_per_step.mkdir(parents=True, exist_ok=True)

    t2s_eval_dataset = T2SEvalDataset(t2s_eval_dataset_raw)
    t2s_dataloader = DataLoader(t2s_eval_dataset, batch_size=t2s_eval_batch_size)

    local_results = []
    mask_token_id = unwrapped_model.config.mask_token_id

    codebook_size = config.model.dynin_omni.codebook_size
    speech_vocab_size = 4096

    for batch in tqdm(t2s_dataloader, desc="T2S MMU Eval"):
        gt_texts = batch["gt_text"]
        sample_ids = batch["sample_id"]

        prompts = [
            f"<|start_header_id|>user<|end_header_id|>\n{random.choice(T2S_INSTRUCTION)}\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            for text in gt_texts
        ]

        batch_size = len(prompts)
        speech_token_length = 512 - 1
        audio_tokens = (
            torch.ones(
                (batch_size, speech_token_length),
                dtype=torch.long,
                device=accelerator.device,
            )
            * mask_token_id
        )
        input_ids, attention_mask = uni_prompting((prompts, audio_tokens), "t2s_gen")

        output_ids = unwrapped_model.t2s_generate_mmu_like(
            input_ids=input_ids,
            max_new_tokens=speech_token_length,
            steps=512 - 1,
            block_length=512 - 1,
            temperature=1.0,
            cfg_scale=3.5,
            mask_token_id=mask_token_id,
            attention_mask=attention_mask,
            uni_prompting=uni_prompting,
            codebook_size=codebook_size,
            audio_codebook_size=speech_vocab_size,
        )

        for i in range(batch_size):
            gt = gt_texts[i].rsplit("\n", 1)[-1].strip()

            gen_speech_tokens = output_ids[i]
            if isinstance(gen_speech_tokens, torch.Tensor):
                gen_speech_tokens = gen_speech_tokens.detach().cpu()

            token_list = gen_speech_tokens.tolist()
            if not token_list:
                logger.warning(
                    f"Generated token list is empty for sample {sample_ids[i]} (MMU eval). Skipping."
                )
                continue

            speech_unit_str = " ".join(map(str, token_list))
            speech_unit_for_decode = "".join(
                [f"<|speech_{unit}|>" for unit in speech_unit_str.split(" ")]
            )

            filename = f"process_{accelerator.process_index}_{sample_ids[i]}_mmu.wav"
            output_wav_path = str(output_dir_per_step / filename)
            condition = "gender-female_emotion-neutral_speed-normal_pitch-normal"

            try:
                vq_model_audio.decode(
                    speech_unit_for_decode,
                    condition=condition,
                    output_wav_file=output_wav_path,
                )
            except Exception as exc:
                logger.error(
                    f"Decoding failed for sample {sample_ids[i]} (MMU eval): {exc}"
                )
                continue

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

    if not local_results:
        logger.warning(
            "Skipping T2S MMU-style evaluation because no samples were generated."
        )
        return

    gt_list = [res["gt_text"] for res in local_results]
    pred_list = [res["whisper_text"] for res in local_results]

    wer, errors, words = calculate_wer(pred_list, gt_list)
    logger.info(
        f"T2S (MMU-style) Final WER: {wer:.4f} | Word Errors: {errors} | Total Words: {words}"
    )

    accelerator.log(
        {
            "eval/t2s_mmu_like_wer": wer,
            "eval/t2s_mmu_like_word_errors": errors,
            "eval/t2s_mmu_like_total_words": words,
        },
        step=global_step,
    )

    results_table = wandb.Table(
        columns=["ID", "Ground Truth", "Whisper Transcription", "Generated Audio"]
    )
    for res in local_results[:8]:
        audio = wandb.Audio(res["audio_path"], caption=res["whisper_text"])
        results_table.add_data(
            res["sample_id"], res["gt_text"], res["whisper_text"], audio
        )

    accelerator.log({"eval/t2s_mmu_like_samples": results_table}, step=global_step)


# V2T evaluation logic.
@torch.no_grad()
def evaluate_v2t(
    model, vq_model_image, uni_prompting, config, accelerator, global_step
):
    # This is a qualitative evaluation, so it only runs on the main process.
    if not accelerator.is_main_process:
        return

    logger.info("***** Running V2T Qualitative Evaluation *****")
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.eval()

    dataset_cfg = getattr(config.dataset, "params", {})
    video_root = getattr(dataset_cfg, "v2t_eval_video_root", None)
    if not video_root:
        logger.warning("No v2t_eval_video_root configured. Skipping V2T evaluation.")
        return
    video_root_path = Path(video_root)
    if not video_root_path.is_absolute():
        video_root_path = Path.cwd() / video_root_path
        if not video_root_path.exists():
            repo_root = Path(__file__).resolve().parents[2]
            alt_path = repo_root / video_root
            if alt_path.exists():
                video_root_path = alt_path
    if not video_root_path.exists():
        logger.warning(
            f"V2T eval root '{video_root_path}' not found. Skipping V2T evaluation."
        )
        return

    file_list = [f for f in os.listdir(video_root_path) if f.lower().endswith(".mp4")]
    if not file_list:
        logger.warning(
            f"No .mp4 files found in '{video_root_path}'. Skipping V2T evaluation."
        )
        return

    question = "Please provide a detailed description of the video."
    results_table = wandb.Table(columns=["Video ID", "Question", "Generated Caption"])

    for file_name in tqdm(
        file_list[:], desc="V2T Evaluation", disable=not accelerator.is_main_process
    ):
        video_path = str(video_root_path / file_name)

        # 1. Load and process video
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
        video_tokens = video_tokens.view(1, -1)  # Flatten tokens

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
        logger.info("V2T eval sample %s: %s", file_name, text)

        # Log result.
        results_table.add_data(file_name, question, text)

    accelerator.log({"eval/v2t_qualitative_samples": results_table}, step=global_step)


# Main evaluation orchestrator.


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
    if accelerator.is_main_process:
        logger.info(f"--- Starting evaluation at step {global_step} ---")
    model.eval()

    # Synchronize all ranks before entering main-process-only evaluation.
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        # Run the stage-3 evaluation set used in current training.
        try:
            eval_batch_size_s2t = int(
                getattr(config.training, "batch_size_s2t", 0) or 0
            )
        except (TypeError, ValueError):
            eval_batch_size_s2t = 0
        try:
            eval_batch_size_t2s = int(
                getattr(config.training, "batch_size_t2s", 0) or 0
            )
        except (TypeError, ValueError):
            eval_batch_size_t2s = 0

        if eval_batch_size_s2t > 0:
            evaluate_s2t(
                model, vq_model_audio, uni_prompting, config, accelerator, global_step
            )
        else:
            logger.info(
                "Skipping S2T evaluation at step %s because batch_size_s2t <= 0.",
                global_step,
            )

        if eval_batch_size_t2s > 0:
            evaluate_t2s_mmu_like(
                model, vq_model_audio, uni_prompting, config, accelerator, global_step
            )
        else:
            logger.info(
                "Skipping T2S evaluation at step %s because batch_size_t2s <= 0.",
                global_step,
            )

    # Ensure non-main ranks resume only after evaluation is complete.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info(
            f"--- Finished evaluation at step {global_step}. Returning to training. ---"
        )
    model.train()


def main():
    _configure_multiprocessing()
    # Set up accelerator.
    config = get_config()
    _sanitize_experiment_intervals(config)
    apply_dataset_sources(config)
    hf_cache_root = _configure_hf_cache_env(config)
    bootstrap_logger.info("HF cache root: %s", hf_cache_root)
    gradient_accumulation_steps = int(config.training.gradient_accumulation_steps)
    if gradient_accumulation_steps % 2 != 0:
        adjusted_steps = gradient_accumulation_steps + 1
        bootstrap_logger.warning(
            "gradient_accumulation_steps=%d is odd; adjusting to %d for two-branch alternation.",
            gradient_accumulation_steps,
            adjusted_steps,
        )
        config.training.gradient_accumulation_steps = adjusted_steps
        gradient_accumulation_steps = adjusted_steps
    bootstrap_logger.info(
        "VQ audio model: %s", resolve_vq_repo_source(config.model.vq_model_audio)
    )
    bootstrap_logger.info(
        "VQ image model: %s", resolve_vq_repo_source(config.model.vq_model_image)
    )

    # Enable TF32 on Ampere GPUs
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.output_dir) / "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
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
        total_batch_size_per_gpu
        * accelerator.num_processes
        * gradient_accumulation_steps
    )

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config[
            "train_micro_batch_size_per_gpu"
        ] = total_batch_size_per_gpu

    # Set up logging, seed, and config.
    # Make one log on every process with the configuration for debugging.
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

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
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

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed)

    # Load models and optimizer.
    logger.info("Loading models and optimizer")
    dynin_omni_cfg = resolve_model_cfg_block(config)
    model_local_files_only = resolve_model_local_files_only(config, default=False)
    if _is_env_truthy("HF_HUB_OFFLINE") or _is_env_truthy("TRANSFORMERS_OFFLINE"):
        model_local_files_only = True

    tokenizer_source, model_source, vq_image_source, vq_audio_source = (
        _resolve_stage3_model_sources(config, model_cfg=dynin_omni_cfg)
    )
    logger.info(
        "Model sources | tokenizer=%s | model=%s | vq_image=%s | vq_audio=%s | local_files_only=%s",
        tokenizer_source,
        model_source,
        vq_image_source,
        vq_audio_source,
        model_local_files_only,
    )

    tokenizer_kwargs = {
        "padding_side": "left",
        "trust_remote_code": True,
        "local_files_only": model_local_files_only,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        **tokenizer_kwargs,
    )

    preproc_config = config.dataset.preprocessing
    max_seq_text = getattr(
        preproc_config, "max_seq_length_text", preproc_config.max_seq_length
    )
    max_seq_lm_input = getattr(preproc_config, "max_seq_length_lm_input", max_seq_text)
    max_seq_mmu = getattr(
        preproc_config, "max_seq_length_mmu", preproc_config.max_seq_length
    )
    max_seq_mmu_input = getattr(preproc_config, "max_seq_length_mmu_input", max_seq_mmu)
    max_seq_s2t = getattr(
        preproc_config, "max_seq_length_s2t", preproc_config.max_seq_length
    )
    max_seq_t2i = getattr(
        preproc_config, "max_seq_length_t2i", preproc_config.max_seq_length
    )
    max_seq_t2s = getattr(
        preproc_config, "max_seq_length_t2s", preproc_config.max_seq_length
    )

    uni_prompting = UniversalPrompting(
        tokenizer,
        max_text_len=max_seq_text,
        max_audio_len=preproc_config.max_aud_length,
        max_audio_len_short=preproc_config.max_aud_length_short,
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
            # Dynin-Omni Special Tokens
            "<|i2i|>",
            "<|ti2ti|>",
            "<|v2t|>",
            "<|v2s|>",
            "<|s2t|>",
            "<|t2s|>",
            "<|s2s|>",
            "<|soa|>",
            "<|eoa|>",
            "<think>",
            "</think>",
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

    logger.info(f"SPEECHVOCABSTART: {speech_vocab_start}")
    logger.info(
        f"int(config.model.dynin_omni.new_vocab_size): {int(config.model.dynin_omni.new_vocab_size)}"
    )
    logger.info(f"AUDIOCODEBOOKSIZE: {audio_codebook_size}")

    t2s_special_token_ids = {
        "eoa": int(uni_prompting.sptids_dict["<|eoa|>"][0].item()),
        "eos": int(uni_prompting.text_tokenizer.eos_token_id),
    }

    # VQ model for processing image into discrete tokens
    vq_model_image = get_vq_model_class(config.model.vq_model_image.type)
    if config.model.vq_model_image.get("pretrained_model_path", None):
        vq_model_image = vq_model_image().to(accelerator.device)
        state_dict = torch.load(config.model.vq_model_image.pretrained_model_path)[
            "model"
        ]
        vq_model_image.load_state_dict(state_dict)
    else:
        vq_model_image = vq_model_image.from_pretrained(
            vq_image_source,
            local_files_only=model_local_files_only,
        ).to(accelerator.device)

    vq_model_audio = get_vq_model_class(config.model.vq_model_audio.type)
    vq_model_audio = vq_model_audio.from_pretrained(
        vq_audio_source,
        local_files_only=model_local_files_only,
    ).to(accelerator.device)

    vq_model_image.eval()
    vq_model_image.requires_grad_(False)

    vq_model_audio.eval()
    vq_model_audio.requires_grad_(False)

    # Speech-token caching configuration
    speech_cache_cfg = getattr(config.dataset, "speech_token_cache", {})
    if not isinstance(speech_cache_cfg, dict):
        speech_cache_cfg = OmegaConf.to_container(speech_cache_cfg, resolve=True)
    speech_cache_cfg = speech_cache_cfg or {}

    speech_cache_enabled = bool(speech_cache_cfg.get("enable", False))
    speech_cache_dir: Optional[Path]
    if speech_cache_enabled:
        cache_root = speech_cache_cfg.get("root", "cache/speech_tokens")
        speech_cache_dir = Path(cache_root)
        try:
            speech_cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            speech_cache_dir = None
            speech_cache_enabled = False
            logger.warning(
                "Failed to create speech cache directory at %s; disabling cache.",
                cache_root,
            )
    else:
        speech_cache_dir = None

    speech_cache_max_items = int(speech_cache_cfg.get("max_items_in_memory", 4096))
    audio_token_cache_mem: Dict[str, torch.Tensor] = {}

    def _get_audio_cache_path(audio_path: Union[str, Path]) -> Optional[Path]:
        if not isinstance(audio_path, (str, os.PathLike)):
            return None
        if not speech_cache_enabled or speech_cache_dir is None:
            return None
        key = os.path.abspath(str(audio_path))
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        subdir = speech_cache_dir / digest[:2] / digest[2:4]
        return subdir / f"{digest}.pt"

    def _load_cached_audio_tokens(
        audio_path: Union[str, Path],
    ) -> Optional[torch.Tensor]:
        if not isinstance(audio_path, (str, os.PathLike)):
            return None
        cache_key = os.path.abspath(str(audio_path))
        cached = audio_token_cache_mem.get(cache_key)
        if cached is not None:
            return cached.clone()

        cache_path = _get_audio_cache_path(audio_path)
        if cache_path is None or not cache_path.exists():
            return None
        try:
            tokens = torch.load(cache_path, map_location="cpu")
            if isinstance(tokens, torch.Tensor):
                if len(audio_token_cache_mem) < speech_cache_max_items:
                    audio_token_cache_mem[cache_key] = tokens
                return tokens.clone()
        except Exception as exc:
            logger.warning(
                "Failed to load cached speech tokens from %s (%s); ignoring cache.",
                cache_path,
                exc,
            )
        return None

    def _store_cached_audio_tokens(
        audio_path: Union[str, Path], tokens: torch.Tensor
    ) -> None:
        if not isinstance(audio_path, (str, os.PathLike)):
            return
        cache_path = _get_audio_cache_path(audio_path)
        if cache_path is None:
            return
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
            torch.save(tokens.cpu(), tmp_path)
            os.replace(tmp_path, cache_path)
        except Exception as exc:
            logger.warning(
                "Failed to write speech token cache to %s (%s).", cache_path, exc
            )
            return
        cache_key = os.path.abspath(str(audio_path))
        if len(audio_token_cache_mem) < speech_cache_max_items:
            audio_token_cache_mem[cache_key] = tokens.cpu()

    model_load_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    if model_local_files_only:
        model_load_kwargs["local_files_only"] = True
    pretrained_config_path = getattr(dynin_omni_cfg, "pretrained_config_path", None)
    if pretrained_config_path:
        pretrained_config_path = str(pretrained_config_path).strip()
        if pretrained_config_path.lower().endswith((".yaml", ".yml")):
            logger.warning(
                "Ignoring model.dynin_omni.pretrained_config_path=%s because it is YAML. "
                "Provide a HF JSON config path if you need explicit config override.",
                pretrained_config_path,
            )
        else:
            model_load_kwargs["config"] = pretrained_config_path
    model = DyninOmniModelLM.from_pretrained(
        model_source,
        **model_load_kwargs,
    ).to(accelerator.device)
    mask_id = model.config.mask_token_id

    # Set up optimizer and learning-rate scheduler.
    optimizer_config = config.optimizer.params

    # Disable weight decay for bias, layernorm, and embeddings.
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

    # Create mask scheduler
    if config.get("mask_schedule", None) is not None:
        schedule = config.mask_schedule.schedule
        args = config.mask_schedule.get("params", {})
        mask_schedule = get_mask_schedule(schedule, **args)
    else:
        mask_schedule = get_mask_schedule(
            config.training.get("mask_schedule", "cosine")
        )

    # Build dataloaders.
    logger.info("Creating dataloaders and lr_scheduler")

    def build_distributed_sampler(dataset, *, shuffle=True, drop_last=True):
        """Create a DistributedSampler only when running with multiple processes."""
        if dataset is None or accelerator.num_processes <= 1:
            return None
        return DistributedSampler(
            dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    batch_size_t2i_cfg = config.training.batch_size_t2i
    batch_size_lm_cfg = config.training.batch_size_lm
    batch_size_mmu_cfg = config.training.batch_size_mmu
    batch_size_t2s_cfg = config.training.batch_size_t2s
    batch_size_s2t_cfg = config.training.batch_size_s2t
    batch_size_v2t_cfg = config.training.batch_size_v2t

    total_batch_size = (
        total_batch_size_per_gpu
        * accelerator.num_processes
        * config.training.gradient_accumulation_steps
    )
    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params

    pin_memory = bool(getattr(dataset_config, "pin_memory", False))
    persistent_workers = bool(getattr(dataset_config, "persistent_workers", False))
    dataloader_timeout = int(getattr(dataset_config, "dataloader_timeout", 120))

    if persistent_workers and dataloader_timeout > 0:
        logger.warning(
            "persistent_workers=True requires dataloader_timeout=0; overriding timeout=%s",
            dataloader_timeout,
        )
        dataloader_timeout = 0

    if (
        not persistent_workers
        and int(getattr(dataset_config, "num_workers", 0)) > 0
        and str(config.dataset.combined_loader_mode) == "max_size_cycle"
    ):
        logger.warning(
            "Using combined_loader_mode='max_size_cycle' with num_workers>0 and "
            "persistent_workers=False can exhaust OS semaphores when loaders cycle. "
            "Set dataset.params.persistent_workers=True to keep worker processes alive."
        )

    # Common resolution helper for T2I/MMU
    t2i_resolution = getattr(
        dataset_config, "t2i_resolution", dataset_config.resolution
    )

    # Load the multi-image interleaved dataset (MMU-style) first.
    logger.info("Loading MMU dataset")
    dataset_mmu = None
    sampler_mmu = None
    train_dataloader_mmu = None
    if batch_size_mmu_cfg > 0:
        mmu_params = dataset_config.get("mmu_interleaved", {})
        if mmu_params is None:
            mmu_params = {}
        elif not isinstance(mmu_params, dict):
            mmu_params = OmegaConf.to_container(mmu_params, resolve=True)

        if isinstance(mmu_params, list):
            mmu_entries = []
            for entry in mmu_params:
                if isinstance(entry, dict):
                    mmu_entries.append(entry)
                else:
                    mmu_entries.append(OmegaConf.to_container(entry, resolve=True))
        else:
            mmu_entries = [mmu_params]

        mmu_datasets: list[Dataset] = []
        mmu_source_sizes: list[tuple[str, Optional[int]]] = []
        for mmu_kwargs in mmu_entries:
            # Align MMU resolution with T2I resolution if not explicitly set
            if "resolution" not in mmu_kwargs:
                mmu_kwargs["resolution"] = t2i_resolution

            if "jsonl_path" in mmu_kwargs:
                dataset = CambrianInterleavedDataset(**mmu_kwargs)
            else:
                dataset = TextImageInterleavedDataset(**mmu_kwargs)

            mmu_datasets.append(dataset)
            source_name = (
                mmu_kwargs.get("dataset_name")
                or mmu_kwargs.get("jsonl_path")
                or "mmu_source"
            )
            try:
                source_len = len(dataset)
            except Exception:
                source_len = None
            mmu_source_sizes.append((str(source_name), source_len))

        if len(mmu_datasets) == 1:
            dataset_mmu = mmu_datasets[0]
        elif len(mmu_datasets) > 1:
            dataset_mmu = torch.utils.data.ConcatDataset(mmu_datasets)

        if mmu_source_sizes and accelerator.is_main_process:
            parts = []
            for name, count in mmu_source_sizes:
                count_str = str(count) if count is not None else "unknown"
                parts.append(f"{name}={count_str}")
            logger.info("MMU source sizes: %s", ", ".join(parts))

    # Text-to-image / Image-to-image datasets
    logger.info("Loading Text-to-image / Image-to-image datasets")
    dataset_t2i = None
    dataset_i2i = None
    train_dataloader_t2i = None
    train_dataloader_i2i = None
    sampler_t2i: Optional[DistributedSampler] = None  # type: ignore[assignment]
    sampler_i2i: Optional[DistributedSampler] = None  # type: ignore[assignment]
    if batch_size_t2i_cfg > 0:
        raw_t2i_choice = dataset_config.get("t2i_dataset", "hqedit")
        if isinstance(raw_t2i_choice, str):
            split_tokens = [
                token.strip() for token in raw_t2i_choice.replace(",", "+").split("+")
            ]
            dataset_choices = [token for token in split_tokens if token]
        else:
            dataset_choices = [
                str(token).strip() for token in raw_t2i_choice if str(token).strip()
            ]

        if not dataset_choices:
            raise ValueError(
                "t2i_dataset configuration produced no valid dataset names."
            )

        t2i_datasets: list[Dataset] = []
        i2i_datasets: list[Dataset] = []
        t2i_source_names: list[str] = []
        i2i_source_names: list[str] = []
        t2i_source_sizes: list[tuple[str, Optional[int]]] = []
        i2i_source_sizes: list[tuple[str, Optional[int]]] = []
        for choice in dataset_choices:
            choice_lower = choice.lower()
            if choice_lower in {"hqedit", "hq-edit", "hq_edit"}:
                dataset = HQEditX2IDataset(
                    split=dataset_config.get("hqedit_split", "train"),
                    resolution=dataset_config.resolution,
                )
                i2i_datasets.append(dataset)
                logger.info(
                    "Using HQ-Edit dataset for T2I/i2i branch (%s split)",
                    dataset_config.get("hqedit_split", "train"),
                )
                i2i_source_names.append(choice)
                try:
                    i2i_source_sizes.append((choice, len(dataset)))
                except Exception:
                    i2i_source_sizes.append((choice, None))
            elif choice_lower in {"pickapic", "pickapic-v2", "pickapic_v2"}:
                dataset = PickAPicV2Dataset(
                    split=dataset_config.get("t2i_split", "train"),
                    resolution=t2i_resolution,
                    dataset_name=dataset_config.get(
                        "pickapic_dataset_name", "Min-Jaewon/pickapic-v2"
                    ),
                    cache_dir=dataset_config.get("pickapic_cache_dir", None),
                )
                t2i_datasets.append(dataset)
                logger.info(
                    "Using pickapic-v2 dataset for T2I branch (split=%s, dataset=%s)",
                    dataset_config.get("t2i_split", "train"),
                    dataset_config.get(
                        "pickapic_dataset_name", "Min-Jaewon/pickapic-v2"
                    ),
                )
                t2i_source_names.append(choice)
                try:
                    t2i_source_sizes.append((choice, len(dataset)))
                except Exception:
                    t2i_source_sizes.append((choice, None))
            elif choice_lower in {
                "flux_reason",
                "flux-reason",
                "flux_reason_6m",
                "flux-reason-6m",
                "fluxreason",
            }:
                dataset = FluxReasonDataset(
                    split=dataset_config.get("t2i_split", "train"),
                    resolution=t2i_resolution,
                    dataset_name=dataset_config.get(
                        "flux_reason_dataset_name", "LucasFang/FLUX-Reason-6M"
                    ),
                    cache_dir=dataset_config.get("flux_reason_cache_dir", None),
                    score_threshold=float(
                        dataset_config.get("flux_reason_score_threshold", 8.0)
                    ),
                    local_files_only=bool(
                        dataset_config.get("flux_reason_local_files_only", False)
                    ),
                )
                t2i_datasets.append(dataset)
                logger.info(
                    "Using FLUX-Reason-6M dataset for T2I branch (split=%s, dataset=%s, score>%.2f)",
                    dataset_config.get("t2i_split", "train"),
                    dataset_config.get(
                        "flux_reason_dataset_name", "LucasFang/FLUX-Reason-6M"
                    ),
                    float(dataset_config.get("flux_reason_score_threshold", 8.0)),
                )
                t2i_source_names.append(choice)
                try:
                    t2i_source_sizes.append((choice, len(dataset)))
                except Exception:
                    t2i_source_sizes.append((choice, None))
            elif choice_lower in {"ultraedit", "ultra_edit", "ultraedit_500k"}:
                dataset = UltraEditDataset(
                    split="FreeForm",
                    resolution=256,
                    dataset_name=dataset_config.get(
                        "ultraedit_dataset_name", "BleachNick/UltraEdit_500k"
                    ),
                    cache_dir=dataset_config.get("ultraedit_cache_dir", None),
                    local_files_only=bool(
                        dataset_config.get("ultraedit_local_files_only", False)
                    ),
                )
                i2i_datasets.append(dataset)
                logger.info(
                    "Using UltraEdit_500k dataset for image-to-image editing (split=%s, dataset=%s)",
                    "FreeForm",
                    dataset_config.get(
                        "ultraedit_dataset_name", "BleachNick/UltraEdit_500k"
                    ),
                )
                i2i_source_names.append(choice)
                try:
                    i2i_source_sizes.append((choice, len(dataset)))
                except Exception:
                    i2i_source_sizes.append((choice, None))
            elif choice_lower in {"journeydb", "journey_db"}:
                journeydb_jsonl_path = dataset_config.get("journeydb_jsonl_path")
                journeydb_image_root = dataset_config.get("journeydb_image_root")
                if not journeydb_jsonl_path or not journeydb_image_root:
                    raise ValueError(
                        "journeydb requires dataset.params.journeydb_jsonl_path and "
                        "dataset.params.journeydb_image_root to be set."
                    )
                dataset = JourneyDBDataset(
                    jsonl_path=journeydb_jsonl_path,
                    image_root=journeydb_image_root,
                    split=dataset_config.get("t2i_split", "train"),
                    resolution=t2i_resolution,
                    local_files_only=bool(
                        dataset_config.get("journeydb_local_files_only", True)
                    ),
                )
                t2i_datasets.append(dataset)
                logger.info(
                    "Using JourneyDB dataset for T2I branch (jsonl=%s)",
                    journeydb_jsonl_path,
                )
                t2i_source_names.append(choice)
                try:
                    t2i_source_sizes.append((choice, len(dataset)))
                except Exception:
                    t2i_source_sizes.append((choice, None))
            elif choice_lower in {
                "prompt_jsonl",
                "prompt-image-jsonl",
                "prompt_image_jsonl",
                "local_prompt",
            }:
                raw_prompt_cfg = getattr(dataset_config, "prompt_image_jsonl", None)
                if raw_prompt_cfg is None:
                    prompt_cfg = {}
                elif not isinstance(raw_prompt_cfg, dict):
                    prompt_cfg = (
                        OmegaConf.to_container(raw_prompt_cfg, resolve=True) or {}
                    )
                else:
                    prompt_cfg = raw_prompt_cfg

                jsonl_path = (
                    prompt_cfg.get("jsonl_path")
                    or dataset_config.get("prompt_image_jsonl_path")
                    or dataset_config.get("local_prompt_jsonl_path")
                )
                if not jsonl_path:
                    raise ValueError(
                        "prompt_image_jsonl requires jsonl_path to be set."
                    )

                dataset = PromptImageJsonlDataset(
                    jsonl_path=jsonl_path,
                    resolution=t2i_resolution,
                    prompt_keys=prompt_cfg.get("prompt_keys", ("prompt", "query")),
                    image_keys=prompt_cfg.get("image_keys", ("image_path", "image")),
                    skip_missing=bool(prompt_cfg.get("skip_missing", True)),
                    cache_path=prompt_cfg.get("cache_path"),
                    max_samples=prompt_cfg.get("max_samples"),
                    seed=int(
                        prompt_cfg.get(
                            "seed", getattr(config.training, "seed", 42) or 42
                        )
                    ),
                )
                t2i_datasets.append(dataset)
                logger.info(
                    "Using prompt-image JSONL dataset for T2I branch (jsonl=%s)",
                    jsonl_path,
                )
                t2i_source_names.append(choice)
                try:
                    t2i_source_sizes.append((choice, len(dataset)))
                except Exception:
                    t2i_source_sizes.append((choice, None))
            elif choice_lower in {"dpg_jsonl", "dpg-prompt-jsonl", "dpg_prompt_jsonl"}:
                raw_dpg_cfg = getattr(dataset_config, "dpg_jsonl", None)
                if raw_dpg_cfg is None:
                    dpg_cfg = {}
                elif not isinstance(raw_dpg_cfg, dict):
                    dpg_cfg = OmegaConf.to_container(raw_dpg_cfg, resolve=True) or {}
                else:
                    dpg_cfg = raw_dpg_cfg

                jsonl_path = dpg_cfg.get("jsonl_path") or dataset_config.get(
                    "dpg_jsonl_path"
                )
                if not jsonl_path:
                    raise ValueError("dpg_jsonl requires jsonl_path to be set.")

                dataset = PromptImageJsonlDataset(
                    jsonl_path=jsonl_path,
                    resolution=t2i_resolution,
                    prompt_keys=dpg_cfg.get("prompt_keys", ("prompt", "query")),
                    image_keys=dpg_cfg.get("image_keys", ("image_path", "image")),
                    skip_missing=bool(dpg_cfg.get("skip_missing", True)),
                    cache_path=dpg_cfg.get("cache_path"),
                    max_samples=dpg_cfg.get("max_samples"),
                    seed=int(
                        dpg_cfg.get("seed", getattr(config.training, "seed", 42) or 42)
                    ),
                )
                t2i_datasets.append(dataset)
                logger.info(
                    "Using DPG JSONL dataset for T2I branch (jsonl=%s)",
                    jsonl_path,
                )
                t2i_source_names.append(choice)
                try:
                    t2i_source_sizes.append((choice, len(dataset)))
                except Exception:
                    t2i_source_sizes.append((choice, None))
            elif choice_lower in {
                "basic_edit_jsonl",
                "i2i_prompt_jsonl",
                "i2i_prompt_image_jsonl",
            }:
                raw_edit_cfg = getattr(dataset_config, "i2i_prompt_image_jsonl", None)
                if raw_edit_cfg is None:
                    raw_edit_cfg = getattr(dataset_config, "prompt_image_jsonl", None)
                if raw_edit_cfg is None:
                    edit_cfg = {}
                elif not isinstance(raw_edit_cfg, dict):
                    edit_cfg = OmegaConf.to_container(raw_edit_cfg, resolve=True) or {}
                else:
                    edit_cfg = raw_edit_cfg

                jsonl_path = (
                    edit_cfg.get("jsonl_path")
                    or dataset_config.get("i2i_prompt_image_jsonl_path")
                    or dataset_config.get("basic_edit_jsonl_path")
                )
                if not jsonl_path:
                    raise ValueError("basic_edit_jsonl requires jsonl_path to be set.")

                dataset = BasicEditJsonlDataset(
                    jsonl_path=jsonl_path,
                    resolution=dataset_config.resolution,
                    prompt_keys=edit_cfg.get("prompt_keys", ("prompt", "query")),
                    image_keys=edit_cfg.get("image_keys", ("image_path", "image")),
                    skip_missing=bool(edit_cfg.get("skip_missing", True)),
                    cache_path=edit_cfg.get("cache_path"),
                    max_samples=edit_cfg.get("max_samples"),
                    seed=int(
                        edit_cfg.get("seed", getattr(config.training, "seed", 42) or 42)
                    ),
                )
                i2i_datasets.append(dataset)
                logger.info(
                    "Using basic-edit JSONL dataset for I2I branch (jsonl=%s)",
                    jsonl_path,
                )
                i2i_source_names.append(choice)
                try:
                    i2i_source_sizes.append((choice, len(dataset)))
                except Exception:
                    i2i_source_sizes.append((choice, None))
            else:
                raise ValueError(f"Unsupported t2i_dataset '{choice}'")

        if t2i_datasets:
            dataset_t2i = (
                t2i_datasets[0]
                if len(t2i_datasets) == 1
                else CombinedX2IDataset(t2i_datasets)
            )
            logger.info(
                "T2I dataloading sources: %s",
                ", ".join(t2i_source_names) if t2i_source_names else "n/a",
            )
            if t2i_source_sizes and accelerator.is_main_process:
                parts = []
                for name, count in t2i_source_sizes:
                    count_str = str(count) if count is not None else "unknown"
                    parts.append(f"{name}={count_str}")
                logger.info("T2I source sizes: %s", ", ".join(parts))

            sampler_t2i = build_distributed_sampler(
                dataset_t2i,
                shuffle=True,
                drop_last=True,
            )

            train_dataloader_t2i = DataLoader(
                dataset_t2i,
                batch_size=batch_size_t2i_cfg,
                sampler=sampler_t2i,
                shuffle=sampler_t2i is None,
                num_workers=dataset_config.num_workers,
                collate_fn=collate_fn_x2i,
                drop_last=True,
                pin_memory=pin_memory,
                timeout=dataloader_timeout,
                persistent_workers=persistent_workers,
            )

        if i2i_datasets:
            dataset_i2i = (
                i2i_datasets[0]
                if len(i2i_datasets) == 1
                else CombinedX2IDataset(i2i_datasets)
            )
            logger.info(
                "I2I dataloading sources: %s",
                ", ".join(i2i_source_names) if i2i_source_names else "n/a",
            )
            if i2i_source_sizes and accelerator.is_main_process:
                parts = []
                for name, count in i2i_source_sizes:
                    count_str = str(count) if count is not None else "unknown"
                    parts.append(f"{name}={count_str}")
                logger.info("I2I source sizes: %s", ", ".join(parts))

            sampler_i2i = build_distributed_sampler(
                dataset_i2i,
                shuffle=True,
                drop_last=True,
            )

            train_dataloader_i2i = DataLoader(
                dataset_i2i,
                batch_size=batch_size_t2i_cfg,
                sampler=sampler_i2i,
                shuffle=sampler_i2i is None,
                num_workers=dataset_config.num_workers,
                collate_fn=collate_fn_x2i,
                drop_last=True,
                pin_memory=pin_memory,
                timeout=dataloader_timeout,
                persistent_workers=persistent_workers,
            )

    # Language modeling dataset (HF instruction mixture or local GSM8K aug)
    logger.info("Loading LM dataset")
    dataset_lm = None
    train_dataloader_lm = None
    if batch_size_lm_cfg > 0:
        reasoning_cfg = getattr(dataset_config, "reasoning_sft_csv", {})
        if not isinstance(reasoning_cfg, dict):
            reasoning_cfg = OmegaConf.to_container(reasoning_cfg, resolve=True)
        reasoning_cfg = reasoning_cfg or {}

        instruction_cfg = getattr(dataset_config, "hf_instruction_lm", {})
        if not isinstance(instruction_cfg, dict):
            instruction_cfg = OmegaConf.to_container(instruction_cfg, resolve=True)
        instruction_cfg = instruction_cfg or {}

        seed_lm = instruction_cfg.get("seed")
        if seed_lm is None:
            seed_lm = getattr(config.training, "seed", 42) or 42

        collate_fn_lm = None
        if reasoning_cfg.get("csv_path"):
            reasoning_seed = reasoning_cfg.get("seed", seed_lm)
            dataset_lm = ReasoningSFTCSVDataset(
                csv_path=reasoning_cfg.get("csv_path"),
                seed=int(reasoning_seed),
                max_total_samples=reasoning_cfg.get("max_total_samples"),
            )
            collate_fn_lm = dataset_lm.collate_fn
        else:
            dataset_lm = HFInstructionTextDataset(
                split=instruction_cfg.get("split", "train"),
                max_samples_per_source=instruction_cfg.get("max_samples_per_source"),
                max_total_samples=instruction_cfg.get("max_total_samples"),
                seed=int(seed_lm),
                sources=instruction_cfg.get("sources"),
            )
            collate_fn_lm = dataset_lm.collate_fn

        sampler_lm = build_distributed_sampler(
            dataset_lm,
            shuffle=True,
            drop_last=True,
        )

        train_dataloader_lm = DataLoader(
            dataset_lm,
            batch_size=batch_size_lm_cfg,
            sampler=sampler_lm,
            shuffle=sampler_lm is None,
            collate_fn=collate_fn_lm,
            num_workers=dataset_config.num_workers,
            drop_last=True,
            pin_memory=pin_memory,
            timeout=dataloader_timeout,
            persistent_workers=persistent_workers,
        )

    # Video Dataset
    logger.info("Loading Video dataset")
    dataset_v2t = None
    train_dataloader_v2t = None
    sampler_v2t = None
    speech_cfg = getattr(dataset_config, "video_speech_dataset", {})
    if not isinstance(speech_cfg, dict):
        speech_cfg = OmegaConf.to_container(speech_cfg, resolve=True)
    speech_cfg = speech_cfg or {}

    if batch_size_v2t_cfg > 0:
        v2t_sample_method = speech_cfg.get(
            "v2t_sample_method",
            speech_cfg.get("sample_method", "uniform"),
        )
        llavavid_max_seconds_cfg = speech_cfg.get("llavavid_max_video_seconds")
        if llavavid_max_seconds_cfg is not None:
            try:
                llavavid_max_seconds_cfg = float(llavavid_max_seconds_cfg)
            except (TypeError, ValueError):
                llavavid_max_seconds_cfg = None
        dataset_v2t_candidates: list[Dataset] = []
        v2t_source_sizes: list[tuple[str, Optional[int]]] = []
        if speech_cfg.get("use_llavavid", True):
            dataset = VideoCaptionDataset(
                transform=image_transform,
                tokenizer=uni_prompting.text_tokenizer,
                max_seq_length=max_seq_mmu_input,
                resolution=preproc_config.resolution,
                sample_method=v2t_sample_method,
                dataset_name=speech_cfg.get("llavavid_dataset_name", "llavavid"),
                llavavid_path=speech_cfg.get(
                    "llavavid_path", "lmms-lab/LLaVA-Video-178K"
                ),
                num_frames=5,
                llavavid_local_files_only=bool(
                    speech_cfg.get("llavavid_local_files_only", False)
                ),
                llavavid_skip_configs=speech_cfg.get("llavavid_skip_configs"),
                llavavid_skip_video_patterns=speech_cfg.get(
                    "llavavid_skip_video_patterns"
                ),
                llavavid_max_samples=speech_cfg.get("llavavid_max_samples"),
                llavavid_sample_seed=speech_cfg.get("llavavid_sample_seed", 42),
                max_video_seconds=llavavid_max_seconds_cfg,
            )
            dataset_v2t_candidates.append(dataset)
            try:
                v2t_source_sizes.append(("llavavid", len(dataset)))
            except Exception:
                v2t_source_sizes.append(("llavavid", None))

        sharegpt_path = speech_cfg.get("sharegptvideo_sft_path")
        sharegpt_paths: list[str] = []
        if isinstance(sharegpt_path, (list, tuple, ListConfig)):
            sharegpt_paths = [str(p) for p in sharegpt_path if p]
        elif sharegpt_path:
            sharegpt_paths = [str(sharegpt_path)]
        for sharegpt_path in sharegpt_paths:
            sharegpt_num_frames = speech_cfg.get("sharegptvideo_num_frames", 5)
            sharegpt_sample_method = speech_cfg.get(
                "sharegptvideo_sample_method", v2t_sample_method
            )
            sharegpt_frame_exts = speech_cfg.get("sharegptvideo_frame_exts")
            sharegpt_strip_token = bool(
                speech_cfg.get("sharegptvideo_strip_video_token", True)
            )
            sharegpt_require_video = bool(
                speech_cfg.get("sharegptvideo_require_video", True)
            )
            dataset = ShareGPTVideoSFTDataset(
                jsonl_path=sharegpt_path,
                transform=image_transform,
                resolution=preproc_config.resolution,
                num_frames=int(sharegpt_num_frames),
                sample_method=sharegpt_sample_method,
                strip_video_token=sharegpt_strip_token,
                frame_exts=sharegpt_frame_exts,
                require_video=sharegpt_require_video,
            )
            dataset_v2t_candidates.append(dataset)
            try:
                v2t_source_sizes.append(
                    (f"sharegptvideo:{os.path.basename(sharegpt_path)}", len(dataset))
                )
            except Exception:
                v2t_source_sizes.append(
                    (f"sharegptvideo:{os.path.basename(sharegpt_path)}", None)
                )

        if not dataset_v2t_candidates:
            logger.warning("No V2T datasets configured; disabling V2T loader.")
        elif len(dataset_v2t_candidates) == 1:
            dataset_v2t = dataset_v2t_candidates[0]
        else:
            dataset_v2t = ConcatDataset(dataset_v2t_candidates)

        if dataset_v2t is not None:
            if v2t_source_sizes and accelerator.is_main_process:
                parts = []
                for name, count in v2t_source_sizes:
                    count_str = str(count) if count is not None else "unknown"
                    parts.append(f"{name}={count_str}")
                logger.info("V2T source sizes: %s", ", ".join(parts))
            sampler_v2t = build_distributed_sampler(
                dataset_v2t,
                shuffle=True,
                drop_last=True,
            )

            train_dataloader_v2t = DataLoader(
                dataset_v2t,
                batch_size=batch_size_v2t_cfg,
                num_workers=dataset_config.num_workers,
                collate_fn=collate_fn_v2t,
                sampler=sampler_v2t,
                shuffle=sampler_v2t is None,
                drop_last=True,
                pin_memory=pin_memory,
                timeout=dataloader_timeout,
                persistent_workers=persistent_workers,
            )

    # Speech Dataset
    speech_enabled = (
        config.training.batch_size_s2t > 0 or config.training.batch_size_t2s > 0
    )
    if speech_enabled:
        logger.info("Loading Speech dataset")
        dataset_sm = MixedSpeechTextDataset(config.dataset.params.audio_data)
        if len(dataset_sm) == 0:
            logger.warning("Speech dataset is empty; disabling s2t/t2s.")
            dataset_sm = None
    else:
        logger.info("Skipping Speech dataset (batch_size_s2t and batch_size_t2s are 0)")
        dataset_sm = None

    logger.info("Dataset Prepared.")
    require_cached_audio_tokens = bool(
        getattr(config.dataset.params, "require_cached_audio_tokens", False)
    )

    def _prepare_audio_flow(paths, tokens, texts):
        """Align path/token/text lists and optionally drop samples without cached tokens."""
        path_list = (
            list(paths) if isinstance(paths, (list, tuple)) else list(paths or [])
        )
        token_iterable = (
            tokens if isinstance(tokens, (list, tuple)) else list(tokens or [])
        )
        text_iterable = texts if isinstance(texts, (list, tuple)) else list(texts or [])

        triplets: list[tuple[Any, Any, str]] = []
        for path, token, text in zip_longest(
            path_list, token_iterable, text_iterable, fillvalue=None
        ):
            if path is None:
                continue
            triplets.append((path, token, text if text is not None else ""))

        skipped = 0
        if require_cached_audio_tokens and triplets:
            filtered = [(p, t, txt) for (p, t, txt) in triplets if t is not None]
            skipped = len(triplets) - len(filtered)
            triplets = filtered

        if not triplets:
            return [], [], [], skipped

        aligned_paths = [p for (p, _, _) in triplets]
        aligned_tokens = [t for (_, t, _) in triplets]
        aligned_texts = [txt for (_, _, txt) in triplets]
        return aligned_paths, aligned_tokens, aligned_texts, skipped

    # Use distinct DistributedSamplers for each speech dataloader to avoid iterator interference
    sampler_s2t = None
    sampler_t2s = None
    if accelerator.num_processes > 1:
        if dataset_sm is not None:
            sampler_s2t = DistributedSampler(
                dataset_sm,
                num_replicas=accelerator.num_processes,
                rank=accelerator.process_index,
                shuffle=True,
                drop_last=True,
            )
            sampler_t2s = DistributedSampler(
                dataset_sm,
                num_replicas=accelerator.num_processes,
                rank=accelerator.process_index,
                shuffle=True,
                drop_last=True,
            )
        if dataset_mmu is not None:
            sampler_mmu = DistributedSampler(
                dataset_mmu,
                num_replicas=accelerator.num_processes,
                rank=accelerator.process_index,
                shuffle=True,
                drop_last=True,
            )
    else:
        sampler_mmu = None

    train_dataloader_s2t = None
    train_dataloader_t2s = None
    if dataset_sm is not None and config.training.batch_size_s2t > 0:
        train_dataloader_s2t = DataLoader(
            dataset_sm,
            batch_size=config.training.batch_size_s2t,
            shuffle=False,
            sampler=sampler_s2t,
            collate_fn=collate_fn_audio,
            num_workers=config.dataset.params.num_workers,
            drop_last=True,
            pin_memory=pin_memory,
            timeout=dataloader_timeout,
            persistent_workers=persistent_workers,
        )
    if dataset_sm is not None and config.training.batch_size_t2s > 0:
        train_dataloader_t2s = DataLoader(
            dataset_sm,
            batch_size=config.training.batch_size_t2s,
            shuffle=False,
            sampler=sampler_t2s,
            collate_fn=collate_fn_audio,
            num_workers=config.dataset.params.num_workers,
            drop_last=True,
            pin_memory=pin_memory,
            timeout=dataloader_timeout,
            persistent_workers=persistent_workers,
        )

    if dataset_mmu is not None:
        train_dataloader_mmu = DataLoader(
            dataset_mmu,
            batch_size=config.training.batch_size_mmu,
            shuffle=False,
            sampler=sampler_mmu,
            collate_fn=collate_fn_mmu_mult,
            num_workers=config.dataset.params.num_workers,
            drop_last=True,
            pin_memory=pin_memory,
            timeout=dataloader_timeout,
            persistent_workers=persistent_workers,
        )

    # Combine these dataloaders into a single iterable model
    iterables = {}
    if train_dataloader_lm is not None:
        iterables["lm_flow"] = train_dataloader_lm
    if train_dataloader_mmu is not None:
        iterables["mmu_flow"] = train_dataloader_mmu

    if not iterables:
        combined_dataloader = None
        if accelerator.is_main_process:
            logger.warning(
                "CombinedLoader has no non-speech iterables; running with T2I/I2I-only batches."
            )
    else:
        combined_dataloader = CombinedLoader(
            iterables, mode=config.dataset.combined_loader_mode
        )

    def _num_steps(dataset_obj, batch_size_cfg):
        if dataset_obj is None or batch_size_cfg <= 0:
            return 0
        total_bs = (
            batch_size_cfg
            * accelerator.num_processes
            * config.training.gradient_accumulation_steps
        )
        if total_bs <= 0:
            return 0
        length = len(dataset_obj)
        if length == 0:
            return 0
        return math.ceil(length / total_bs)

    num_update_steps_per_epoch_t2i = _num_steps(
        dataset_t2i, config.training.batch_size_t2i
    )
    num_update_steps_per_epoch_i2i = _num_steps(
        dataset_i2i, config.training.batch_size_t2i
    )
    num_update_steps_per_epoch_lm = _num_steps(
        dataset_lm, config.training.batch_size_lm
    )
    num_update_steps_per_epoch_s2t = _num_steps(
        dataset_sm, config.training.batch_size_s2t
    )
    num_update_steps_per_epoch_t2s = _num_steps(
        dataset_sm, config.training.batch_size_t2s
    )
    num_update_steps_per_epoch_v2t = _num_steps(dataset_v2t, batch_size_v2t_cfg)
    num_update_steps_per_epoch_mmu = _num_steps(
        dataset_mmu, config.training.batch_size_mmu
    )

    # Calculate num_train_epochs
    num_update_steps_per_epoch = max(
        num_update_steps_per_epoch_t2i,
        num_update_steps_per_epoch_lm,
        num_update_steps_per_epoch_s2t,
        num_update_steps_per_epoch_t2s,
        num_update_steps_per_epoch_v2t,
        num_update_steps_per_epoch_mmu,
        num_update_steps_per_epoch_i2i,
    )

    num_train_epochs = (
        math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)
        if num_update_steps_per_epoch > 0
        else 1
    )

    logger.info(f"len of T2I: {len(dataset_t2i) if dataset_t2i is not None else 0}")
    logger.info(f"len of I2I: {len(dataset_i2i) if dataset_i2i is not None else 0}")
    logger.info(f"len of LM: {len(dataset_lm) if dataset_lm is not None else 0}")
    logger.info(f"len of Speech: {len(dataset_sm) if dataset_sm is not None else 0}")
    logger.info(
        f"len of Video Caption: {len(dataset_v2t) if dataset_v2t is not None else 0}"
    )
    logger.info(f"len of MMU: {len(dataset_mmu) if dataset_mmu is not None else 0}")

    logger.info(f"Train steps: {config.training.max_train_steps}")
    logger.info(f"Num train epochs: {num_train_epochs}")

    # Resume model state.
    global_step = 0
    first_epoch = 0
    start_step = 0
    micro_step = 0

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
            micro_step = global_step * config.training.gradient_accumulation_steps
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
            # Load a sharded safetensors checkpoint when available.
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

    # Prepare accelerator state.
    logger.info("Preparing model, optimizer and dataloaders")
    model, optimizer = accelerator.prepare(model, optimizer)

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
        min_lr_scale=config.lr_scheduler.params.min_lr_scale,
    )

    vq_model_image.to(device=accelerator.device)
    vq_model_audio.to(device=accelerator.device)

    mask_dtype = model.get_input_embeddings().weight.dtype

    def _log_and_flag_failure(message: str, exc: Exception = None):
        """Log preprocessing failures on both logger and accelerator console."""
        if exc is not None:
            logger.exception(message)
        else:
            logger.error(message)
        accelerator.print(message)

    def _maybe_trim_audio_file(
        audio_path: Union[str, os.PathLike], max_duration: float
    ) -> tuple[Union[str, os.PathLike], Optional[str]]:
        """Return a path to an audio file trimmed to max_duration seconds.

        If trimming succeeds, returns (trimmed_path, temp_path) where trimmed_path is the
        file to use for encoding and temp_path should be deleted afterwards. If trimming
        fails, returns (audio_path, None).
        """
        if max_duration <= 0:
            return audio_path, None
        trim_timeout = float(
            getattr(config.dataset.preprocessing, "audio_trim_timeout_sec", 30.0)
        )
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp.name
            tmp.close()
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(audio_path),
                "-t",
                str(max_duration),
                "-c",
                "copy",
                tmp_path,
            ]
            subprocess.run(cmd, check=True, timeout=trim_timeout)
            return tmp_path, tmp_path
        except Exception as exc:
            warnings.warn(
                f"Failed to trim audio {audio_path} to {max_duration}s: {exc}"
            )
            try:
                if "tmp_path" in locals() and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            return audio_path, None

    def _format_path_for_log(path: Union[str, os.PathLike, torch.Tensor, None]) -> str:
        if isinstance(path, (str, os.PathLike)):
            try:
                return os.fspath(path)
            except TypeError:
                return str(path)
        if isinstance(path, torch.Tensor):
            return f"<tensor shape={tuple(path.shape)}>"
        if isinstance(path, np.ndarray):
            return f"<ndarray shape={path.shape}>"
        if isinstance(path, Sequence) and not isinstance(
            path, (str, bytes, os.PathLike)
        ):
            try:
                return f"<token-seq len={len(path)}>"
            except Exception:
                return "<token-seq>"
        return repr(path)

    def safe_audio_encode(
        audio_path: Union[str, torch.Tensor, np.ndarray, Sequence[int]], flow_name: str
    ):
        if isinstance(audio_path, torch.Tensor):
            return audio_path.cpu().clone(), None
        if isinstance(audio_path, np.ndarray):
            try:
                tensor = torch.from_numpy(audio_path).to(dtype=torch.long)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to convert numpy audio tokens to tensor for flow '{flow_name}': {exc}"
                ) from exc
            return tensor, None
        if isinstance(audio_path, Sequence) and not isinstance(
            audio_path, (str, bytes, os.PathLike)
        ):
            try:
                tensor = torch.as_tensor(audio_path, dtype=torch.long)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to convert cached audio tokens to tensor for flow '{flow_name}': {exc}"
                ) from exc
            return tensor, None
        path_repr = _format_path_for_log(audio_path)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[rank %s] (%s) audio encode request: %s",
                accelerator.process_index,
                flow_name,
                path_repr,
            )

        max_retries = int(
            getattr(config.dataset.preprocessing, "audio_encode_max_retries", 3)
        )
        backoff = float(
            getattr(config.dataset.preprocessing, "audio_encode_retry_backoff_sec", 0.5)
        )
        duration_limit = float(
            getattr(config.dataset.preprocessing, "max_audio_duration_sec", 15.0)
        )

        cached = _load_cached_audio_tokens(audio_path)
        if cached is not None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[rank %s] (%s) audio encode hit cache: %s",
                    accelerator.process_index,
                    flow_name,
                    path_repr,
                )
            return cached, None

        for attempt in range(1, max_retries + 1):
            trimmed_path: Union[str, os.PathLike] = audio_path
            temp_path: Optional[str] = None
            try:
                if isinstance(audio_path, (str, os.PathLike)):
                    trimmed_path, temp_path = _maybe_trim_audio_file(
                        audio_path, duration_limit
                    )
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "[rank %s] (%s) audio encode attempt %d/%d (trimmed=%s): %s",
                            accelerator.process_index,
                            flow_name,
                            attempt,
                            max_retries,
                            "yes" if temp_path is not None else "no",
                            _format_path_for_log(trimmed_path),
                        )
                tokens = vq_model_audio.encode(str(trimmed_path)).cpu()
                _store_cached_audio_tokens(audio_path, tokens)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "[rank %s] (%s) audio encode success: %s",
                        accelerator.process_index,
                        flow_name,
                        path_repr,
                    )
                return tokens, None
            except Exception as exc:
                if attempt == max_retries:
                    msg = (
                        f"[Rank {accelerator.process_index}] {flow_name} audio encode failed "
                        f"for '{audio_path}': {exc}"
                    )
                    _log_and_flag_failure(msg, exc)
                    return None, msg
                sleep_time = min(backoff * attempt, 2.0)
                time.sleep(sleep_time)
            finally:
                if temp_path is not None and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass

    def safe_video_get_code(video_tensor_sample: torch.Tensor, sample_index: int):
        max_retries = int(
            getattr(config.dataset.preprocessing, "video_encode_max_retries", 3)
        )
        backoff = float(
            getattr(config.dataset.preprocessing, "video_encode_retry_backoff_sec", 0.5)
        )
        for attempt in range(1, max_retries + 1):
            try:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "[rank %s] video encode request sample=%d attempt=%d/%d",
                        accelerator.process_index,
                        sample_index,
                        attempt,
                        max_retries,
                    )
                video_token = vq_model_image.get_code(video_tensor_sample)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "[rank %s] video encode success sample=%d",
                        accelerator.process_index,
                        sample_index,
                    )
                return video_token, None
            except Exception as exc:
                if attempt == max_retries:
                    msg = (
                        f"[Rank {accelerator.process_index}] v2t video encode failed "
                        f"for sample index {sample_index}: {exc}"
                    )
                    _log_and_flag_failure(msg, exc)
                    return None, msg
                logger.warning(
                    "[rank %s] video encode retry sample=%d attempt=%d/%d error=%s",
                    accelerator.process_index,
                    sample_index,
                    attempt,
                    max_retries,
                    exc,
                )
                sleep_time = min(backoff * attempt, 2.0)
                time.sleep(sleep_time)

    def safe_image_get_code(image_tensor_sample: torch.Tensor, sample_index: int):
        max_retries = int(
            getattr(config.dataset.preprocessing, "image_encode_max_retries", 3)
        )
        backoff = float(
            getattr(config.dataset.preprocessing, "image_encode_retry_backoff_sec", 0.5)
        )
        for attempt in range(1, max_retries + 1):
            try:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "[rank %s] image encode request sample=%d attempt=%d/%d",
                        accelerator.process_index,
                        sample_index,
                        attempt,
                        max_retries,
                    )
                if image_tensor_sample.dim() == 3:
                    image_tensor_sample = image_tensor_sample.unsqueeze(0)
                elif image_tensor_sample.dim() != 4:
                    raise ValueError(
                        f"Expected image tensor with 3 or 4 dims, got shape {tuple(image_tensor_sample.shape)}"
                    )
                image_token = vq_model_image.get_code(image_tensor_sample)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "[rank %s] image encode success sample=%d",
                        accelerator.process_index,
                        sample_index,
                    )
                return image_token, None
            except Exception as exc:
                if attempt == max_retries:
                    msg = (
                        f"[Rank {accelerator.process_index}] image encode failed "
                        f"for sample index {sample_index}: {exc}"
                    )
                    _log_and_flag_failure(msg, exc)
                    return None, msg
                logger.warning(
                    "[rank %s] image encode retry sample=%d attempt=%d/%d error=%s",
                    accelerator.process_index,
                    sample_index,
                    attempt,
                    max_retries,
                    exc,
                )
                sleep_time = min(backoff * attempt, 2.0)
                time.sleep(sleep_time)

    # Run training.
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {total_batch_size_per_gpu}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}"
    )

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
        # create MLM mask and labels
        input_ids, labels, loss_weight, mask_prob = mask_or_random_replace_tokens(
            image_tokens,
            mask_id,
            config,
            mask_schedule=mask_schedule,
            is_train=is_train,
        )
        input_ids, masks, labels = uni_prompting(
            (texts, input_ids, labels),
            "t2i",
            config={"max_text_len_override": max_seq_t2i},
        )
        return input_ids, labels, mask_prob, image_tokens, masks

    @torch.no_grad()
    def prepare_inputs_and_labels_for_i2i(
        source_images: torch.FloatTensor,
        target_images: torch.FloatTensor,
        prompts: list[str],
        is_train: bool = True,
        target_texts: Optional[list[str]] = None,
    ):
        """Build masked i2i sequences from source/target image pairs."""

        # Tokenize source/target images with VQ model and offset by text vocab size
        source_tokens = vq_model_image.get_code(source_images) + len(
            uni_prompting.text_tokenizer
        )
        target_tokens = vq_model_image.get_code(target_images) + len(
            uni_prompting.text_tokenizer
        )

        cond_dropout_prob = config.training.get(
            "i2i_cond_dropout_prob",
            config.training.cond_dropout_prob,
        )

        if (
            is_train
            and torch.rand(1, device=source_tokens.device).item() < cond_dropout_prob
        ):
            effective_prompts = [""] * len(prompts)
            masked_target_source = source_tokens
        else:
            effective_prompts = list(prompts)
            masked_target_source = target_tokens

        masked_target_tokens, labels, _, mask_prob = mask_or_random_replace_tokens(
            masked_target_source,
            mask_id,
            config,
            mask_schedule=mask_schedule,
            is_train=is_train,
        )

        input_ids, attention_masks, labels = uni_prompting(
            (effective_prompts, source_tokens, masked_target_tokens, labels),
            "i2i",
            config={"max_text_len_override": max_seq_t2i},
        )

        return input_ids, labels, mask_prob, attention_masks

    @torch.no_grad()
    def prepare_inputs_and_labels_for_text(
        texts: Union[str, str], max_seq_len_out, eps=1e-3
    ):
        # create MLM mask and labels
        # truncate inputs to max_seq_lm_input, but pad/label to max_seq_len_out
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_lm_input,
        )["input_ids"]
        input_ids_lm, prompt_mask, labels_lm = uni_prompting.lm_chat_prompt(
            tokenized, max_seq_len_out
        )
        if prompt_mask.numel() > 0:
            labels_lm = labels_lm.clone()
            labels_lm[prompt_mask.bool()] = -100
        b, l = input_ids_lm.shape
        t = torch.rand(b, device=input_ids_lm.device)
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, l)

        rand_mask = torch.rand((b, l), device=input_ids_lm.device) < p_mask
        # ensure at least one mask per sample
        empty_mask = ~rand_mask.any(dim=1)
        if empty_mask.any():
            fallback_positions = torch.randint(
                0, l, (empty_mask.sum().item(),), device=input_ids_lm.device
            )
            rand_mask[empty_mask, fallback_positions] = True

        noisy_batch = torch.where(rand_mask, mask_id, input_ids_lm)
        noisy_batch[prompt_mask.bool()] = input_ids_lm[prompt_mask.bool()]
        masked_indices = noisy_batch == mask_id

        return noisy_batch, labels_lm, p_mask

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
        input_ids_t2s, prompt_masks, labels_t2s, eps=1e-3
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

    v2t_iterator: Optional[Iterator] = None
    t2i_iterator: Optional[Iterator] = None
    i2i_iterator: Optional[Iterator] = None

    def _next_from_v2t():
        nonlocal v2t_iterator
        if train_dataloader_v2t is None:
            return None
        try:
            return next(v2t_iterator)
        except StopIteration:
            v2t_iterator = iter(train_dataloader_v2t)
            return next(v2t_iterator)

    def _next_from_t2i():
        nonlocal t2i_iterator
        if train_dataloader_t2i is None:
            return None
        try:
            return next(t2i_iterator)
        except StopIteration:
            t2i_iterator = iter(train_dataloader_t2i)
            return next(t2i_iterator)

    def _next_from_i2i():
        nonlocal i2i_iterator
        if train_dataloader_i2i is None:
            return None
        try:
            return next(i2i_iterator)
        except StopIteration:
            i2i_iterator = iter(train_dataloader_i2i)
            return next(i2i_iterator)

    def _next_from_s2t():
        nonlocal s2t_iterator
        if train_dataloader_s2t is None:
            return None
        try:
            return next(s2t_iterator)
        except StopIteration:
            s2t_iterator = iter(train_dataloader_s2t)
            return next(s2t_iterator)

    def _next_from_t2s():
        nonlocal t2s_iterator
        if train_dataloader_t2s is None:
            return None
        try:
            return next(t2s_iterator)
        except StopIteration:
            t2s_iterator = iter(train_dataloader_t2s)
            return next(t2s_iterator)

    v2t_iterator = (
        iter(train_dataloader_v2t) if train_dataloader_v2t is not None else None
    )
    t2i_iterator = (
        iter(train_dataloader_t2i) if train_dataloader_t2i is not None else None
    )
    i2i_iterator = (
        iter(train_dataloader_i2i) if train_dataloader_i2i is not None else None
    )
    s2t_iterator = (
        iter(train_dataloader_s2t) if train_dataloader_s2t is not None else None
    )
    t2s_iterator = (
        iter(train_dataloader_t2s) if train_dataloader_t2s is not None else None
    )
    log_every = config.experiment.get("log_every", None)
    save_every = config.experiment.get("save_every", None)
    eval_every = config.experiment.get("eval_every", None)

    for epoch in tqdm(
        range(first_epoch, num_train_epochs),
        desc="Epochs",
        disable=not accelerator.is_main_process,
        position=0,
    ):
        # Ensure all samplers reshuffle in a rank-consistent way each epoch
        try:
            if isinstance(sampler_t2i, DistributedSampler):
                sampler_t2i.set_epoch(epoch)
            if isinstance(sampler_i2i, DistributedSampler):
                sampler_i2i.set_epoch(epoch)
            if isinstance(sampler_v2t, DistributedSampler):
                sampler_v2t.set_epoch(epoch)
            if accelerator.num_processes > 1:
                if sampler_s2t is not None:
                    sampler_s2t.set_epoch(epoch)
                if sampler_t2s is not None:
                    sampler_t2s.set_epoch(epoch)
        except Exception:
            pass
        model.train()
        combined_iterator = (
            iter(range(num_update_steps_per_epoch))
            if combined_dataloader is None
            else iter(combined_dataloader)
        )
        while True:
            skip_local = 0
            timeout_encountered = False
            timeout_message: Optional[str] = None
            try:
                if combined_dataloader is None:
                    next(combined_iterator)
                    batch = {}
                    batch_idx = None
                    dataloader_idx = None
                else:
                    batch, batch_idx, dataloader_idx = next(combined_iterator)
            except StopIteration:
                break
            except RuntimeError as exc:
                if "DataLoader timed out" in str(exc):
                    skip_local = 1
                    timeout_encountered = True
                    timeout_message = str(exc)
                    batch = None
                    batch_idx = None
                    dataloader_idx = None
                else:
                    raise

            if batch is None:
                skip_local = 1

            skip_tensor = torch.tensor(
                skip_local, device=accelerator.device, dtype=torch.int32
            )
            skip_sum = accelerator.reduce(skip_tensor, reduction="sum")
            if skip_sum.item() > 0:
                timeout_tensor = torch.tensor(
                    1 if timeout_encountered else 0,
                    device=accelerator.device,
                    dtype=torch.int32,
                )
                timeout_sum = accelerator.reduce(timeout_tensor, reduction="sum")
                if accelerator.is_main_process:
                    if timeout_sum.item() > 0:
                        logger.warning(
                            "Skipping global step %s due to DataLoader timeout: %s",
                            global_step,
                            timeout_message or "timeout on non-main rank",
                        )
                    else:
                        logger.warning(
                            "Skipping global step %s due to empty batch from CombinedLoader.",
                            global_step,
                        )
                batch_time_m.reset()
                data_time_m.reset()
                end = time.time()
                continue

            accum_steps = max(1, config.training.gradient_accumulation_steps)
            micro_idx = micro_step % accum_steps
            # Alternate branches per micro-step so two branches compose one optimizer step.
            use_stage1 = micro_idx % 2 == 0
            use_stage2 = not use_stage1

            use_t2i = use_stage1 and train_dataloader_t2i is not None
            use_i2i = use_stage1 and train_dataloader_i2i is not None
            use_mmu = use_stage1 and train_dataloader_mmu is not None
            use_lm = use_stage2 and train_dataloader_lm is not None
            use_v2 = use_stage2 and train_dataloader_v2t is not None
            use_speech = use_stage2 and (
                train_dataloader_s2t is not None or train_dataloader_t2s is not None
            )

            v2t_batch = None
            t2i_batch = None
            i2i_batch = None

            if use_v2:
                v2t_batch = _next_from_v2t()

            batch["v2t_flow"] = v2t_batch

            # Initialize speech flows with empty placeholders; they will be populated if selected.
            batch["s2t_flow"] = _empty_audio_batch()
            batch["t2s_flow"] = _empty_audio_batch()

            speech_choices: list[str] = []
            if use_speech and train_dataloader_s2t is not None:
                speech_choices.append("s2t")
            if use_speech and train_dataloader_t2s is not None:
                speech_choices.append("t2s")

            selected_speech_branch: Optional[str] = None
            if speech_choices:
                choice_idx = global_step % len(speech_choices)
                selected_speech_branch = speech_choices[choice_idx]
                if selected_speech_branch == "s2t":
                    speech_batch = _next_from_s2t()
                    if speech_batch is None:
                        skip_local = 1
                    else:
                        batch["s2t_flow"] = speech_batch
                else:
                    speech_batch = _next_from_t2s()
                    if speech_batch is None:
                        skip_local = 1
                    else:
                        batch["t2s_flow"] = speech_batch

            x2i_choices: list[str] = []
            if use_t2i:
                x2i_choices.append("t2i")
            if use_i2i:
                x2i_choices.append("i2i")

            if use_t2i:
                t2i_batch = _next_from_t2i()
            if use_i2i:
                i2i_batch = _next_from_i2i()

            # Synchronize skip decision across all ranks to avoid collective mismatches
            required_flows = ["t2s_flow", "s2t_flow"]
            if use_lm:
                required_flows.append("lm_flow")
            if use_mmu:
                required_flows.append("mmu_flow")

            local_skip = 0
            skip_reasons = []
            for key in required_flows:
                if batch.get(key) is None:
                    local_skip = 1
                    skip_reasons.append(f"{key} missing")
                    break
            if use_v2 and v2t_batch is None:
                local_skip = 1
                skip_reasons.append("v2t batch missing")
            if use_t2i:
                if t2i_batch is None:
                    local_skip = 1
                    skip_reasons.append("t2i batch missing")
                else:
                    t2i_images = t2i_batch["t2i"].get("images")
                    if (
                        not isinstance(t2i_images, torch.Tensor)
                        or t2i_images.shape[0] == 0
                    ):
                        local_skip = 1
                        skip_reasons.append("t2i images empty")
            if use_i2i:
                if i2i_batch is None:
                    local_skip = 1
                    skip_reasons.append("i2i batch missing")
                else:
                    i2i_sources = i2i_batch["i2i"].get("source_images")
                    i2i_targets = i2i_batch["i2i"].get("target_images")
                    if (
                        not isinstance(i2i_sources, torch.Tensor)
                        or not isinstance(i2i_targets, torch.Tensor)
                        or i2i_sources.shape[0] == 0
                        or i2i_targets.shape[0] == 0
                    ):
                        local_skip = 1
                        skip_reasons.append("i2i source/target empty")
            try:
                skip_tensor = torch.tensor(
                    local_skip, device=accelerator.device, dtype=torch.int32
                )
                skip_sum = accelerator.reduce(skip_tensor, reduction="sum")
                should_skip = skip_sum.item() > 0
            except Exception:
                # Fallback if reduce isn't available for any reason
                should_skip = local_skip == 1

            if should_skip:
                if accelerator.is_main_process and local_skip:
                    reason_msg = (
                        "; ".join(skip_reasons)
                        if skip_reasons
                        else "required multimodal batch missing"
                    )
                    logger.warning(
                        f"Skipping step {global_step} ({reason_msg}) [synced]"
                    )
                continue

            device = accelerator.device
            # Text-to-image samples
            batch_size_t2i = 0
            mask_prob = torch.tensor(0.0, device=device)
            t2i_masks = torch.empty((0, 1), dtype=torch.long, device=device)
            input_ids_t2i = torch.empty((0, 1), dtype=torch.long, device=device)
            labels_t2i = torch.empty((0, 1), dtype=torch.long, device=device)
            batch_size_i2i = 0
            mask_prob_i2i = torch.tensor(0.0, device=device)
            input_ids_i2i = torch.empty((0, 1), dtype=torch.long, device=device)
            labels_i2i = torch.empty((0, 1), dtype=torch.long, device=device)
            attention_masks_i2i = torch.empty((0, 1), dtype=torch.long, device=device)

            if use_t2i and t2i_batch is not None:
                t2i_texts = t2i_batch["t2i"].get("texts", [])
                t2i_images_tensor = t2i_batch["t2i"].get("images")
                if (
                    isinstance(t2i_images_tensor, torch.Tensor)
                    and t2i_images_tensor.shape[0] > 0
                ):
                    t2i_images_tensor = t2i_images_tensor.to(device, non_blocking=True)
                    batch_size_t2i = t2i_images_tensor.shape[0]
                    (
                        input_ids_t2i,
                        labels_t2i,
                        mask_prob,
                        _,
                        t2i_masks,
                    ) = prepare_inputs_and_labels(
                        t2i_images_tensor, t2i_texts, config.training.min_masking_rate
                    )
                    input_ids_t2i = input_ids_t2i.to(device, non_blocking=True)
                    labels_t2i = labels_t2i.to(device, non_blocking=True)
                    t2i_masks = t2i_masks.to(device, non_blocking=True)
                    if mask_prob.device != device:
                        mask_prob = mask_prob.to(device)

            if use_i2i and i2i_batch is not None:
                i2i_prompts = i2i_batch["i2i"].get("prompts", [])
                i2i_source_tensor = i2i_batch["i2i"].get("source_images")
                i2i_target_tensor = i2i_batch["i2i"].get("target_images")
                i2i_target_texts = i2i_batch["i2i"].get("target_texts", [])
                if (
                    isinstance(i2i_source_tensor, torch.Tensor)
                    and isinstance(i2i_target_tensor, torch.Tensor)
                    and i2i_source_tensor.shape[0] > 0
                    and i2i_target_tensor.shape[0] > 0
                ):
                    i2i_source_tensor = i2i_source_tensor.to(device, non_blocking=True)
                    i2i_target_tensor = i2i_target_tensor.to(device, non_blocking=True)
                    batch_size_i2i = i2i_source_tensor.shape[0]
                    (
                        input_ids_i2i,
                        labels_i2i,
                        mask_prob_i2i,
                        attention_masks_i2i,
                    ) = prepare_inputs_and_labels_for_i2i(
                        i2i_source_tensor,
                        i2i_target_tensor,
                        i2i_prompts,
                        is_train=True,
                        target_texts=i2i_target_texts if i2i_target_texts else None,
                    )
                    input_ids_i2i = input_ids_i2i.to(device, non_blocking=True)
                    labels_i2i = labels_i2i.to(device, non_blocking=True)
                    attention_masks_i2i = attention_masks_i2i.to(
                        device, non_blocking=True
                    )
                    if mask_prob_i2i.device != device:
                        mask_prob_i2i = mask_prob_i2i.to(device)

            # Language modeling samples
            batch_size_lm = 0
            input_ids_lm = torch.empty((0, 1), dtype=torch.long, device=device)
            labels_lm = torch.empty((0, 1), dtype=torch.long, device=device)
            p_mask_lm = torch.empty((0, 1), dtype=torch.float32, device=device)
            local_masking_rate_lm = torch.tensor(0.0, device=device)
            if use_lm:
                lm_batch = batch.get("lm_flow")
                if lm_batch is not None:
                    texts_lm = lm_batch["input_ids"]
                    batch_size_lm = len(texts_lm)
                    # LM inputs truncated to max_seq_lm_input, outputs padded to max_seq_text
                    input_ids_lm, labels_lm, p_mask_lm = (
                        prepare_inputs_and_labels_for_text(texts_lm, max_seq_text)
                    )
                    input_ids_lm = input_ids_lm.to(device, non_blocking=True)
                    labels_lm = labels_lm.to(device, non_blocking=True)
                    p_mask_lm = p_mask_lm.to(device, non_blocking=True)
                    local_masking_rate_lm = (input_ids_lm == mask_id).float().mean()

            if isinstance(v2t_batch, dict):
                video_tensor_text_raw = v2t_batch.get("video")
                texts_vid = v2t_batch.get("captions", [])
            else:
                video_tensor_text_raw = None
                texts_vid = []

            batch_size_v2t = (
                video_tensor_text_raw.shape[0]
                if isinstance(video_tensor_text_raw, torch.Tensor)
                else 0
            )

            # Keep video tensors on CPU and move per-sample later to avoid spiking GPU memory.
            video_tensor_text = (
                video_tensor_text_raw
                if isinstance(video_tensor_text_raw, torch.Tensor)
                else torch.empty((0, 1, 1, 1, 1))
            )

            mmu_batch = batch.get("mmu_flow")
            batch_size_mmu = 0
            image_tensor_list = []
            texts_image = []
            if use_mmu and mmu_batch is not None:
                image_tensor_list = mmu_batch.get("images", [])
                texts_image = mmu_batch.get("text", [])
                batch_size_mmu = len(image_tensor_list)

            s2t_flow = batch.get("s2t_flow", {})
            t2s_flow = batch.get("t2s_flow", {})
            audio_paths_s2t_raw, texts_s2t_raw = s2t_flow.get(
                "audio_path", []
            ), s2t_flow.get("text", [])
            audio_paths_t2s_raw, texts_t2s_raw = t2s_flow.get(
                "audio_path", []
            ), t2s_flow.get("text", [])
            audio_tokens_s2t_raw = s2t_flow.get("audio_tokens", [])
            audio_tokens_t2s_raw = t2s_flow.get("audio_tokens", [])

            audio_paths_s2t, audio_tokens_s2t, texts_s2t, skipped_s2t = (
                _prepare_audio_flow(
                    audio_paths_s2t_raw,
                    audio_tokens_s2t_raw,
                    texts_s2t_raw,
                )
            )
            audio_paths_t2s, audio_tokens_t2s, texts_t2s, skipped_t2s = (
                _prepare_audio_flow(
                    audio_paths_t2s_raw,
                    audio_tokens_t2s_raw,
                    texts_t2s_raw,
                )
            )
            batch_size_s2t = len(audio_paths_s2t)
            batch_size_t2s_text = len(audio_paths_t2s)
            if require_cached_audio_tokens and accelerator.is_main_process:
                skipped_total = skipped_s2t + skipped_t2s
                if skipped_total and (global_step % 50 == 0):
                    logger.info(
                        "Skipped %d speech samples lacking cached tokens (s2t=%d, t2s=%d).",
                        skipped_total,
                        skipped_s2t,
                        skipped_t2s,
                    )

            if batch_size_s2t > 0 and batch_size_t2s_text > 0:
                drop_t2s = (global_step % 2) == 0

                if drop_t2s:
                    audio_paths_t2s = []
                    texts_t2s = []
                    audio_tokens_t2s = []
                    batch_size_t2s_text = 0
                else:
                    audio_paths_s2t = []
                    texts_s2t = []
                    audio_tokens_s2t = []
                    batch_size_s2t = 0

            batch_size_s2t = len(audio_paths_s2t)
            batch_size_t2s_text = len(audio_paths_t2s)

            if use_t2i and use_i2i:
                active_x2i_branch = "both"
            elif use_t2i:
                active_x2i_branch = "t2i"
            elif use_i2i:
                active_x2i_branch = "i2i"
            else:
                active_x2i_branch = "none"
            logger.info(
                f"x2i_branch: {active_x2i_branch}, batch_size_t2i: {batch_size_t2i}, batch_size_i2i: {batch_size_i2i}, batch_size_lm: {batch_size_lm}, "
                f"batch_size_v2t: {batch_size_v2t}, batch_size_t2s: {batch_size_t2s_text}, "
                f"batch_size_s2t: {batch_size_s2t}, batch_size_mmu: {batch_size_mmu}"
            )
            offset = speech_vocab_start

            data_time_m.update(time.time() - end)

            failure_messages = []
            step_failed = False

            input_ids_vid = torch.empty((0, 1), dtype=torch.long, device=device)
            labels_vid = torch.empty((0, 1), dtype=torch.long, device=device)
            p_mask_vid = torch.empty((0, 1), dtype=torch.float32, device=device)
            answer_lengths_vid = torch.empty((0, 1), dtype=torch.long, device=device)

            if batch_size_v2t > 0:
                video_token_list = []
                for vid_idx, video in enumerate(video_tensor_text):
                    video = video.to(device, non_blocking=True)
                    tokens, err = safe_video_get_code(video, vid_idx)
                    if err is not None:
                        failure_messages.append(err)
                        step_failed = True
                        break
                    tokens = tokens + len(uni_prompting.text_tokenizer)
                    video_token_list.append(tokens.view(-1).cpu())

                if not step_failed and video_token_list:
                    video_tokens_text = torch.stack(video_token_list, dim=0)

                    texts_with_prompt: List[str]
                    is_vid_inst = False
                    if (
                        texts_vid
                        and isinstance(texts_vid[0], (list, tuple))
                        and isinstance(texts_vid[0][0], dict)
                    ):
                        is_vid_inst = True
                        vid_inst_prompt: List[str] = []
                        vid_inst_answer: List[str] = []
                        for conv in texts_vid:
                            human_msg = ""
                            assistant_msg = ""
                            for turn in conv:
                                role = turn.get("from")
                                value = turn.get("value", "")
                                if role == "human":
                                    human_msg = value.replace("<image>\n", "")
                                elif role == "gpt":
                                    assistant_msg = value
                            vid_inst_prompt.append(human_msg)
                            vid_inst_answer.append(assistant_msg)
                        texts_with_prompt = [
                            "<|start_header_id|>user<|end_header_id|>\n"
                            f"{vid_inst_prompt[i]}<|eot_id|>"
                            "<|start_header_id|>assistant<|end_header_id|>\n"
                            f"{vid_inst_answer[i]}"
                            for i in range(len(vid_inst_answer))
                        ]
                    else:
                        prompt_v2t_selected = random.choice(V2T_INSTRUCTION)
                        texts_with_prompt = [
                            "<|start_header_id|>user<|end_header_id|>\n"
                            f"{prompt_v2t_selected}<|eot_id|>"
                            "<|start_header_id|>assistant<|end_header_id|>\n"
                            f"{text if isinstance(text, str) else str(text)}"
                            for text in texts_vid
                        ]

                    input_ids_vid_tmp, prompt_masks_vid, labels_vid_tmp = uni_prompting(
                        (video_tokens_text, texts_with_prompt),
                        "v2t",
                        config={"max_text_len_override": max_seq_mmu},
                    )
                    (
                        input_ids_vid_tmp,
                        labels_vid_tmp,
                        p_mask_vid,
                        answer_lengths_vid,
                    ) = prepare_inputs_and_labels_for_mmu(
                        input_ids_vid_tmp, prompt_masks_vid, labels_vid_tmp
                    )
                    input_ids_vid = input_ids_vid_tmp.to(device, non_blocking=True)
                    labels_vid = labels_vid_tmp.to(device, non_blocking=True)
                    p_mask_vid = p_mask_vid.to(device, non_blocking=True)
                    answer_lengths_vid = answer_lengths_vid.to(
                        device, non_blocking=True
                    )
                else:
                    batch_size_v2t = 0

            # Build formatted sequences for speech understanding.
            if not step_failed and batch_size_s2t > 0:
                prompt_s2t = [
                    "<|start_header_id|>user<|end_header_id|>\n"
                    + prompt
                    + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
                    for prompt in S2T_INSTRUCTION
                ]

                all_audio_tokens = []
                if not audio_tokens_s2t:
                    audio_tokens_s2t = [None] * len(audio_paths_s2t)
                elif len(audio_tokens_s2t) < len(audio_paths_s2t):
                    audio_tokens_s2t = list(audio_tokens_s2t) + [None] * (
                        len(audio_paths_s2t) - len(audio_tokens_s2t)
                    )

                for path, cached_tokens in zip(audio_paths_s2t, audio_tokens_s2t):
                    source = cached_tokens if cached_tokens is not None else path
                    tokens, err = safe_audio_encode(source, "s2t")
                    if err is not None:
                        failure_messages.append(err)
                        step_failed = True
                        break
                    tokens = tokens.to(accelerator.device, non_blocking=True)
                    tokens_with_offset = tokens + offset
                    all_audio_tokens.append(tokens_with_offset)

                if not step_failed:
                    prompt = random.choice(prompt_s2t)
                    texts_with_prompt = [f"{prompt}{text}" for text in texts_s2t]

                    input_ids_s2t, prompt_masks_s2t, labels_s2t = uni_prompting(
                        (all_audio_tokens, texts_with_prompt),
                        "s2t",
                        config={"max_text_len_override": max_seq_s2t},
                    )
                    # Preserve trailing EOS tokens in s2t targets for explicit prediction.
                    input_ids_s2t, labels_s2t, p_mask_s2t, answer_lengths_s2t = (
                        prepare_inputs_and_labels_for_s2t(
                            input_ids_s2t, prompt_masks_s2t, labels_s2t
                        )
                    )
            else:
                input_ids_s2t = torch.empty(
                    (0, 1), dtype=torch.long, device=accelerator.device
                )
                labels_s2t = torch.empty(
                    (0, 1), dtype=torch.long, device=accelerator.device
                )
                p_mask_s2t = torch.empty(
                    (0, 1), dtype=torch.float32, device=accelerator.device
                )
                answer_lengths_s2t = torch.empty(
                    (0, 1), dtype=torch.long, device=accelerator.device
                )

            # Build formatted sequences for speech generation.
            if not step_failed and batch_size_t2s_text > 0:
                prompt_t2s = [prompt for prompt in T2S_INSTRUCTION]

                all_audio_tokens = []
                if not audio_tokens_t2s:
                    audio_tokens_t2s = [None] * len(audio_paths_t2s)
                elif len(audio_tokens_t2s) < len(audio_paths_t2s):
                    audio_tokens_t2s = list(audio_tokens_t2s) + [None] * (
                        len(audio_paths_t2s) - len(audio_tokens_t2s)
                    )

                for path, cached_tokens in zip(audio_paths_t2s, audio_tokens_t2s):
                    source = cached_tokens if cached_tokens is not None else path
                    tokens, err = safe_audio_encode(source, "t2s")
                    if err is not None:
                        failure_messages.append(err)
                        step_failed = True
                        break
                    tokens = tokens.to(accelerator.device, non_blocking=True)
                    tokens_with_offset = tokens + offset
                    all_audio_tokens.append(tokens_with_offset)

                if not step_failed:
                    # Use chat-style instruction formatting for T2S training.
                    prompt = random.choice(prompt_t2s)
                    texts_with_prompt = [
                        f"<|start_header_id|>user<|end_header_id|>\n{prompt}\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
                        for text in texts_t2s
                    ]

                    input_ids_t2s, prompt_masks_t2s, labels_t2s = uni_prompting(
                        (texts_with_prompt, all_audio_tokens),
                        "t2s",
                        config={"max_text_len_override": max_seq_t2s},
                    )
                    input_ids_t2s, labels_t2s, p_mask_t2s, answer_lengths_t2s = (
                        prepare_inputs_and_labels_for_t2s(
                            input_ids_t2s, prompt_masks_t2s, labels_t2s
                        )
                    )
            else:
                input_ids_t2s = torch.empty(
                    (0, 1), dtype=torch.long, device=accelerator.device
                )
                labels_t2s = torch.empty(
                    (0, 1), dtype=torch.long, device=accelerator.device
                )
                p_mask_t2s = torch.empty(
                    (0, 1), dtype=torch.float32, device=accelerator.device
                )
                answer_lengths_t2s = torch.empty(
                    (0, 1), dtype=torch.long, device=accelerator.device
                )

            input_ids_mmu = None
            labels_mmu = None
            p_mask_mmu = None
            answer_lengths_mmu = None

            if not step_failed and batch_size_mmu > 0:
                batch_image_ids_list = []
                batch_text_ids = []

                for b_idx, image_list in enumerate(image_tensor_list):
                    per_img_ids = []
                    for j, img in enumerate(image_list):
                        tok, err = safe_image_get_code(
                            img.to(accelerator.device, non_blocking=True),
                            sample_index=j,
                        )
                        if err is not None:
                            failure_messages.append(err)
                            step_failed = True
                            break

                        tok = (
                            tok.to(accelerator.device, non_blocking=True)
                            .view(-1)
                            .long()
                        )
                        tok = tok + len(uni_prompting.text_tokenizer)
                        per_img_ids.append(tok)

                    if step_failed:
                        break

                    batch_image_ids_list.append(per_img_ids)
                    text_ids = uni_prompting.text_tokenizer.encode(
                        texts_image[b_idx], add_special_tokens=False
                    )
                    batch_text_ids.append(text_ids)

                if not step_failed:
                    input_ids_mmu, prompt_masks_mmu, labels_mmu = (
                        uni_prompting.mmu_mult_prompt(
                            batch_image_ids_list=batch_image_ids_list,
                            batch_text_ids=batch_text_ids,
                            max_text_len_override=max_seq_mmu,
                        )
                    )

                    (input_ids_mmu, labels_mmu, p_mask_mmu, answer_lengths_mmu) = (
                        prepare_inputs_and_labels_for_mmu(
                            input_ids_mmu, prompt_masks_mmu, labels_mmu
                        )
                    )

                    input_ids_mmu = input_ids_mmu.to(
                        accelerator.device, non_blocking=True
                    )
                    labels_mmu = labels_mmu.to(accelerator.device, non_blocking=True)
                    p_mask_mmu = p_mask_mmu.to(accelerator.device, non_blocking=True)
                    answer_lengths_mmu = answer_lengths_mmu.to(
                        accelerator.device, non_blocking=True
                    )

            if batch_size_mmu == 0 or input_ids_mmu is None:
                input_ids_mmu = torch.empty(
                    (0, 1), dtype=torch.long, device=accelerator.device
                )
                labels_mmu = torch.empty(
                    (0, 1), dtype=torch.long, device=accelerator.device
                )
                p_mask_mmu = torch.empty(
                    (0, 1), dtype=torch.float32, device=accelerator.device
                )
                answer_lengths_mmu = torch.empty(
                    (0, 1), dtype=torch.long, device=accelerator.device
                )
            if not step_failed:
                total_batch_size_t2s = batch_size_t2s_text
            else:
                total_batch_size_t2s = batch_size_t2s_text

            failure_tensor = torch.tensor(
                1 if step_failed else 0, device=accelerator.device, dtype=torch.int32
            )
            failure_sum = accelerator.reduce(failure_tensor, reduction="sum")
            if failure_sum.item() > 0:
                if accelerator.is_main_process and failure_messages:
                    for msg in failure_messages:
                        logger.warning(
                            f"Skipping global step {global_step} due to preprocessing failure: {msg}"
                        )
                batch_time_m.reset()
                data_time_m.reset()
                end = time.time()
                continue

            # 1. Define padding values
            pad_token_id = uni_prompting.text_tokenizer.eos_token_id
            pad_id = uni_prompting.pad_id

            # 2. Find the maximum sequence length in the current batch
            seq_lengths = []
            if input_ids_t2i.shape[0] > 0:
                seq_lengths.append(input_ids_t2i.shape[1])
            if input_ids_i2i.shape[0] > 0:
                seq_lengths.append(input_ids_i2i.shape[1])
            if input_ids_lm.shape[0] > 0:
                seq_lengths.append(input_ids_lm.shape[1])
            seq_lengths.extend(
                [
                    input_ids_vid.shape[1],
                    input_ids_s2t.shape[1],
                    input_ids_t2s.shape[1],
                ]
            )
            if input_ids_mmu.shape[0] > 0:
                seq_lengths.append(input_ids_mmu.shape[1])

            if accelerator.is_main_process:
                batch_len_msg = []
                for name, tensor in [
                    ("t2i", input_ids_t2i),
                    ("i2i", input_ids_i2i),
                    ("lm", input_ids_lm),
                    ("mmu", input_ids_mmu),
                    ("vid", input_ids_vid),
                    ("s2t", input_ids_s2t),
                    ("t2s", input_ids_t2s),
                ]:
                    if tensor is None:
                        continue
                    bs = tensor.shape[0]
                    seq_len = tensor.shape[1] if tensor.dim() > 1 else 0
                    batch_len_msg.append(f"{name}:bs={bs},len={seq_len}")
                logger.info(
                    "Per-modality batch lengths (pre-pad): %s", "; ".join(batch_len_msg)
                )
            max_len = max(seq_lengths)

            # 3. Pad all tensors to the max_len
            input_ids_t2i = pad_tensor(input_ids_t2i, max_len, pad_token_id)
            labels_t2i = pad_tensor(labels_t2i, max_len, -100)
            if t2i_masks.shape[0] > 0:
                t2i_masks = pad_tensor(t2i_masks.long(), max_len, 0)
            else:
                t2i_masks = torch.empty((0, max_len), dtype=torch.long, device=device)

            input_ids_i2i = pad_tensor(input_ids_i2i, max_len, pad_token_id)
            labels_i2i = pad_tensor(labels_i2i, max_len, -100)
            if attention_masks_i2i.shape[0] > 0:
                attention_masks_i2i = pad_tensor(attention_masks_i2i.long(), max_len, 0)
            else:
                attention_masks_i2i = torch.empty(
                    (0, max_len), dtype=torch.long, device=device
                )

            input_ids_lm = pad_tensor(input_ids_lm, max_len, pad_token_id)
            labels_lm = pad_tensor(labels_lm, max_len, -100)
            p_mask_lm = pad_tensor(p_mask_lm, max_len, 1.0)

            input_ids_vid = pad_tensor(input_ids_vid, max_len, pad_token_id)
            input_ids_s2t = pad_tensor(input_ids_s2t, max_len, pad_token_id)
            input_ids_t2s = pad_tensor(input_ids_t2s, max_len, pad_token_id)
            input_ids_mmu = pad_tensor(input_ids_mmu, max_len, pad_token_id)
            labels_vid = pad_tensor(labels_vid, max_len, -100)
            labels_s2t = pad_tensor(labels_s2t, max_len, -100)
            labels_t2s = pad_tensor(labels_t2s, max_len, -100)
            labels_mmu = pad_tensor(labels_mmu, max_len, -100)
            p_mask_vid = pad_tensor(p_mask_vid, max_len, 1.0)
            p_mask_s2t = pad_tensor(p_mask_s2t, max_len, 1.0)
            p_mask_t2s = pad_tensor(p_mask_t2s, max_len, 1.0)
            p_mask_mmu = pad_tensor(p_mask_mmu, max_len, 1.0)
            answer_lengths_vid = pad_answer_lengths(answer_lengths_vid, max_len)
            answer_lengths_s2t = pad_answer_lengths(answer_lengths_s2t, max_len)
            answer_lengths_t2s = pad_answer_lengths(answer_lengths_t2s, max_len)
            answer_lengths_mmu = pad_answer_lengths(answer_lengths_mmu, max_len)

            def _pad_attention_mask(tensor):
                if tensor.shape[0] == 0:
                    return torch.empty((0, max_len), dtype=torch.long, device=device)
                return ((tensor != pad_id) & (tensor != pad_token_id)).long()

            attention_mask_t2i = _pad_attention_mask(input_ids_t2i)
            attention_mask_i2i = _pad_attention_mask(input_ids_i2i)
            attention_mask_lm = _pad_attention_mask(input_ids_lm)
            attention_mask_mmu = _pad_attention_mask(input_ids_mmu)
            attention_mask_vid = _pad_attention_mask(input_ids_vid)
            attention_mask_s2t = _pad_attention_mask(input_ids_s2t)
            attention_mask_t2s = _pad_attention_mask(input_ids_t2s)

            input_ids = torch.cat(
                (
                    input_ids_t2i,
                    input_ids_i2i,
                    input_ids_lm,
                    input_ids_mmu,
                    input_ids_vid,
                    input_ids_s2t,
                    input_ids_t2s,
                ),
                dim=0,
            )
            attention_mask = torch.cat(
                (
                    attention_mask_t2i,
                    attention_mask_i2i,
                    attention_mask_lm,
                    attention_mask_mmu,
                    attention_mask_vid,
                    attention_mask_s2t,
                    attention_mask_t2s,
                ),
                dim=0,
            )
            labels = torch.cat(
                (
                    labels_t2i,
                    labels_i2i,
                    labels_lm,
                    labels_mmu,
                    labels_vid,
                    labels_s2t,
                    labels_t2s,
                ),
                dim=0,
            )

            # w/o texts and images
            if batch_size_lm == 0:
                p_mask_lm = torch.empty(
                    (0, max_len), dtype=torch.float32, device=device
                )
            if batch_size_t2i == 0 and t2i_masks.shape[0] == 0:
                t2i_masks = torch.empty((0, max_len), dtype=torch.long, device=device)

            if global_step == 0 and epoch == 0:
                logger.info("Input ids: {}".format(input_ids))
                logger.info("Labels: {}".format(labels))

            logger.debug("Input ids shape: %s", input_ids.shape)
            (
                logits,
                loss_t2i,
                loss_i2i,
                _loss_ti2ti,
                loss_lm,
                loss_mmu,
                loss_vid,
                _loss_v2s,
                loss_s2t,
                _loss_s2s,
                loss_t2s,
            ) = accelerator.unwrap_model(model).forward_process(
                input_ids=input_ids,
                labels=labels,
                batch_size_t2i=batch_size_t2i,
                batch_size_i2i=batch_size_i2i,
                batch_size_lm=batch_size_lm,
                batch_size_mmu=batch_size_mmu,
                batch_size_v2t=batch_size_v2t,
                batch_size_v2s=0,
                batch_size_s2t=batch_size_s2t,
                batch_size_s2s=0,
                batch_size_t2s=total_batch_size_t2s,
                max_seq_length=config.dataset.preprocessing.max_seq_length,
                attention_mask=attention_mask,
                attention_masks_i2i=attention_masks_i2i,
                p_mask_lm=p_mask_lm,
                p_mask_mmu=p_mask_mmu,
                p_mask_vid=p_mask_vid,
                p_mask_v2s=torch.empty(
                    (0, max_len), dtype=torch.float32, device=device
                ),
                p_mask_s2t=p_mask_s2t,
                p_mask_s2s=torch.empty(
                    (0, max_len), dtype=torch.float32, device=device
                ),
                p_mask_t2s=p_mask_t2s,
                answer_lengths_mmu=answer_lengths_mmu,
                answer_lengths_vid=answer_lengths_vid,
                answer_lengths_v2s=torch.empty(
                    (0, max_len), dtype=torch.long, device=device
                ),
                answer_lengths_s2t=answer_lengths_s2t,
                answer_lengths_s2s=torch.empty(
                    (0, max_len), dtype=torch.long, device=device
                ),
                answer_lengths_t2s=answer_lengths_t2s,
                t2i_masks=t2i_masks,
                t2s_vocab_start=speech_vocab_start,
                t2s_codebook_size=audio_codebook_size,
                t2s_special_token_ids=t2s_special_token_ids,
                text_vocab_size_override=len(uni_prompting.text_tokenizer),
            )

            if batch_size_t2i == 0:
                loss_t2i = loss_t2i.new_zeros(())
            if batch_size_i2i == 0:
                loss_i2i = loss_i2i.new_zeros(())

            # Gather the losses across all processes for logging (use reduce to avoid shape mismatches)
            avg_loss_t2i = accelerator.reduce(loss_t2i, reduction="mean")
            avg_loss_i2i = accelerator.reduce(loss_i2i, reduction="mean")
            avg_loss_lm = accelerator.reduce(loss_lm, reduction="mean")
            avg_loss_mmu = accelerator.reduce(loss_mmu, reduction="mean")
            avg_loss_vid = accelerator.reduce(loss_vid, reduction="mean")
            avg_loss_s2t = accelerator.reduce(loss_s2t, reduction="mean")
            if not torch.isfinite(loss_t2s):
                if labels_t2s.numel() > 0:
                    speech_vocab_end = speech_vocab_start + audio_codebook_size
                    valid_mask = labels_t2s != -100
                    if valid_mask.any():
                        labels_valid = labels_t2s[valid_mask]
                        below_count = (labels_valid < speech_vocab_start).sum().item()
                        above_count = (labels_valid >= speech_vocab_end).sum().item()
                        labels_min = labels_valid.min().item()
                        labels_max = labels_valid.max().item()
                    else:
                        below_count = above_count = 0
                        labels_min = labels_max = -100
                    p_mask_min = (
                        p_mask_t2s.min().item()
                        if p_mask_t2s.numel() > 0
                        else float("nan")
                    )
                    ans_len_min = (
                        answer_lengths_t2s.min().item()
                        if answer_lengths_t2s.numel() > 0
                        else float("nan")
                    )
                    accelerator.print(
                        "[t2s NaN debug] "
                        f"rank={accelerator.process_index} step={global_step} "
                        f"slice=({speech_vocab_start}, {speech_vocab_end}) "
                        f"labels_min={labels_min} labels_max={labels_max} "
                        f"below_slice={below_count} above_slice={above_count} "
                        f"p_mask_min={p_mask_min} answer_len_min={ans_len_min}"
                    )
                accelerator.print(
                    f"[rank {accelerator.process_index}] t2s loss became NaN/Inf at global step {global_step} "
                    f"(local value: {loss_t2s.item()})"
                )
                logger.warning(
                    "[rank %s] t2s loss became NaN/Inf at global step %s (local value: %s)",
                    accelerator.process_index,
                    global_step,
                    loss_t2s.item(),
                )
            avg_loss_t2s = accelerator.reduce(loss_t2s, reduction="mean")
            if not torch.isfinite(avg_loss_t2s):
                accelerator.print(
                    f"[rank {accelerator.process_index}] reduced t2s loss NaN/Inf at global step {global_step} "
                    f"(value after all-reduce: {avg_loss_t2s.item()})"
                )
                if accelerator.is_main_process:
                    logger.warning(
                        "Reduced t2s loss became NaN/Inf at global step %s (value after all-reduce: %s)",
                        global_step,
                        avg_loss_t2s.item(),
                    )

            mmu_coeff = getattr(config.training, "mmu_coeff", 0.0)
            i2i_coeff = getattr(config.training, "i2i_coeff", config.training.t2i_coeff)
            loss = (
                config.training.t2i_coeff * loss_t2i
                + i2i_coeff * loss_i2i
                + config.training.lm_coeff * loss_lm
                + mmu_coeff * loss_mmu
                + config.training.v2t_coeff * loss_vid
                + config.training.s2t_coeff * loss_s2t
                + config.training.t2s_coeff * loss_t2s
            )

            if batch_size_t2i > 0:
                local_masking_rate = mask_prob.float().mean()
            else:
                local_masking_rate = torch.tensor(0.0, device=accelerator.device)
            avg_masking_rate = accelerator.reduce(local_masking_rate, reduction="mean")

            if batch_size_i2i > 0:
                local_masking_rate_i2i = mask_prob_i2i.float().mean()
            else:
                local_masking_rate_i2i = torch.tensor(0.0, device=accelerator.device)
            avg_masking_rate_i2i = accelerator.reduce(
                local_masking_rate_i2i, reduction="mean"
            )

            avg_masking_rate_lm = accelerator.reduce(
                local_masking_rate_lm.to(accelerator.device), reduction="mean"
            )

            micro_wandb_step = global_step * accum_steps + micro_idx + 1
            if (
                accelerator.is_main_process
                and log_every is not None
                and micro_wandb_step % log_every == 0
            ):
                micro_logs = {"micro_stage": 1 if use_stage1 else 2}
                micro_losses = {
                    "micro_loss_t2i": avg_loss_t2i.item(),
                    "micro_loss_i2i": avg_loss_i2i.item(),
                    "micro_loss_lm": avg_loss_lm.item(),
                    "micro_loss_mmu": avg_loss_mmu.item(),
                    "micro_loss_vid": avg_loss_vid.item(),
                    "micro_loss_s2t": avg_loss_s2t.item(),
                    "micro_loss_t2s": avg_loss_t2s.item(),
                }
                for key, value in micro_losses.items():
                    if value != 0.0:
                        micro_logs[key] = value
                if len(micro_logs) > 1:
                    accelerator.log(micro_logs, step=micro_wandb_step)

            accum_models = (
                ()
                if accelerator.distributed_type == DistributedType.DEEPSPEED
                else (model,)
            )
            with accelerator.accumulate(*accum_models):
                accelerator.backward(loss)

                if (
                    config.training.max_grad_norm is not None
                    and accelerator.sync_gradients
                ):
                    accelerator.clip_grad_norm_(
                        model.parameters(), config.training.max_grad_norm
                    )

                if accelerator.sync_gradients:
                    optimizer.step()
                    lr_scheduler.step()

                    # log gradient norm before zeroing it
                    if (
                        (global_step + 1) % config.experiment.log_grad_norm_every == 0
                        and accelerator.is_main_process
                    ):
                        log_grad_norm(model, accelerator, global_step + 1)

                    optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                batch_time_m.update(time.time() - end)
                end = time.time()

                # Log metrics
                if log_every is not None and (global_step + 1) % log_every == 0:
                    samples_per_second_per_gpu = (
                        config.training.gradient_accumulation_steps
                        * total_batch_size_per_gpu
                        / batch_time_m.val
                    )
                    logs = {
                        "lr": lr_scheduler.get_last_lr()[0],
                        "avg_masking_rate": avg_masking_rate.item(),
                        "avg_masking_rate_i2i": avg_masking_rate_i2i.item(),
                        "avg_masking_rate_lm": avg_masking_rate_lm.item(),
                        "samples/sec/gpu": samples_per_second_per_gpu,
                        "data_time": data_time_m.val,
                        "batch_time": batch_time_m.val,
                    }

                    loss_entries = [
                        ("step_loss_t2i", avg_loss_t2i),
                        ("step_loss_i2i", avg_loss_i2i),
                        ("step_loss_lm", avg_loss_lm),
                        ("step_loss_mmu", avg_loss_mmu),
                        ("step_loss_vid", avg_loss_vid),
                        ("step_loss_s2t", avg_loss_s2t),
                        ("step_loss_t2s", avg_loss_t2s),
                    ]

                    loss_log_parts = []
                    for key, value in loss_entries:
                        loss_value = value.item()
                        if loss_value != 0.0:
                            logs[key] = loss_value
                            loss_log_parts.append(
                                f"{key.replace('step_', '').capitalize()}: {loss_value:0.4f}"
                            )

                    global_wandb_step = (global_step + 1) * accum_steps
                    accelerator.log(logs, step=global_wandb_step)

                    loss_str = " ".join(loss_log_parts)
                    logger.info(
                        "Step: %d %s Data (t): %.4f, %.2f/s/gpu Batch (t): %.4f LR: %.6f"
                        % (
                            global_step + 1,
                            loss_str,
                            data_time_m.val,
                            samples_per_second_per_gpu,
                            batch_time_m.val,
                            lr_scheduler.get_last_lr()[0],
                        )
                    )

                    # resetting batch / data time meters per log window
                    batch_time_m.reset()
                    data_time_m.reset()

                # Save model checkpoint
                if save_every is not None and (global_step + 1) % save_every == 0:
                    save_checkpoint(
                        model, config, accelerator, global_step + 1, uni_prompting
                    )

                # Run evaluation.
                if eval_every is not None and (global_step + 1) % eval_every == 0:
                    run_evaluation(
                        model=accelerator.unwrap_model(model),
                        vq_model_image=vq_model_image,
                        vq_model_audio=vq_model_audio,
                        uni_prompting=uni_prompting,
                        config=config,
                        accelerator=accelerator,
                        global_step=global_step + 1,
                    )
                    # The evaluation function sets model back to train mode internally.

                global_step += 1

            micro_step += 1

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
    # This function is not called in the main loop but kept for compatibility
    pass


@torch.no_grad()
def generate_images(*args, **kwargs):
    # This function is not called in the main loop but kept for compatibility
    pass


@torch.no_grad()
def understanding_images(*args, **kwargs):
    # This function is not called in the main loop but kept for compatibility
    pass


def save_checkpoint(model, config, accelerator, global_step, uni_prompting):
    output_dir = config.experiment.output_dir
    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    if accelerator.is_main_process and checkpoints_total_limit is not None:
        checkpoints = os.listdir(output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
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

    # retrieve the model on all processes for deepspeed stage 3 to work then save on one process (we are not using stage 3 yet)
    # XXX: could also make this conditional on deepspeed
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

        # save tokenizer
        uni_prompting.text_tokenizer.save_pretrained(save_path / "unwrapped_model")


def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)


if __name__ == "__main__":
    main()
