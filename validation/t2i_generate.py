# coding=utf-8
# Copyright 2025 MMaDA Team
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

import os
import json
import sys

from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import wandb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import MAGVITv2, get_mask_schedule, MMadaModelLM, DyninOmniModelLM
from training.config_resolver import (
    configure_hf_cache_env,
    resolve_model_local_files_only,
    resolve_model_pretrained_source,
    resolve_model_type_from_pretrained,
    resolve_tokenizer_source,
    resolve_vq_repo_source,
)
from training.prompting_utils import UniversalPrompting
from training.utils import get_config, flatten_omega_conf
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "true"

DEFAULT_T2I_PROMPTS = [
    "A cinematic portrait of an astronaut walking in a rainy neon city at night.",
    "A watercolor painting of a lighthouse on a stormy sea at sunset.",
    "A cozy reading room with plants, warm sunlight, and wooden furniture.",
]


def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")


def _resolve_model_cfg(config):
    if "mmada" in config.model:
        return config.model.mmada
    if "dynin_omni" in config.model:
        return config.model.dynin_omni
    raise ValueError("Config is missing model.mmada/model.dynin_omni block.")


def _resolve_vq_cfg(config):
    if "vq_model" in config.model:
        return config.model.vq_model
    if "vq_model_image" in config.model:
        return config.model.vq_model_image
    raise ValueError("Config is missing model.vq_model/model.vq_model_image block.")


def main():
    config = get_config()
    configure_hf_cache_env(config, project_root=PROJECT_ROOT)

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))

    use_wandb = bool(config.wandb.get("enable", True)) and rank == 0
    if use_wandb and os.environ.get("WANDB_DISABLED", "").lower() not in {"1", "true"}:
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb.init(
            project="demo",
            name=config.experiment.name + "_t2i",
            config=wandb_config,
        )
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    model_cfg = _resolve_model_cfg(config)
    vq_cfg = _resolve_vq_cfg(config)
    pretrained_path = config.get(
        "t2i_model_path", resolve_model_pretrained_source(config)
    )
    tok_src = resolve_tokenizer_source(config, default=pretrained_path)
    local_only = resolve_model_local_files_only(config, default=False)
    tokenizer = AutoTokenizer.from_pretrained(
        tok_src,
        padding_side="left",
        trust_remote_code=True,
        local_files_only=local_only,
    )

    uni_prompting = UniversalPrompting(
        tokenizer,
        max_text_len=config.dataset.preprocessing.max_seq_length,
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
        ),
        ignore_id=-100,
        cond_dropout_prob=config.training.cond_dropout_prob,
        use_reserved_token=True,
    )

    vq_model = get_vq_model_class(vq_cfg.type)
    vq_model = vq_model.from_pretrained(
        resolve_vq_repo_source(vq_cfg),
        local_files_only=local_only,
    ).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()

    model_type = resolve_model_type_from_pretrained(
        pretrained_path,
        local_files_only=local_only,
    )
    model_cls = DyninOmniModelLM if model_type == "dynin_omni" else MMadaModelLM
    load_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = model_cls.from_pretrained(
        pretrained_path,
        trust_remote_code=True,
        torch_dtype=load_dtype,
        local_files_only=local_only,
    ).to(device)
    model.eval()

    mask_token_id = model.config.mask_token_id
    if config.get("validation_prompts_file", None) is not None:
        config.dataset.params.validation_prompts_file = config.validation_prompts_file
    config.training.batch_size = int(
        config.get(
            "batch_size",
            config.training.get("batch_size", config.training.get("batch_size_t2i", 1)),
        )
    )
    if config.get("guidance_scale", None) is not None:
        config.training.guidance_scale = config.guidance_scale
    if config.get("generation_timesteps", None) is not None:
        config.training.generation_timesteps = config.generation_timesteps

    num_vq_tokens = int(model_cfg.num_vq_tokens)
    codebook_size = int(model_cfg.codebook_size)

    metadata_list = None
    if config.get("metadata_file", None):
        with open(config.metadata_file, "r", encoding="utf-8") as f:
            metadata_list = [json.loads(line) for line in f]
        validation_prompts = [m["prompt"] for m in metadata_list]
    else:
        prompts_path = getattr(config.dataset.params, "validation_prompts_file", None)
        if prompts_path is not None and os.path.exists(prompts_path):
            with open(prompts_path, "r", encoding="utf-8") as f:
                validation_prompts = []
                for line in f:
                    text = line.strip()
                    if not text:
                        continue
                    candidate = text
                    try:
                        parsed = json.loads(text)
                    except json.JSONDecodeError:
                        parsed = None
                    if isinstance(parsed, dict):
                        prompt_text = parsed.get("prompt")
                        if prompt_text is not None and str(prompt_text).strip():
                            candidate = str(prompt_text).strip()
                    elif isinstance(parsed, str) and parsed.strip():
                        candidate = parsed.strip()
                    validation_prompts.append(candidate)
            if not validation_prompts:
                validation_prompts = list(DEFAULT_T2I_PROMPTS)
                if rank == 0:
                    print(
                        f"[WARN] Prompt file is empty ({prompts_path}); using built-in prompts."
                    )
        else:
            validation_prompts = list(DEFAULT_T2I_PROMPTS)
            if rank == 0:
                print(
                    "[WARN] validation_prompts_file is missing; using built-in prompts."
                )

    outdir = config.get("outdir", config.experiment.output_dir)
    os.makedirs(outdir, exist_ok=True)
    n_samples = int(config.get("n_samples", 1))

    all_indices = list(range(len(validation_prompts)))
    shard_indices = all_indices[rank::world_size]
    for offset in tqdm(
        range(0, len(shard_indices), config.training.batch_size), disable=(rank != 0)
    ):
        batch_indices = shard_indices[offset : offset + config.training.batch_size]
        prompts = [validation_prompts[i] for i in batch_indices]
        prompts_rep = [p for p in prompts for _ in range(n_samples)]
        base_idx = batch_indices

        image_tokens = (
            torch.ones(
                (len(prompts_rep), num_vq_tokens),
                dtype=torch.long,
                device=device,
            )
            * mask_token_id
        )
        input_ids, attention_mask = uni_prompting(
            (prompts_rep, image_tokens), "t2i_gen"
        )
        if config.training.guidance_scale > 0:
            uncond_input_ids, uncond_attention_mask = uni_prompting(
                ([""] * len(prompts_rep), image_tokens), "t2i_gen"
            )
        else:
            uncond_input_ids = None
            uncond_attention_mask = None

        if config.get("mask_schedule", None) is not None:
            schedule = config.mask_schedule.schedule
            args = config.mask_schedule.get("params", {})
            mask_schedule = get_mask_schedule(schedule, **args)
        else:
            mask_schedule = get_mask_schedule(
                config.training.get("mask_schedule", "cosine")
            )
        with torch.no_grad():
            gen_token_ids = model.t2i_generate(
                input_ids=input_ids,
                uncond_input_ids=uncond_input_ids,
                attention_mask=attention_mask,
                uncond_attention_mask=uncond_attention_mask,
                guidance_scale=config.training.guidance_scale,
                temperature=config.training.get("generation_temperature", 1.0),
                timesteps=config.training.generation_timesteps,
                noise_schedule=mask_schedule,
                noise_type=config.training.get("noise_type", "mask"),
                seq_len=num_vq_tokens,
                uni_prompting=uni_prompting,
                config=config,
            )

        gen_token_ids = torch.clamp(gen_token_ids, max=codebook_size - 1, min=0)
        images = vq_model.decode_code(gen_token_ids)
        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
        images *= 255.0
        images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        pil_images = [Image.fromarray(image) for image in images]

        for i, prompt in enumerate(prompts):
            prompt_index = base_idx[i]
            outpath = os.path.join(outdir, f"{prompt_index:0>5}")
            os.makedirs(os.path.join(outpath, "samples"), exist_ok=True)
            if metadata_list is not None:
                with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
                    json.dump(metadata_list[prompt_index], fp)
            for j in range(n_samples):
                image_idx = i * n_samples + j
                pil_images[image_idx].save(
                    os.path.join(outpath, "samples", f"{j:05}.png")
                )

        if use_wandb and wandb.run is not None:
            wandb_images = [
                wandb.Image(image, caption=prompts_rep[i])
                for i, image in enumerate(pil_images)
            ]
            wandb.log(
                {"generated_images": wandb_images},
                step=batch_indices[0] if batch_indices else 0,
            )


if __name__ == "__main__":
    main()
