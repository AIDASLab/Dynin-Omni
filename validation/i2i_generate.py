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

import argparse
import json
import os
import sys

from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import MAGVITv2, MMadaModelLM, DyninOmniModelLM, get_mask_schedule
from training.config_resolver import (
    configure_hf_cache_env,
    resolve_model_local_files_only,
    resolve_model_pretrained_source,
    resolve_model_type_from_pretrained,
    resolve_tokenizer_source,
    resolve_vq_repo_source,
)
from training.prompting_utils import UniversalPrompting
from training.utils import image_transform

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def get_vq_model_class(model_type: str):
    if model_type == "magvitv2":
        return MAGVITv2
    raise ValueError(f"model_type {model_type} not supported.")


def load_config(path: str):
    cfg = OmegaConf.load(path)
    return cfg


def resolve_model_class(pretrained_path: str, local_files_only: bool = False):
    model_type = resolve_model_type_from_pretrained(
        pretrained_path,
        local_files_only=local_files_only,
    )
    return DyninOmniModelLM if model_type == "dynin_omni" else MMadaModelLM


def batch_iter(items, batch_size):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def main():
    parser = argparse.ArgumentParser(
        description="Dynin-Omni i2i generation for ImgEdit Basic benchmark"
    )
    parser.add_argument("--config", required=True, help="Path to yaml config")
    parser.add_argument(
        "--model_path",
        default=None,
        help="Override model.dynin_omni.repo_id or pretrained path",
    )
    parser.add_argument("--edit_json", required=True, help="Basic edit json file")
    parser.add_argument(
        "--origin_img_root", required=True, help="Root folder for original images"
    )
    parser.add_argument(
        "--outdir", required=True, help="Output folder for edited images"
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--source_resolution", type=int, default=None)
    parser.add_argument("--target_resolution", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--generation_timesteps", type=int, default=64)
    parser.add_argument("--generation_temperature", type=float, default=1.0)
    parser.add_argument(
        "--use_train_i2i_prompt",
        action="store_true",
        help="Use training i2i prompt template (<|i2i|> ...). This is the default.",
    )
    parser.add_argument(
        "--no-use_train_i2i_prompt",
        dest="use_train_i2i_prompt",
        action="store_false",
        help="Use i2i_gen prompt template (<|t2i|> ...).",
    )
    parser.set_defaults(use_train_i2i_prompt=True)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_hf_cache_env(cfg, project_root=PROJECT_ROOT)
    if args.guidance_scale is not None:
        cfg.training.guidance_scale = args.guidance_scale
    if args.generation_timesteps is not None:
        cfg.training.generation_timesteps = args.generation_timesteps

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    if "mmada" in cfg.model:
        model_cfg = cfg.model.mmada
    elif "dynin_omni" in cfg.model:
        model_cfg = cfg.model.dynin_omni
    else:
        raise ValueError("Config is missing model.mmada/model.dynin_omni block.")

    pretrained_path = args.model_path or resolve_model_pretrained_source(cfg)
    tok_src = resolve_tokenizer_source(cfg, default=pretrained_path)
    local_only = resolve_model_local_files_only(cfg, default=False)
    tokenizer = AutoTokenizer.from_pretrained(
        tok_src,
        padding_side="left",
        trust_remote_code=True,
        local_files_only=local_only,
    )
    uni_prompting = UniversalPrompting(
        tokenizer,
        max_text_len=cfg.dataset.preprocessing.max_seq_length,
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
        cond_dropout_prob=cfg.training.cond_dropout_prob,
        use_reserved_token=True,
    )
    if args.use_train_i2i_prompt:
        # Disable prompt dropout for deterministic inference while keeping training-style template.
        uni_prompting.cond_dropout_prob = 0.0

    if "vq_model" in cfg.model:
        vq_cfg = cfg.model.vq_model
    else:
        vq_cfg = cfg.model.vq_model_image
    vq_model_cls = get_vq_model_class(vq_cfg.type)
    vq_model = vq_model_cls.from_pretrained(
        resolve_vq_repo_source(vq_cfg),
        local_files_only=local_only,
    ).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()

    model_cls = resolve_model_class(pretrained_path, local_files_only=local_only)
    model = model_cls.from_pretrained(
        pretrained_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        local_files_only=local_only,
    ).to(device)
    model.eval()

    mask_token_id = model.config.mask_token_id
    base_resolution = (
        int(args.resolution)
        if args.resolution is not None
        else int(cfg.dataset.params.resolution)
    )
    src_resolution = (
        int(args.source_resolution)
        if args.source_resolution is not None
        else int(getattr(cfg.dataset.params, "i2i_source_resolution", base_resolution))
    )
    tgt_resolution = (
        int(args.target_resolution)
        if args.target_resolution is not None
        else int(getattr(cfg.dataset.params, "i2i_target_resolution", base_resolution))
    )
    num_vq_tokens = int((tgt_resolution // 16) ** 2)
    codebook_size = int(model_cfg.codebook_size)

    if cfg.get("mask_schedule", None) is not None:
        schedule = cfg.mask_schedule.schedule
        args_sched = cfg.mask_schedule.get("params", {})
        mask_schedule = get_mask_schedule(schedule, **args_sched)
    else:
        mask_schedule = get_mask_schedule(cfg.training.get("mask_schedule", "cosine"))

    with open(args.edit_json, "r", encoding="utf-8") as f:
        edit_infos = json.load(f)

    def sort_key(item):
        key = item[0]
        try:
            return int(key)
        except ValueError:
            return key

    items = sorted(edit_infos.items(), key=sort_key)
    items = items[rank::world_size]

    os.makedirs(args.outdir, exist_ok=True)

    for batch in tqdm(
        batch_iter(items, args.batch_size),
        total=(len(items) + args.batch_size - 1) // args.batch_size,
    ):
        batch_keys = []
        batch_prompts = []
        batch_images = []

        for key, item in batch:
            out_path = os.path.join(args.outdir, f"{key}.png")
            if args.skip_existing and os.path.isfile(out_path):
                continue
            origin_path = os.path.join(args.origin_img_root, item["id"])
            if not os.path.isfile(origin_path):
                print(f"[ERROR] Missing origin image: {origin_path}")
                continue
            try:
                img = Image.open(origin_path).convert("RGB")
            except Exception as exc:
                print(f"[ERROR] Failed to open {origin_path}: {exc}")
                continue

            batch_keys.append(key)
            batch_prompts.append(item["prompt"])
            batch_images.append(img)

        if not batch_keys:
            continue

        images = torch.stack(
            [image_transform(img, resolution=src_resolution) for img in batch_images],
            dim=0,
        ).to(device)
        input_image_tokens = vq_model.get_code(images) + len(
            uni_prompting.text_tokenizer
        )

        output_placeholder = (
            torch.ones(
                (len(batch_prompts), num_vq_tokens),
                dtype=torch.long,
                device=device,
            )
            * mask_token_id
        )

        if args.use_train_i2i_prompt:
            # Match training i2i template: <|i2i|> <|soi|> src <|eoi|> <bos> text <eos> <|soi|> [masked tgt] <|eoi|>
            labels_placeholder = torch.full(
                (len(batch_prompts), num_vq_tokens),
                uni_prompting.ignore_id,
                dtype=torch.long,
                device=device,
            )
            input_ids, attention_mask, _ = uni_prompting(
                (
                    batch_prompts,
                    input_image_tokens,
                    output_placeholder,
                    labels_placeholder,
                ),
                "i2i",
            )
            attention_mask = attention_mask.bool()
            if cfg.training.guidance_scale > 0:
                uncond_input_ids, uncond_attention_mask, _ = uni_prompting(
                    (
                        [""] * len(batch_prompts),
                        input_image_tokens,
                        output_placeholder,
                        labels_placeholder,
                    ),
                    "i2i",
                )
                uncond_attention_mask = uncond_attention_mask.bool()
            else:
                uncond_input_ids = None
                uncond_attention_mask = None
        else:
            input_ids, attention_mask = uni_prompting(
                (batch_prompts, input_image_tokens, output_placeholder), "i2i_gen"
            )

            if cfg.training.guidance_scale > 0:
                uncond_input_ids, uncond_attention_mask = uni_prompting(
                    ([""] * len(batch_prompts), input_image_tokens, output_placeholder),
                    "i2i_gen",
                )
            else:
                uncond_input_ids = None
                uncond_attention_mask = None

        with torch.no_grad():
            gen_token_ids = model.i2i_generate(
                input_ids=input_ids,
                uncond_input_ids=uncond_input_ids,
                attention_mask=attention_mask,
                uncond_attention_mask=uncond_attention_mask,
                guidance_scale=cfg.training.guidance_scale,
                temperature=cfg.training.get(
                    "generation_temperature", args.generation_temperature
                ),
                noise_schedule=mask_schedule,
                noise_type=cfg.training.get("noise_type", "mask"),
                seq_len=num_vq_tokens,
                uni_prompting=uni_prompting,
                config=cfg,
            )

        gen_token_ids = torch.clamp(gen_token_ids, max=codebook_size - 1, min=0)
        images = vq_model.decode_code(gen_token_ids)
        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0) * 255.0
        images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

        for key, img_arr in zip(batch_keys, images):
            out_path = os.path.join(args.outdir, f"{key}.png")
            Image.fromarray(img_arr).save(out_path)


if __name__ == "__main__":
    main()
