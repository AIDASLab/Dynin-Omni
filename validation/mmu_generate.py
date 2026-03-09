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

import json
import os
import sys

import cv2
import numpy as np
import torch
import wandb
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import MAGVITv2, MMadaModelLM, DyninOmniModelLM
from training.config_resolver import (
    configure_hf_cache_env,
    resolve_model_local_files_only,
    resolve_model_pretrained_source,
    resolve_model_type_from_pretrained,
    resolve_tokenizer_source,
    resolve_vq_repo_source,
)
from training.prompting_utils import UniversalPrompting
from training.utils import flatten_omega_conf, get_config, image_transform

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
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


def _resolve_model_class(pretrained_path: str, local_files_only: bool = False):
    model_type = resolve_model_type_from_pretrained(
        pretrained_path,
        local_files_only=local_files_only,
    )
    return DyninOmniModelLM if model_type == "dynin_omni" else MMadaModelLM


def _parse_questions(config):
    questions = config.get("question", None)
    if questions is None:
        return ["Please describe this input in detail."]
    if isinstance(questions, str):
        parsed = [q.strip() for q in questions.split(" *** ") if q.strip()]
        return parsed if parsed else ["Please describe this input in detail."]
    if isinstance(questions, (list, tuple)):
        parsed = [str(q).strip() for q in questions if str(q).strip()]
        return parsed if parsed else ["Please describe this input in detail."]
    return ["Please describe this input in detail."]


def _build_query_input_ids(uni_prompting, question, device):
    prompt = (
        "<|start_header_id|>user<|end_header_id|>\n"
        + question
        + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )
    query_ids = uni_prompting.text_tokenizer([prompt])["input_ids"]
    return torch.tensor(query_ids, dtype=torch.long, device=device)


def _concat_mmu_inputs(task_token, media_tokens, query_ids, uni_prompting, device):
    batch_size = query_ids.shape[0]
    task_id = int(uni_prompting.sptids_dict[task_token])
    soi_id = int(uni_prompting.sptids_dict["<|soi|>"])
    eoi_id = int(uni_prompting.sptids_dict["<|eoi|>"])
    sot_id = int(uni_prompting.sptids_dict["<|sot|>"])

    task = torch.full((batch_size, 1), task_id, device=device, dtype=torch.long)
    soi = torch.full((batch_size, 1), soi_id, device=device, dtype=torch.long)
    eoi = torch.full((batch_size, 1), eoi_id, device=device, dtype=torch.long)
    sot = torch.full((batch_size, 1), sot_id, device=device, dtype=torch.long)
    return torch.cat([task, soi, media_tokens, eoi, sot, query_ids], dim=1)


def _load_video_tokens(
    video_path, config, uni_prompting, vq_model, device, sample_method, num_frames
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    cap.release()

    total_frames = len(frames)
    if total_frames < num_frames:
        raise ValueError(
            f"Video {video_path} has {total_frames} frames, required >= {num_frames}."
        )

    if sample_method in {"uniform", "uniform_sequential"}:
        indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    else:
        raise ValueError(f"Sampling method {sample_method} not supported.")

    token_list = []
    for idx in indices:
        frame = image_transform(
            frames[idx], resolution=config.dataset.params.resolution
        ).to(device)
        frame = frame.unsqueeze(0)
        token_list.append(vq_model.get_code(frame) + len(uni_prompting.text_tokenizer))
    return torch.cat(token_list, dim=1)


def _run_i2t(config, model, vq_model, uni_prompting, questions, device):
    image_root = config.get("mmu_image_root", None)
    if image_root is None:
        image_root = getattr(config.dataset.params, "mmu_image_root", None)
    if image_root is None or not os.path.isdir(image_root):
        print(f"[ERROR] Skipped: invalid image root ({image_root})")
        return []

    image_files = sorted(
        f
        for f in os.listdir(image_root)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    )
    if not image_files:
        print(f"[ERROR] Skipped: no image files in {image_root}")
        return []

    max_new_tokens = int(config.get("mmu_i2t_max_new_tokens", 1024))
    steps = int(config.get("mmu_i2t_steps", 512))
    block_length = int(config.get("mmu_i2t_block_length", 1024))
    results = []

    for file_name in tqdm(image_files, desc="MMU i2t"):
        image_path = os.path.join(image_root, file_name)
        try:
            image_ori = Image.open(image_path).convert("RGB")
        except Exception as exc:
            print(f"[ERROR] Failed to load {image_path}: {exc}")
            continue

        image = image_transform(
            image_ori, resolution=config.dataset.params.resolution
        ).to(device)
        image = image.unsqueeze(0)
        image_tokens = vq_model.get_code(image) + len(uni_prompting.text_tokenizer)

        for question in questions:
            query_ids = _build_query_input_ids(uni_prompting, question, device)
            input_ids = _concat_mmu_inputs(
                "<|mmu|>", image_tokens, query_ids, uni_prompting, device
            )
            with torch.no_grad():
                output_ids = model.mmu_generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    steps=steps,
                    block_length=block_length,
                )
            response = uni_prompting.text_tokenizer.batch_decode(
                output_ids[:, input_ids.shape[1] :], skip_special_tokens=True
            )[0]
            print(f"[MMU] {file_name} | Q: {question} | A: {response}")
            results.append(
                {
                    "modality": "i2t",
                    "file_name": file_name,
                    "question": question,
                    "response": response,
                }
            )
    return results


def _run_v2t(config, model, vq_model, uni_prompting, questions, device):
    video_root = config.get("video_image_root", None)
    if video_root is None:
        video_root = getattr(
            getattr(config, "evaluation", None), "v2t_video_root", None
        )
    if video_root is None:
        video_root = getattr(config.dataset.params, "video_root", None)
    if video_root is None or not os.path.isdir(video_root):
        print(f"[ERROR] Skipped: invalid video root ({video_root})")
        return []

    video_files = sorted(
        f
        for f in os.listdir(video_root)
        if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm"))
    )
    if not video_files:
        print(f"[ERROR] Skipped: no video files in {video_root}")
        return []

    video_cfg = getattr(config.dataset.params, "video_caption_dataset", None)
    default_frames = (
        int(getattr(video_cfg, "num_frames", 8)) if video_cfg is not None else 8
    )
    default_sample = (
        getattr(video_cfg, "sample_method", "uniform")
        if video_cfg is not None
        else "uniform"
    )

    num_frames = int(config.get("mmu_v2t_num_frames", default_frames))
    sample_method = str(config.get("mmu_v2t_sample_method", default_sample))
    max_new_tokens = int(config.get("mmu_v2t_max_new_tokens", 128))
    steps = int(config.get("mmu_v2t_steps", 128))
    block_length = int(config.get("mmu_v2t_block_length", 128))

    results = []
    for file_name in tqdm(video_files, desc="MMU v2t"):
        video_path = os.path.join(video_root, file_name)
        try:
            video_tokens = _load_video_tokens(
                video_path=video_path,
                config=config,
                uni_prompting=uni_prompting,
                vq_model=vq_model,
                device=device,
                sample_method=sample_method,
                num_frames=num_frames,
            )
        except Exception as exc:
            print(f"[ERROR] Skipped {video_path}: {exc}")
            continue

        for question in questions:
            query_ids = _build_query_input_ids(uni_prompting, question, device)
            input_ids = _concat_mmu_inputs(
                "<|v2t|>", video_tokens, query_ids, uni_prompting, device
            )
            with torch.no_grad():
                output_ids = model.mmu_generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    steps=steps,
                    block_length=block_length,
                )
            response = uni_prompting.text_tokenizer.batch_decode(
                output_ids[:, input_ids.shape[1] :], skip_special_tokens=True
            )[0]
            print(f"[V2T] {file_name} | Q: {question} | A: {response}")
            results.append(
                {
                    "modality": "v2t",
                    "file_name": file_name,
                    "question": question,
                    "response": response,
                }
            )
    return results


def main():
    config = get_config()
    configure_hf_cache_env(config, project_root=PROJECT_ROOT)

    wandb_cfg = getattr(config, "wandb", None)
    use_wandb = bool(wandb_cfg is not None and wandb_cfg.get("enable", True))
    if use_wandb:
        run_id = wandb_cfg.get("run_id", None)
        if run_id is None:
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb.init(
            project="demo",
            name=config.experiment.name + "_mmu",
            config=wandb_config,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vq_cfg = _resolve_vq_cfg(config)
    local_only = resolve_model_local_files_only(config, default=False)

    model_path = config.get("mmu_model_path", resolve_model_pretrained_source(config))
    tokenizer_path = resolve_tokenizer_source(config, default=model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
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
            "<|v2t|>",
        ),
        ignore_id=-100,
        cond_dropout_prob=config.training.cond_dropout_prob,
        use_reserved_token=True,
    )

    vq_model_cls = get_vq_model_class(vq_cfg.type)
    vq_model = vq_model_cls.from_pretrained(
        resolve_vq_repo_source(vq_cfg),
        local_files_only=local_only,
    ).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()

    model_cls = _resolve_model_class(model_path, local_files_only=local_only)
    load_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = model_cls.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=load_dtype,
        local_files_only=local_only,
    ).to(device)
    model.eval()

    questions = _parse_questions(config)
    i2t_results = _run_i2t(config, model, vq_model, uni_prompting, questions, device)
    v2t_results = _run_v2t(config, model, vq_model, uni_prompting, questions, device)
    all_results = i2t_results + v2t_results

    output_path = config.get(
        "mmu_output_file", os.path.join(os.getcwd(), "mmu_inference_results.json")
    )
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if not all_results:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        print(
            f"[WARN] No MMU samples processed. Saved empty result file to {output_path}."
        )
        return

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(all_results)} results to {output_path}")

    if use_wandb and wandb.run is not None:
        table = wandb.Table(columns=["modality", "file_name", "question", "response"])
        for row in all_results:
            table.add_data(
                row["modality"], row["file_name"], row["question"], row["response"]
            )
        wandb.log({"mmu_results": table})


if __name__ == "__main__":
    main()
