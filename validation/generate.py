import os
import sys
import json

import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import MMadaModelLM, DyninOmniModelLM
from training.config_resolver import (
    configure_hf_cache_env,
    resolve_model_local_files_only,
    resolve_model_pretrained_source,
    resolve_model_type_from_pretrained,
    resolve_tokenizer_source,
)
from training.utils import get_config

DEFAULT_TEXT_QUESTIONS = [
    "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"
]


def _resolve_model_cfg(config):
    if "dynin_omni" in config.model:
        return config.model.dynin_omni
    if "mmada" in config.model:
        return config.model.mmada
    raise ValueError("Config is missing model.dynin_omni/model.mmada block.")


def _resolve_model_class(pretrained_path: str, local_files_only: bool = False):
    model_type = resolve_model_type_from_pretrained(
        pretrained_path,
        local_files_only=local_files_only,
    )
    return DyninOmniModelLM if model_type == "dynin_omni" else MMadaModelLM


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


def _load_questions_from_jsonl(path: str):
    questions = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                candidate = text
            else:
                if isinstance(parsed, str):
                    candidate = parsed
                elif isinstance(parsed, dict):
                    candidate = (
                        parsed.get("question")
                        or parsed.get("prompt")
                        or parsed.get("text")
                        or parsed.get("query")
                        or ""
                    )
                else:
                    candidate = ""
            candidate = str(candidate).strip()
            if candidate:
                questions.append(candidate)
    return questions


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
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
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


def main():
    config = get_config()
    configure_hf_cache_env(config, project_root=PROJECT_ROOT)
    _resolve_model_cfg(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_path = resolve_model_pretrained_source(config)
    tokenizer_path = resolve_tokenizer_source(config, default=pretrained_path)
    local_files_only = resolve_model_local_files_only(config, default=False)
    model_cls = _resolve_model_class(
        pretrained_path,
        local_files_only=local_files_only,
    )
    load_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model = model_cls.from_pretrained(
        pretrained_path,
        trust_remote_code=True,
        torch_dtype=load_dtype,
        local_files_only=local_files_only,
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )

    tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n' }}"
    questions_file = config.get("questions_file", None)
    questions = []
    if questions_file:
        questions_path = str(questions_file)
        if not os.path.isabs(questions_path):
            questions_path = os.path.join(PROJECT_ROOT, questions_path)
        if os.path.isfile(questions_path):
            questions = _load_questions_from_jsonl(questions_path)
            if not questions:
                print(
                    f"[WARN] questions_file is empty or invalid ({questions_path}); using built-in prompt."
                )
        else:
            print(f"[WARN] questions_file not found ({questions_path}); using built-in prompt.")

    if not questions:
        questions = list(DEFAULT_TEXT_QUESTIONS)

    for idx, prompt in enumerate(questions, start=1):
        messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        input_ids = tokenizer(
            text=prompt_text, return_tensors="pt", padding=True, padding_side="left"
        )["input_ids"].to(device)
        out = generate(
            model,
            input_ids,
            steps=128,
            gen_length=512,
            block_length=128,
            temperature=1,
            cfg_scale=0.0,
            remasking="low_confidence",
        )
        answer = tokenizer.batch_decode(
            out[:, input_ids.shape[1] :], skip_special_tokens=True
        )[0]
        print(f"[Q{idx}] {prompt}")
        print(f"[A{idx}] {answer}")


if __name__ == "__main__":
    main()
