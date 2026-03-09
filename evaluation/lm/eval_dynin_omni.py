import accelerate
import torch
import re
from pathlib import Path
import random
import numpy as np
import torch.nn.functional as F

if not hasattr(np, "complex"):
    np.complex = complex
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
import os
import sys

# Make the repository root importable when running from evaluation/lm.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from transformers import AutoTokenizer
from generate import generate
from models import DyninOmniModelLM
import json
import time

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


system_prompt = "You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here"
system_prompt = ""
NTM_TOKEN = ""
THINK_TOKEN = "<think>"
TASKS_USE_THINK = {"gsm8k", "hendrycks_math500"}
TASKS_USE_NTM = set()
EOS_PAD_LENGTH = None
MAX_EVAL_LENGTH = 4096


@register_model("dynin_omni_dist")
class DyninOmniEvalHarness(LM):
    def __init__(
        self,
        model_path="",
        mask_id=126336,
        batch_size=48,
        mc_num=1,
        is_check_greedy=True,
        steps=1024,
        gen_length=1024,
        block_length=1024,
        remasking="low_confidence",
        device="cuda",
        threshold=None,
        factor=None,
        logits_eos_inf=False,
        confidence_eos_eot_inf=True,
        save_dir=None,
        show_speed=False,
        cfg_scale=0.0,
        shard_id=0,
        num_shards=1,
        use_fastdllm_v1=False,
        model_config_path=None,
        **kwargs,
    ):
        """
        Args:
            model_path: DyninOmni model path.
            mask_id: The token id of [MASK] is 126336.
            batch_size: mini batch size.
            mc_num: Monte Carlo estimation iterations
            is_check_greedy: For certain metrics like LAMBADA, the evaluation requires the model to verify whether the answer
                             is generated through greedy sampling conditioned on the prompt (note that this differs from conditional
                             generation). We implement this verification through the suffix_greedy_prediction() function, which
                             returns a True/False judgment used for accuracy calculation.
                             When is_check_greedy is set to True, the lm-evaluation-harness library automatically invokes this function.
                             However, since none of the metrics in the LLaDA paper (https://arxiv.org/abs/2502.09992) require this functionality,
                             we recommend setting is_check_greedy to False. This configuration causes suffix_greedy_prediction() to return False
                             by default, significantly accelerating the evaluation process.
            cfg_scale: Unsupervised classifier-free guidance scale.
        """
        super().__init__()

        self.device = torch.device(device)
        self._rank = 0
        self._world_size = 1

        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
            self.device = torch.device(accelerator.device)
            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
        else:
            self.accelerator = None

        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({"device_map": {"": f"{self.accelerator.device}"}})

        pretrained_kwargs = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        }
        if model_config_path:
            pretrained_kwargs["config"] = model_config_path
        elif os.path.isdir(model_path):
            local_config = os.path.join(model_path, "config.json")
            if os.path.isfile(local_config):
                pretrained_kwargs["config"] = local_config

        self.model = DyninOmniModelLM.from_pretrained(
            model_path,
            **pretrained_kwargs,
        ).to(self.device)
        self.model.eval()

        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
        else:
            self.model = self.model.to(self.device)

        self.mask_id = mask_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        self.cfg = float(cfg_scale)
        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}.")
        self.sampling_eps = 0.0
        self.is_check_greedy = is_check_greedy

        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.remasking = remasking
        self.threshold = threshold
        self.factor = factor
        self.logits_eos_inf = logits_eos_inf
        self.confidence_eos_eot_inf = confidence_eos_eot_inf
        # self.is_instruct = True if 'instruct' in model_path.lower() else False
        self.is_instruct = True
        self.save_dir = save_dir
        self.show_speed = show_speed
        self.use_fastdllm_v1 = use_fastdllm_v1
        self.shard_id = int(shard_id)
        self.num_shards = int(num_shards)

    def _apply_shard(self, requests):
        if self.num_shards <= 1:
            return
        if self.shard_id < 0 or self.shard_id >= self.num_shards:
            raise ValueError(
                f"shard_id must be in [0, {self.num_shards - 1}], got {self.shard_id}."
            )
        shard_requests = [
            req
            for i, req in enumerate(requests)
            if i % self.num_shards == self.shard_id
        ]
        requests[:] = shard_requests

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _forward_process(self, batch, prompt_index, target_lens=None):
        b, l = batch.shape
        if prompt_index.dim() == 1:
            prompt_index = prompt_index.unsqueeze(0).repeat(b, 1)

        prompt_lens = prompt_index.sum(dim=1).long()
        if target_lens is None:
            target_lens = (l - prompt_lens).long()
        else:
            target_lens = target_lens.long()

        x = torch.empty(b, device=batch.device, dtype=torch.long)
        is_mask = torch.zeros((b, l), dtype=torch.bool, device=batch.device)
        for i in range(b):
            tlen = int(target_lens[i].item())
            if tlen <= 0:
                continue
            x[i] = torch.randint(1, tlen + 1, (), device=batch.device)
            mask = torch.arange(tlen, device=batch.device) < x[i]
            mask = mask[torch.randperm(tlen)]
            start = int(prompt_lens[i].item())
            is_mask[i, start : start + tlen] = mask

        noisy_batch = torch.where(is_mask, self.mask_id, batch)
        p_mask = (x / target_lens).unsqueeze(1).repeat(1, l)
        return noisy_batch, p_mask

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.cfg > 0.0:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if self.cfg > 0.0:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, : batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood_batch(self, prefixes, targets):
        if len(prefixes) == 0:
            return []
        seq_lens = [len(p) + len(t) for p, t in zip(prefixes, targets)]
        max_len = max(seq_lens)
        if any(len(t) == 0 for t in targets):
            raise ValueError("target length must be > 0 for loglikelihood.")
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        seq = torch.full(
            (len(prefixes), max_len), pad_id, dtype=torch.long, device=self.device
        )
        prompt_index = torch.zeros(
            (len(prefixes), max_len), dtype=torch.bool, device=self.device
        )
        target_lens = torch.zeros(len(prefixes), dtype=torch.long, device=self.device)
        for i, (p, t) in enumerate(zip(prefixes, targets)):
            full = torch.concatenate([p, t]).to(self.device)
            seq[i, : len(full)] = full
            prompt_index[i, : len(p)] = True
            target_lens[i] = len(t)

        loss_acc = torch.zeros(seq.shape[0], device=self.device)
        remaining = self.mc_num
        mc_batch_size = max(1, self.batch_size // seq.shape[0])
        vocab_size = None
        while remaining > 0:
            cur_mc = min(mc_batch_size, remaining)
            expanded_seq = seq.repeat_interleave(cur_mc, dim=0)
            expanded_prompt_index = prompt_index.repeat_interleave(cur_mc, dim=0)
            expanded_target_lens = target_lens.repeat_interleave(cur_mc, dim=0)
            perturbed_seq, p_mask = self._forward_process(
                expanded_seq,
                expanded_prompt_index,
                target_lens=expanded_target_lens,
            )

            mask_indices = perturbed_seq == self.mask_id
            logits = self.get_logits(perturbed_seq, expanded_prompt_index)
            if vocab_size is None:
                vocab_size = logits.shape[-1]

            token_loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                expanded_seq.view(-1),
                reduction="none",
            ).view(expanded_seq.shape[0], -1)
            token_loss = token_loss / p_mask
            token_loss = token_loss * mask_indices
            sample_loss = token_loss.sum(dim=1).view(seq.shape[0], cur_mc).sum(dim=1)
            loss_acc += sample_loss
            remaining -= cur_mc

        loss_acc = loss_acc / self.mc_num
        return (-loss_acc).tolist()

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False

        seq = torch.full(
            (1, len(prefix) + len(target)), self.mask_id, device=self.device
        )
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, : len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = seq == self.mask_id
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(
                dim=-1
            )
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix) :]
        correct = torch.all(correct)
        return correct

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    @staticmethod
    def _ensure_ntm_prefix(text: str) -> str:
        stripped = text.lstrip()
        if stripped.startswith(NTM_TOKEN):
            return text
        return f"{NTM_TOKEN}{text}"

    @staticmethod
    def _append_think_token(prompt: str) -> str:
        if not prompt:
            return prompt
        if prompt.rstrip().endswith(THINK_TOKEN):
            return prompt
        return f"{prompt}{THINK_TOKEN}"

    @staticmethod
    def _normalize_task_name(task_name: str) -> str:
        return str(task_name).strip().lower() if task_name else ""

    def _resolve_task_name(self, req) -> str:
        if hasattr(req, "task_name") and req.task_name:
            return self._normalize_task_name(req.task_name)
        if hasattr(req, "task") and req.task:
            return self._normalize_task_name(req.task)
        if hasattr(req, "doc") and isinstance(req.doc, dict):
            for key in ("task", "dataset", "subject"):
                if key in req.doc:
                    return self._normalize_task_name(req.doc.get(key))
        return ""

    def _build_prompt(self, question: str, task_name: str = "") -> str:
        task_name = self._normalize_task_name(task_name)
        use_think = task_name in TASKS_USE_THINK
        prompt = question.rstrip("\n")
        if use_think:
            if prompt.startswith("Question:"):
                prompt = prompt[len("Question:") :].lstrip()
            answer_idx = prompt.find("\nAnswer:")
            if answer_idx != -1:
                prompt = prompt[:answer_idx].rstrip()
            elif prompt.endswith("Answer:"):
                prompt = prompt[: -len("Answer:")].rstrip()

        suffix = "/think" if use_think else "/no_think"
        user_text = f"{prompt}\n{suffix}"

        if system_prompt:
            user_text = system_prompt + "\n\n" + user_text

        if self.is_instruct:
            return (
                "<|start_header_id|>user<|end_header_id|>\n"
                f"{user_text}\n"
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            )
        return user_text

    def loglikelihood(self, requests):
        self._apply_shard(requests)

        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = []
        for req in requests:
            task_name = self._resolve_task_name(req)
            ds.append(
                {
                    "prefix": self._build_prompt(req.args[0], task_name),
                    "target": req.args[1],
                }
            )
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= MAX_EVAL_LENGTH

        with torch.no_grad():
            out = [None] * len(ds)
            for start in tqdm(
                range(0, len(ds), self.batch_size), desc="Computing likelihood..."
            ):
                batch_indices = list(
                    range(start, min(start + self.batch_size, len(ds)))
                )
                prefixes = [ds[i]["prefix"] for i in batch_indices]
                targets = [ds[i]["target"] for i in batch_indices]
                lls = self.get_loglikelihood_batch(prefixes, targets)
                for i, ll in zip(batch_indices, lls):
                    prefix = ds[i]["prefix"]
                    target = ds[i]["target"]
                    is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)
                    out[i] = (ll, 1.0 if is_target_greedy_dec else 0.0)
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests):
        if self.rank == 0:
            print(f"total_requests_before_shard: {len(requests)}")
        self._apply_shard(requests)
        if self.rank == 0:
            print(
                f"total_requests_after_shard: {len(requests)} (rank={self.rank}, world_size={self.world_size})"
            )
        output = []
        num_tokens = 0
        num_nfe = 0
        processed_count = 0
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            rank = self.rank
            save_path = os.path.join(self.save_dir, f"rank_{rank}.jsonl")
            print(f"save_path: {save_path}")
            if os.path.exists(save_path):
                print(f"load from {save_path}")
                with open(save_path, "r", encoding="utf-8") as f:
                    output = [json.loads(line) for line in f]
                    processed_count = len(output)
                print(f"processed_count: {processed_count}")

        batched_requests = [[]]
        for i, req in enumerate(tqdm(requests, desc="Batching...")):
            if i < processed_count:
                continue
            batched_requests[-1].append(req)
            if len(batched_requests[-1]) == self.batch_size:
                batched_requests.append([])

        if len(batched_requests[-1]) == 0:
            batched_requests.pop()

        start_time = time.time()

        for batch in tqdm(batched_requests, desc="Generating..."):
            batched_input_ids = []
            prompt_lens = []
            use_think_flags = []
            max_len = 0
            for req in batch:
                task_name = self._resolve_task_name(req)
                use_think_flags.append(task_name in TASKS_USE_THINK)
                user_input = self._build_prompt(req.args[0], task_name)
                input_ids = self.tokenizer(user_input)["input_ids"]
                prompt_lens.append(len(input_ids))
                batched_input_ids.append(input_ids)
                max_len = max(max_len, len(input_ids))
            pad_len = [max_len - len(input_ids) for input_ids in batched_input_ids]

            gen_length = self.gen_length
            block_length = self.block_length
            steps = self.steps

            # pad batched_input_ids to the same length
            eos_pad_id = (
                self.tokenizer.eos_token_id
                if self.tokenizer.eos_token_id is not None
                else self.tokenizer.pad_token_id
            )
            batched_input_ids = [
                torch.cat(
                    [
                        torch.tensor(
                            input_ids, dtype=torch.long, device=self.device
                        ).unsqueeze(0),
                        torch.full(
                            (1, max_len - len(input_ids)),
                            eos_pad_id,
                            dtype=torch.long,
                            device=self.device,
                        ),
                    ],
                    dim=1,
                )
                for input_ids in batched_input_ids
            ]
            batched_input_ids = torch.cat(batched_input_ids, dim=0)
            batched_input_ids = batched_input_ids.to(self.device)

            if self.batch_size == 1:
                attention_mask = None
            else:
                attention_mask = torch.zeros(
                    (
                        batched_input_ids.shape[0],
                        1,
                        max_len + self.gen_length,
                        max_len + self.gen_length,
                    ),
                    device=self.device,
                    dtype=torch.bool,
                )
                for i in range(len(pad_len)):
                    attention_mask[i, :, pad_len[i] :, pad_len[i] :] = True

            stop_tokens_list = [r.args[1]["until"] for r in batch]
            input_ids = batched_input_ids
            if self.use_fastdllm_v1:
                fastdllm_model = (
                    self.model.module if hasattr(self.model, "module") else self.model
                )
                generated_answer = fastdllm_model.mmu_generate_fastdllm_v1(
                    input_ids,
                    max_new_tokens=gen_length,
                    steps=steps,
                    block_length=block_length,
                    temperature=0,
                    remasking=self.remasking,
                    mask_id=self.mask_id,
                    attention_mask=attention_mask,
                    use_cache=False,
                    threshold=self.threshold,
                    factor=self.factor,
                )
                nfe = steps
            else:
                generated_answer, nfe = generate(
                    self.model,
                    input_ids,
                    attention_mask=attention_mask,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=0,
                    cfg_scale=self.cfg,
                    remasking=self.remasking,
                    mask_id=self.mask_id,
                    logits_eos_inf=self.logits_eos_inf,
                    confidence_eos_eot_inf=self.confidence_eos_eot_inf,
                )

            is_humaneval_single = (
                self.is_instruct
                and len(batch) == 1
                and "task_id" in batch[0].doc
                and str(batch[0].doc["task_id"]).lower().startswith("humaneval")
            )
            if is_humaneval_single:
                if self.show_speed:
                    num_tokens += (generated_answer != 126081).sum()
                    num_nfe += nfe
                generated_answer = self.tokenizer.decode(
                    generated_answer[0][input_ids.shape[1] :], skip_special_tokens=False
                )
                batched_generated_answer = [generated_answer]
                raw_generated_answer = [generated_answer]
            else:
                batched_generated_answer = []
                raw_generated_answer = []
                for i in range(len(generated_answer)):
                    gen_ids = generated_answer[i][prompt_lens[i] :]
                    target_len = EOS_PAD_LENGTH or self.gen_length
                    if target_len and len(gen_ids) < target_len:
                        eos_id = self.tokenizer.eos_token_id
                        if eos_id is None:
                            eos_id = self.tokenizer.pad_token_id or 0
                        pad_len = target_len - len(gen_ids)
                        pad = torch.full(
                            (pad_len,),
                            eos_id,
                            dtype=gen_ids.dtype,
                            device=gen_ids.device,
                        )
                        gen_ids = torch.cat([gen_ids, pad], dim=0)
                    generated_answer_i = self.tokenizer.decode(
                        gen_ids, skip_special_tokens=False
                    )
                    raw_generated_answer.append(generated_answer_i)
                    if use_think_flags[
                        i
                    ] and not generated_answer_i.lstrip().startswith(THINK_TOKEN):
                        generated_answer_i = f"{THINK_TOKEN}{generated_answer_i}"
                    for stop_seq in stop_tokens_list[i]:
                        if stop_seq in generated_answer_i:
                            generated_answer_i = generated_answer_i.split(stop_seq)[0]
                    if self.show_speed:
                        eos_id = self.tokenizer.eos_token_id
                        if eos_id is None:
                            eos_id = self.tokenizer.pad_token_id or 0
                        num_tokens += (gen_ids != eos_id).sum()
                        num_nfe += nfe
                    batched_generated_answer.append(generated_answer_i)

            # output.append(generated_answer)
            output.extend(batched_generated_answer)

            if self.save_dir is not None:
                with open(save_path, "a", encoding="utf-8") as f:
                    for generated_answer in batched_generated_answer:
                        f.write(json.dumps(generated_answer, ensure_ascii=False) + "\n")

            for i in range(len(batched_generated_answer)):
                print("=" * 20)
                if "raw_generated_answer" in locals() and i < len(raw_generated_answer):
                    print("answer: ", raw_generated_answer[i])
                else:
                    print("answer: ", batched_generated_answer[i])
                print("nfe: ", nfe)
                print("avg nfe: ", num_nfe / len(output))
                print("=" * 20, end="\n\n")
        end_time = time.time()
        if self.show_speed:
            print(f"Total number of tokens generated: {num_tokens}")
            print(f"Total time taken: {end_time - start_time} seconds")
            print(f"Tokens per second: {num_tokens / (end_time - start_time)}")
            print(f"Total NFE is {num_nfe}")

        return output


if __name__ == "__main__":
    cli_evaluate()
