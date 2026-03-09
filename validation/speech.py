#!/usr/bin/env python3

import argparse
import json
import os
import sys
import random
import re
import glob
import unicodedata
from functools import partial
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Configure HF cache env before importing transformers so runtime does not
# inherit unwritable global cache paths such as /models.
_default_hf_cache_root = (
    os.environ.get("DYNIN_OMNI_HF_CACHE_DIR")
    or os.environ.get("HF_HOME")
    or str(Path(PROJECT_ROOT) / "datasets" / "huggingface")
)
_default_hf_cache_root = str(Path(_default_hf_cache_root).expanduser())
os.environ["DYNIN_OMNI_HF_CACHE_DIR"] = _default_hf_cache_root
os.environ["HF_HOME"] = _default_hf_cache_root
os.environ["HF_DATASETS_CACHE"] = _default_hf_cache_root
os.environ["HF_HUB_CACHE"] = str(Path(_default_hf_cache_root) / "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.environ["HF_HUB_CACHE"]
os.environ.pop("TRANSFORMERS_CACHE", None)
Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["HF_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import soundfile as sf
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from datasets import Dataset as HFDataset, load_dataset
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from transformers import AutoTokenizer, pipeline

from models import DyninOmniModelLM
from models.modeling_emova_speech_tokenizer import EMOVASpeechTokenizer
from training.config_resolver import (
    configure_hf_cache_env,
    resolve_model_local_files_only,
    resolve_model_pretrained_source,
    resolve_tokenizer_source,
    resolve_vq_repo_source,
)
from training.data import S2T_INSTRUCTION, T2S_INSTRUCTION
from training.prompting_utils import UniversalPrompting

try:
    import editdistance  # type: ignore
except Exception:

    def _levenshtein_distance(seq_a, seq_b):
        len_a, len_b = len(seq_a), len(seq_b)
        if len_a == 0:
            return len_b
        if len_b == 0:
            return len_a
        prev = list(range(len_b + 1))
        curr = [0] * (len_b + 1)
        for i in range(1, len_a + 1):
            curr[0] = i
            a_tok = seq_a[i - 1]
            for j in range(1, len_b + 1):
                cost = 0 if a_tok == seq_b[j - 1] else 1
                curr[j] = min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + cost,
                )
            prev, curr = curr, prev
        return prev[len_b]

    class _EditDistanceCompat:
        @staticmethod
        def eval(seq_a, seq_b):
            return _levenshtein_distance(seq_a, seq_b)

    editdistance = _EditDistanceCompat()


_ASR_WS_RE = re.compile(r"\s+")


def normalize_asr_text(text: str) -> str:
    """
    Aggressive ASR text normalization for WER.

    - lowercases
    - removes punctuation/symbols
    - keeps letters and numbers across languages
    - collapses whitespace
    """
    if text is None:
        return ""
    text = text.replace("<|endoftext|>", "")
    text = text.lower()
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("L") or cat.startswith("N"):
            out.append(ch)
        elif ch.isspace():
            out.append(" ")
        # Drop punctuation, symbols, and controls.
    text = "".join(out)
    text = _ASR_WS_RE.sub(" ", text).strip()
    return text


def _get_audio_field(audio_entry, key: str):
    if audio_entry is None:
        return None
    if isinstance(audio_entry, dict):
        return audio_entry.get(key)
    try:
        return audio_entry[key]
    except Exception:
        return getattr(audio_entry, key, None)


class S2TEvalDataset(Dataset):
    def __init__(self, hf_dataset, root_path):
        self.hf_dataset = hf_dataset
        self.root_path = root_path

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        if isinstance(idx, (list, tuple)):
            return [self.__getitem__(i) for i in idx]
        example = self.hf_dataset[idx]
        sample_id = example["id"]
        audio_entry = example.get("audio")
        if isinstance(audio_entry, str):
            audio_path = audio_entry
        else:
            audio_path = _get_audio_field(audio_entry, "path")
        if not audio_path and self.root_path:
            speaker_id, chapter_id, _ = sample_id.split("-")
            audio_path = os.path.join(
                self.root_path, speaker_id, chapter_id, f"{sample_id}.flac"
            )
        return {
            "audio": audio_entry,
            "audio_path": audio_path,
            "gt_text": example["text"],
            "sample_id": sample_id,
        }


class T2SEvalDataset(Dataset):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        example = self.hf_dataset[idx]
        return {"gt_text": example["text"], "sample_id": example["id"]}


def normalize_text(text):
    text = text.lower()
    text = text.replace("'", "")
    text = re.sub(r"[^\w\s]", "", text)
    return text


def calculate_wer(predictions, references):
    total_errors = 0
    total_words = 0
    for pred, ref in zip(predictions, references):
        pred_words = normalize_text(pred).split()
        ref_words = normalize_text(ref).split()
        total_errors += editdistance.eval(pred_words, ref_words)
        total_words += len(ref_words)
    wer = total_errors / total_words if total_words > 0 else 0.0
    return wer, total_errors, total_words


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        return rank, world_size
    return 0, 1


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def gather_results(local_results, world_size):
    if world_size == 1:
        return local_results
    all_results = [None] * world_size
    dist.all_gather_object(all_results, local_results)
    merged = []
    for chunk in all_results:
        if chunk:
            merged.extend(chunk)
    return merged


def _get_audio_sampling_rate(vq_model_audio) -> int:
    for attr in ("u2s_config", "u2s_cfg", "u2s_hps"):
        cfg = getattr(vq_model_audio, attr, None)
        if cfg is None:
            continue
        data = getattr(cfg, "data", None)
        if data is not None and hasattr(data, "sampling_rate"):
            return int(data.sampling_rate)
    return 16000


def _resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    try:
        from scipy.signal import resample_poly
    except Exception:
        # Fallback to linear interpolation if SciPy is unavailable.
        duration = audio.shape[0] / float(orig_sr)
        new_length = int(round(duration * target_sr))
        if new_length <= 0:
            return audio
        x_old = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
        x_new = np.linspace(0.0, duration, num=new_length, endpoint=False)
        return np.interp(x_new, x_old, audio).astype(audio.dtype)
    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    return resample_poly(audio, up, down).astype(audio.dtype)


def _find_test_arrow(hf_cache_dir: str) -> str:
    patterns = [
        os.path.join(
            hf_cache_dir,
            "openslr___librispeech_asr",
            "**",
            "librispeech_asr-test.arrow",
        ),
        os.path.join(
            hf_cache_dir, "librispeech_asr", "**", "librispeech_asr-test.arrow"
        ),
    ]
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return sorted(matches)[-1]
    return ""


def _load_librispeech_test_from_cache(hf_cache_dir: str) -> HFDataset:
    test_arrow = _find_test_arrow(hf_cache_dir)
    if not test_arrow:
        raise RuntimeError(
            "Cached librispeech_asr test split not found. "
            "Make sure the dataset cache exists in the HF cache dir."
        )
    return HFDataset.from_file(test_arrow)


def _load_librispeech_split(
    hf_cache_dir: str,
    split: str = "test",
    from_hf: bool = False,
    hf_parquet_only: bool = False,
    hf_max_files: int = 0,
) -> HFDataset:
    if from_hf:
        if hf_parquet_only:
            return _load_librispeech_split_from_hf_parquet(
                hf_cache_dir=hf_cache_dir,
                split=split,
                max_files=hf_max_files,
            )
        return load_dataset(
            "openslr/librispeech_asr",
            "clean",
            split=split,
            cache_dir=hf_cache_dir,
        )

    if split != "test":
        raise RuntimeError(
            f"split='{split}' requested, but local-arrow mode supports only split='test'. "
            "Use --librispeech-from-hf for test.other/test.clean."
        )
    return _load_librispeech_test_from_cache(hf_cache_dir)


def _split_to_patterns(split: str):
    if split == "test.other":
        return ["all/test.other/*.parquet"]
    if split == "test.clean":
        return ["all/test.clean/*.parquet"]
    if split == "test":
        return ["all/test.clean/*.parquet", "all/test.other/*.parquet"]
    raise ValueError(f"Unsupported split: {split}")


def _load_librispeech_split_from_hf_parquet(
    hf_cache_dir: str,
    split: str,
    max_files: int = 0,
) -> HFDataset:
    allow_patterns = _split_to_patterns(split)
    local_repo = snapshot_download(
        repo_id="openslr/librispeech_asr",
        repo_type="dataset",
        cache_dir=hf_cache_dir,
        allow_patterns=allow_patterns,
    )

    parquet_files = []
    for pattern in allow_patterns:
        parquet_files.extend(sorted(str(p) for p in Path(local_repo).glob(pattern)))

    if not parquet_files:
        raise RuntimeError(
            f"No parquet files resolved for split={split} under {local_repo}"
        )

    if max_files and max_files > 0:
        parquet_files = parquet_files[:max_files]

    return load_dataset(
        "parquet",
        data_files={"test": parquet_files},
        split="test",
        cache_dir=hf_cache_dir,
    )


def s2t_eval_collate_fn(
    batch, vq_model_audio, tokenizer, uni_prompting, config, audio_cache_dir
):
    audio_tokens_batch = []
    offset = len(uni_prompting.text_tokenizer) + int(config.model.dynin_omni.codebook_size)
    for item in batch:
        audio_entry = item["audio_path"]
        if isinstance(audio_entry, torch.Tensor):
            tokens = audio_entry.cpu()
        else:
            audio_path = None
            if isinstance(audio_entry, str) and os.path.exists(audio_entry):
                audio_path = audio_entry
            else:
                audio_info = item.get("audio")
                candidate = _get_audio_field(audio_info, "path")
                if isinstance(candidate, str) and os.path.exists(candidate):
                    audio_path = candidate
                else:
                    audio_array = _get_audio_field(audio_info, "array")
                    if audio_array is not None:
                        audio_path = os.path.join(
                            audio_cache_dir, f"{item['sample_id']}.wav"
                        )
                        if not os.path.exists(audio_path):
                            sampling_rate = _get_audio_field(
                                audio_info, "sampling_rate"
                            )
                            sampling_rate = int(sampling_rate or 16000)
                            sf.write(audio_path, np.asarray(audio_array), sampling_rate)
            if audio_path is None:
                raise FileNotFoundError(
                    f"Audio path missing for sample {item.get('sample_id', 'unknown')}"
                )
            tokens = vq_model_audio.encode(audio_path).cpu()
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
        full_prompt_text = (
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{prompt_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )
        prompt_tensor = tokenizer(full_prompt_text, return_tensors="pt").input_ids.to(
            device
        )

        final_sequence = torch.cat([audio_block, prompt_tensor], dim=1)
        batched_input_ids.append(final_sequence.squeeze(0))

    max_len = max(seq.size(0) for seq in batched_input_ids)
    pad_token_id = 126093
    final_batch_input_ids = torch.full(
        (len(batched_input_ids), max_len),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )

    for i, seq in enumerate(batched_input_ids):
        final_batch_input_ids[i, -len(seq) :] = seq

    return {
        "input_ids": final_batch_input_ids,
        "gt_texts": [item["gt_text"] for item in batch],
        "sample_ids": [item["sample_id"] for item in batch],
    }


def run_s2t(
    model,
    vq_model_audio,
    tokenizer,
    uni_prompting,
    config,
    args,
    device,
    rank,
    world_size,
    shard_id,
    num_shards,
):
    hf_dataset = _load_librispeech_split(
        args.hf_cache_dir,
        split=args.librispeech_split,
        from_hf=args.librispeech_from_hf,
        hf_parquet_only=args.librispeech_hf_parquet_only,
        hf_max_files=args.librispeech_hf_max_files,
    )
    if args.s2t_samples > 0:
        hf_dataset = hf_dataset.select(range(args.s2t_samples))
    if num_shards > 1:
        indices = list(range(shard_id, len(hf_dataset), num_shards))
        hf_dataset = hf_dataset.select(indices)
    eval_dataset = S2TEvalDataset(hf_dataset, root_path=args.librispeech_root)
    sampler = (
        DistributedSampler(
            eval_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        if world_size > 1
        else None
    )
    os.makedirs(args.s2t_audio_cache_dir, exist_ok=True)
    collate_for_eval = partial(
        s2t_eval_collate_fn,
        vq_model_audio=vq_model_audio,
        tokenizer=tokenizer,
        uni_prompting=uni_prompting,
        config=config,
        audio_cache_dir=args.s2t_audio_cache_dir,
    )
    dataloader = DataLoader(
        eval_dataset,
        batch_size=args.s2t_batch_size,
        sampler=sampler,
        shuffle=False if sampler is not None else False,
        collate_fn=collate_for_eval,
        num_workers=0,
        pin_memory=True,
    )

    local_results = []
    model.eval()
    for batch in tqdm(dataloader, desc="S2T eval", disable=(rank != 0)):
        input_ids = batch["input_ids"].to(device)
        gt_texts = batch["gt_texts"]
        sample_ids = batch["sample_ids"]

        with torch.no_grad():
            output_ids = model.mmu_generate(
                input_ids,
                max_new_tokens=args.s2t_new_tokens,
                steps=args.s2t_steps,
                block_length=args.s2t_block_length,
                remasking=args.s2t_remasking,
            )
            decoded_texts = tokenizer.batch_decode(
                output_ids[:, input_ids.shape[1] :], skip_special_tokens=True
            )

        eos_token = tokenizer.eos_token or "</s>"
        for i in range(len(decoded_texts)):
            full_text = decoded_texts[i]
            eos_idx = full_text.find(eos_token)
            cleaned_text = full_text[:eos_idx] if eos_idx != -1 else full_text
            cleaned_text = cleaned_text.replace(eos_token, "").strip()
            cleaned_text = normalize_asr_text(cleaned_text)
            local_results.append(
                {
                    "sample_id": sample_ids[i],
                    "gt_text": gt_texts[i],
                    "decoded_text": cleaned_text,
                }
            )

    return gather_results(local_results, world_size)


def run_t2s(
    model,
    vq_model_audio,
    tokenizer,
    uni_prompting,
    config,
    args,
    device,
    rank,
    world_size,
    shard_id,
    num_shards,
):
    hf_dataset = _load_librispeech_split(
        args.hf_cache_dir,
        split=args.librispeech_split,
        from_hf=args.librispeech_from_hf,
        hf_parquet_only=args.librispeech_hf_parquet_only,
        hf_max_files=args.librispeech_hf_max_files,
    )
    if args.t2s_samples > 0:
        hf_dataset = hf_dataset.select(range(args.t2s_samples))
    if num_shards > 1:
        indices = list(range(shard_id, len(hf_dataset), num_shards))
        hf_dataset = hf_dataset.select(indices)
    eval_dataset = T2SEvalDataset(hf_dataset)
    sampler = (
        DistributedSampler(
            eval_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        if world_size > 1
        else None
    )
    dataloader = DataLoader(
        eval_dataset,
        batch_size=args.t2s_batch_size,
        sampler=sampler,
        shuffle=False if sampler is not None else False,
        num_workers=0,
    )

    pipe_device = device.index if device.type == "cuda" else -1
    whisper_pipe = pipeline(
        "automatic-speech-recognition",
        model=args.whisper_model,
        device=pipe_device,
    )

    output_dir = args.out_dir
    os.makedirs(output_dir, exist_ok=True)

    local_results = []
    model.eval()
    mask_token_id = model.config.mask_token_id
    codebook_size = int(config.model.dynin_omni.codebook_size)
    speech_vocab_size = args.speech_vocab_size
    sampling_rate = _get_audio_sampling_rate(vq_model_audio)
    whisper_sr = whisper_pipe.feature_extractor.sampling_rate

    for batch_idx, batch in enumerate(
        tqdm(dataloader, desc="T2S eval", disable=(rank != 0))
    ):
        gt_texts = batch["gt_text"]
        sample_ids = batch["sample_id"]
        prompts = [
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{random.choice(T2S_INSTRUCTION)}\n{text}"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            for text in gt_texts
        ]

        batch_size = len(prompts)
        audio_tokens = (
            torch.ones(
                (batch_size, args.t2s_token_length), dtype=torch.long, device=device
            )
            * mask_token_id
        )
        input_ids, attention_mask = uni_prompting((prompts, audio_tokens), "t2s_gen")

        with torch.no_grad():
            output_ids = model.t2s_generate_mmu_like(
                input_ids=input_ids,
                max_new_tokens=args.t2s_token_length,
                steps=args.t2s_steps,
                block_length=args.t2s_block_length,
                temperature=args.t2s_temperature,
                cfg_scale=args.t2s_cfg_scale,
                mask_token_id=mask_token_id,
                attention_mask=attention_mask,
                uni_prompting=uni_prompting,
                codebook_size=codebook_size,
                audio_codebook_size=speech_vocab_size,
            )

        for i in range(batch_size):
            gt = gt_texts[i].rsplit("\n", 1)[-1].strip()
            token_list = output_ids[i].detach().cpu().tolist()
            if not token_list:
                continue

            speech_unit_str = " ".join(map(str, token_list))
            speech_unit_for_decode = "".join(
                [f"<|speech_{unit}|>" for unit in speech_unit_str.split(" ")]
            )

            filename = f"rank_{rank}_batch_{batch_idx}_id_{sample_ids[i]}.wav"
            output_wav_path = os.path.join(output_dir, filename)
            condition = args.t2s_condition
            if os.path.exists(output_wav_path):
                audio_array, sampling_rate = sf.read(output_wav_path)
                if hasattr(audio_array, "ndim") and audio_array.ndim > 1:
                    audio_array = audio_array[:, 0]
            else:
                audio_array = vq_model_audio.decode(
                    speech_unit_for_decode,
                    condition=condition,
                    output_wav_file=output_wav_path,
                )

            if hasattr(audio_array, "dtype"):
                audio_array = audio_array.astype(np.float32, copy=False)
            if sampling_rate != whisper_sr:
                audio_array = _resample_audio(audio_array, sampling_rate, whisper_sr)
                sampling_rate = whisper_sr

            whisper_result = whisper_pipe(
                {"array": audio_array, "sampling_rate": sampling_rate},
                generate_kwargs={"language": "english"},
            )
            whisper_text = whisper_result.get("text", "")
            whisper_text_norm = normalize_asr_text(whisper_text)

            local_results.append(
                {
                    "sample_id": sample_ids[i],
                    "gt_text": gt,
                    "whisper_text": whisper_text,
                    "whisper_text_norm": whisper_text_norm,
                }
            )

    return gather_results(local_results, world_size)


def main():
    parser = argparse.ArgumentParser(description="Run LibriSpeech S2T/T2S evaluation.")
    parser.add_argument(
        "--ckpt-path",
        default=None,
        help="Checkpoint path or Hugging Face repo ID. Defaults to model.dynin_omni.repo_id from config.",
    )
    parser.add_argument("--config", required=True, help="Config YAML path.")
    parser.add_argument(
        "--librispeech-root",
        default=None,
        help="Optional local LibriSpeech root with .flac files (used when HF sample has no path).",
    )
    parser.add_argument(
        "--librispeech-split",
        default="test",
        help="LibriSpeech split: test | test.clean | test.other",
    )
    parser.add_argument(
        "--librispeech-from-hf",
        action="store_true",
        help="Load LibriSpeech split directly from HuggingFace datasets.",
    )
    parser.add_argument(
        "--no-librispeech-from-hf",
        dest="librispeech_from_hf",
        action="store_false",
        help="Disable HF loading and read cached local Arrow files only.",
    )
    parser.add_argument(
        "--librispeech-hf-parquet-only",
        action="store_true",
        help="When using --librispeech-from-hf, download only split parquet files via allow_patterns.",
    )
    parser.add_argument(
        "--librispeech-hf-max-files",
        type=int,
        default=0,
        help="When using parquet-only HF mode, limit number of parquet files (0 = all).",
    )
    parser.add_argument(
        "--out-dir", required=True, help="Output directory for T2S audio."
    )
    parser.add_argument(
        "--hf-cache-dir",
        default=None,
        help="HuggingFace cache directory. Defaults to dataset.hf.cache_dir in config.",
    )
    parser.add_argument(
        "--s2t-audio-cache-dir",
        default=None,
        help="Cache directory for decoded S2T audio files (optional).",
    )
    parser.add_argument("--s2t-samples", type=int, default=256)
    parser.add_argument("--t2s-samples", type=int, default=16)
    parser.add_argument("--s2t-batch-size", type=int, default=8)
    parser.add_argument("--t2s-batch-size", type=int, default=8)
    parser.add_argument("--s2t-steps", type=int, default=256)
    parser.add_argument("--s2t-block-length", type=int, default=2)
    parser.add_argument("--s2t-new-tokens", type=int, default=128)
    parser.add_argument("--s2t-remasking", default="low_confidence")
    parser.add_argument("--t2s-token-length", type=int, default=383)
    parser.add_argument("--t2s-steps", type=int, default=383)
    parser.add_argument("--t2s-block-length", type=int, default=128)
    parser.add_argument("--t2s-temperature", type=float, default=1.0)
    parser.add_argument("--t2s-cfg-scale", type=float, default=2.5)
    parser.add_argument(
        "--t2s-condition",
        default="gender-female_emotion-neutral_speed-normal_pitch-normal",
    )
    parser.add_argument("--speech-vocab-size", type=int, default=4096)
    parser.add_argument(
        "--whisper-model",
        default="openai/whisper-large-v3",
        help="Whisper model name for T2S evaluation.",
    )
    parser.add_argument(
        "--shard-id", type=int, default=0, help="Shard index for manual splitting."
    )
    parser.add_argument(
        "--num-shards", type=int, default=1, help="Total shards for manual splitting."
    )
    parser.set_defaults(librispeech_from_hf=True)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    resolved_hf_cache_dir = configure_hf_cache_env(config, project_root=PROJECT_ROOT)
    if args.hf_cache_dir:
        args.hf_cache_dir = str(Path(args.hf_cache_dir).expanduser())
    else:
        args.hf_cache_dir = resolved_hf_cache_dir
    os.environ["DYNIN_OMNI_HF_CACHE_DIR"] = args.hf_cache_dir
    os.environ["HF_HOME"] = args.hf_cache_dir
    os.environ["HF_DATASETS_CACHE"] = args.hf_cache_dir
    os.environ["HF_HUB_CACHE"] = str(Path(args.hf_cache_dir) / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.environ["HF_HUB_CACHE"]
    os.environ.pop("TRANSFORMERS_CACHE", None)

    if not args.librispeech_from_hf:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if not args.s2t_audio_cache_dir:
        args.s2t_audio_cache_dir = str(Path(args.out_dir) / "s2t_audio_cache")

    rank, world_size = setup_distributed()
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    model_local_only = resolve_model_local_files_only(config, default=False)
    model_source = resolve_model_pretrained_source(config)
    ckpt_source = args.ckpt_path or model_source
    tokenizer_source = resolve_tokenizer_source(config, default=ckpt_source)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        padding_side="left",
        trust_remote_code=True,
        local_files_only=model_local_only,
    )

    preproc_config = config.dataset.preprocessing
    max_seq_text = getattr(
        preproc_config, "max_seq_length_text", preproc_config.max_seq_length
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
            "<|i2i|>",
            "<|ti2ti|>",
            "<|v2t|>",
            "<|v2s|>",
            "<|s2t|>",
            "<|t2s|>",
            "<|s2s|>",
            "<|soa|>",
            "<|eoa|>",
        ),
        ignore_id=-100,
        cond_dropout_prob=config.training.cond_dropout_prob,
        use_reserved_token=True,
    )

    vq_model = EMOVASpeechTokenizer.from_pretrained(
        resolve_vq_repo_source(config.model.vq_model_audio),
        local_files_only=model_local_only,
    ).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()

    model_config_path = None
    if os.path.isdir(ckpt_source):
        candidate_paths = [
            os.path.join(ckpt_source, "config.json"),
            os.path.join(os.path.dirname(ckpt_source), "config.json"),
        ]
        for path in candidate_paths:
            if os.path.exists(path):
                model_config_path = path
                break

    model_load_kwargs = {
        "torch_dtype": torch.bfloat16 if device.type == "cuda" else torch.float32,
        "local_files_only": model_local_only,
    }
    if model_config_path:
        model_load_kwargs["config"] = model_config_path
    model = DyninOmniModelLM.from_pretrained(
        ckpt_source,
        **model_load_kwargs,
    ).to(device)

    if world_size > 1:
        if device.type == "cuda":
            model = DDP(model, device_ids=[rank])
        else:
            model = DDP(model)

    core_model = model.module if isinstance(model, DDP) else model

    # Manual sharding is opt-in via --num-shards > 1.
    # When using DDP, DistributedSampler already shards by rank.
    shard_id = args.shard_id
    num_shards = args.num_shards

    s2t_results = run_s2t(
        core_model,
        vq_model,
        tokenizer,
        uni_prompting,
        config,
        args,
        device,
        rank,
        world_size,
        shard_id,
        num_shards,
    )
    metrics = {}
    if rank == 0:
        if s2t_results:
            wer, errors, words = calculate_wer(
                [r["decoded_text"] for r in s2t_results],
                [r["gt_text"] for r in s2t_results],
            )
            print(
                f"[S2T] WER: {wer:.4f} | Word Errors: {errors} | Total Words: {words}"
            )
            metrics.update(
                {
                    "s2t_wer": wer,
                    "s2t_word_errors": errors,
                    "s2t_total_words": words,
                }
            )
            s2t_path = os.path.join(args.out_dir, "s2t_results.jsonl")
            with open(s2t_path, "w", encoding="utf-8") as f:
                for r in s2t_results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        else:
            print("[S2T] No results produced.")

    t2s_results = run_t2s(
        core_model,
        vq_model,
        tokenizer,
        uni_prompting,
        config,
        args,
        device,
        rank,
        world_size,
        shard_id,
        num_shards,
    )
    if rank == 0:
        if t2s_results:
            wer, errors, words = calculate_wer(
                [r.get("whisper_text_norm", "") for r in t2s_results],
                [r["gt_text"] for r in t2s_results],
            )
            print(
                f"[T2S] WER: {wer:.4f} | Word Errors: {errors} | Total Words: {words}"
            )
            metrics.update(
                {
                    "t2s_wer": wer,
                    "t2s_word_errors": errors,
                    "t2s_total_words": words,
                }
            )
            t2s_path = os.path.join(args.out_dir, "t2s_results.jsonl")
            with open(t2s_path, "w", encoding="utf-8") as f:
                for r in t2s_results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        else:
            print("[T2S] No results produced.")
        if metrics:
            metrics_path = os.path.join(args.out_dir, "metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

    cleanup_distributed()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    main()
