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

import bisect
import csv
import logging
import itertools
import json
import math
import os
import hashlib
import contextlib
from pathlib import Path
from accelerate import Accelerator

import os.path as osp
import time
import requests

try:
    from huggingface_hub import snapshot_download
except Exception:  # pragma: no cover - optional dependency in some environments
    snapshot_download = None

import random
import re
import datasets
import pandas as pd
from functools import partial
from typing import List, Optional, Union, Dict, Any, Sequence, Set, Tuple
from glob import glob
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
import torch
import soundfile as sf

from io import BytesIO

Image.warnings.simplefilter("error", Image.DecompressionBombWarning)

import webdataset as wds
from braceexpand import braceexpand
from torch.utils.data import default_collate, Dataset
from torchvision import transforms
from transformers import PreTrainedTokenizer
from datasets import (
    Dataset as HFDataset,
    load_dataset as _hf_datasets_load,
    load_from_disk,
    DatasetDict,
    DownloadConfig,
    get_dataset_config_names,
    concatenate_datasets,
)
import warnings
from training.utils import (
    image_transform as utils_image_transform,
    image_transform_squash as utils_image_transform_squash,
)
from webdataset.tariterators import (
    base_plus_ext,
    tar_file_expander,
    url_opener,
    valid_sample,
)

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_path(*parts: str) -> str:
    return str(PROJECT_ROOT.joinpath(*parts))


def _env_or_project_path(env_var: str, *default_parts: str) -> str:
    return os.getenv(env_var, _project_path(*default_parts))


def _default_hf_cache_dir() -> str:
    return _env_or_project_path("DYNIN_OMNI_HF_CACHE_DIR", "datasets", "huggingface")


def _default_librispeech_cache_dir() -> str:
    return os.getenv("DYNIN_OMNI_LIBRISPEECH_CACHE_DIR", _default_hf_cache_dir())


def _parse_bool_env(value: str) -> Optional[bool]:
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _resolve_local_files_only(local_files_only: Optional[bool] = None) -> bool:
    if local_files_only is not None:
        return bool(local_files_only)
    for env_key in ("DYNIN_OMNI_DATASET_LOCAL_FILES_ONLY", "DYNIN_OMNI_LOCAL_FILES_ONLY"):
        env_value = os.getenv(env_key)
        if env_value is None:
            continue
        parsed = _parse_bool_env(env_value)
        if parsed is not None:
            return parsed
    return False


def _hf_download_config(local_files_only: bool) -> Optional[DownloadConfig]:
    return DownloadConfig(local_files_only=True) if local_files_only else None


def _load_hf_dataset(*args, local_files_only: Optional[bool] = None, **kwargs):
    """Single gateway for HF dataset loading with unified local/offline policy."""
    kwargs.setdefault("cache_dir", _default_hf_cache_dir())
    effective_local_only = _resolve_local_files_only(local_files_only)
    if effective_local_only:
        kwargs["download_config"] = kwargs.get(
            "download_config"
        ) or _hf_download_config(True)
        kwargs.setdefault("local_files_only", True)
    return _hf_datasets_load(*args, **kwargs)


def _resolve_fs_path(
    path: Union[str, Path], *, base_dir: Optional[Union[str, Path]] = None
) -> str:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        anchor = Path(base_dir) if base_dir is not None else PROJECT_ROOT
        resolved = anchor / resolved
    return str(resolved.resolve(strict=False))


def _glob_paths(
    pattern: str, *, base_dir: Optional[Union[str, Path]] = None
) -> list[str]:
    expanded = Path(str(pattern)).expanduser()
    if not expanded.is_absolute():
        anchor = Path(base_dir) if base_dir is not None else PROJECT_ROOT
        expanded = anchor / expanded
    return sorted(glob(str(expanded)))


def _to_plain(value: Any) -> Any:
    """Convert config-like containers (e.g. OmegaConf) to plain Python objects."""
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_plain(v) for v in value)

    # Handle dict-like wrappers (e.g. DictConfig) without importing OmegaConf here.
    if hasattr(value, "items"):
        try:
            return {k: _to_plain(v) for k, v in value.items()}
        except Exception:
            pass

    # Handle list-like wrappers (e.g. ListConfig), excluding strings/bytes.
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
        try:
            return [_to_plain(v) for v in list(value)]
        except Exception:
            pass

    return value


LIBRISPEECH_HF_CONFIG = "clean"
LIBRISPEECH_TRAIN_SPLITS = ("train.100", "train.360")


def _resolve_librispeech_train_split(split: Optional[str]) -> str:
    if split in (None, "", "all"):
        return "+".join(LIBRISPEECH_TRAIN_SPLITS)
    split_text = str(split)
    lowered = split_text.lower()
    if any(token in lowered for token in ("validation", "test", "dev")):
        raise ValueError(
            f"LibriSpeech training split must be train-only, got split='{split_text}'."
        )
    return split_text


def _canonical_speech_dataset_name(name: Optional[str]) -> str:
    if name is None:
        raise ValueError("Speech dataset name cannot be None.")
    key = str(name).strip().lower()
    alias_map = {
        "gigaspeech": "gigaspeech",
        "speechcolab/gigaspeech": "gigaspeech",
        "librispeech": "librispeech",
        "librispeech_asr": "librispeech",
        "openslr/librispeech_asr": "librispeech",
        "commonvoice": "commonvoice",
        "common_voice": "commonvoice",
        "mozilla-foundation/common_voice_22_0": "commonvoice",
        "jsonl": "jsonl",
    }
    return alias_map.get(key, key)


S2T_INSTRUCTION = [
    "Transcribe the given audio.",
    "Write down what you hear in the audio.",
    "Provide a transcript for the given speech.",
    "What does the speaker in the audio say?",
    "Convert the speech in the audio to text.",
    "Listen to the audio and write out the text.",
]

T2S_INSTRUCTION = [
    "Generate speech for the given text.",
    "Read the given sentence aloud.",
    "Say the given words.",
    "Convert the given text into spoken audio.",
    "Speak the given text.",
    "Synthesize the text into speech.",
]

V2T_INSTRUCTION = [
    "Describe the video in detail.",
    "Please provide a detailed description of the video.",
    "What is happening in the video?",
    "Describe the content of the video in detail.",
]

V2S_INSTRUCTION = [
    "Generate speech that describes the given video.",
    "Narrate the events happening in the video.",
    "Produce spoken audio describing the video content.",
    "Convert the video into a detailed spoken narration.",
    "Speak a description of what is shown in the video.",
    "Synthesize speech that explains the content of the video.",
]

person_token = ["a person", "someone", "somebody"]


def replace_person_token(t):
    "Used for CC12M - handles all case variations of <person> tag"
    t = re.sub(
        r"<person>([,\s]*(and)*[,\s]*<person>)+", " people ", t, flags=re.IGNORECASE
    )

    person_pattern = re.compile(r"<person>", re.IGNORECASE)
    while person_pattern.search(t):
        match = person_pattern.search(t)
        t = t[: match.start()] + f" {random.choice(person_token)} " + t[match.end() :]

    return t


def filter_keys(key_set):
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}

    return _f


def group_by_keys_nothrow(
    data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None, src=None
):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        if "fname" not in filesample.keys():
            logger.warning("fname not in filesample keys: %s (src=%s)", filesample, src)
            continue
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()

        if (
            current_sample is None
            or prefix != current_sample["__key__"]
            or suffix in current_sample
        ):
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=wds.warn_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw

    streams = url_opener(src, handler=handler)
    files = tar_file_expander(
        streams, handler=handler
    )  # [{fname,data,__url__}, ...]  __url__ 字段标识当前读取的文件来自哪个 tar 包
    samples = group_by_keys_nothrow(files, handler=handler, src=src)
    return samples


def image_transform(sample, resolution=256):
    image = sample["images"]
    image = transforms.Resize(
        resolution, interpolation=transforms.InterpolationMode.BICUBIC
    )(image)
    image = transforms.CenterCrop((resolution, resolution))(image)
    image = transforms.ToTensor()(image)
    image = transforms.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
    )(image)
    sample["images"] = image
    return sample


def image_transform_squash(sample, resolution=256):
    image = sample["images"]
    image = transforms.Resize(
        (resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC
    )(image)
    image = transforms.ToTensor()(image)
    image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(image)
    sample["images"] = image
    return sample


def conditional_image_transform(sample, resolution=256):
    url = sample.get("__url__", "")
    special_datasets = ["ai2d", "clevr", "docvqa", "geo"]
    use_squash = False
    for keyword in special_datasets:
        if keyword in url:
            use_squash = True
            break
    if use_squash:
        return image_transform_squash(sample, resolution)
    else:
        return image_transform(sample, resolution)


def remove_prefix(caption):
    caption = (
        caption.replace("The image features ", "")
        .replace("The image presents ", "")
        .replace("The image you've sent is, ", "")
        .replace("In the center of the image, ", "")
        .replace("The image showcases ", "")
        .replace("The image is ", "")
        .replace("The image captures ", "")
        .replace("In the given image ", "")
        .replace("The image portrays ", "")
        .replace("In the image, ", "")
        .replace("In this image, we see ", "")
        .replace("The image depicts ", "")
        .replace("This is ", "")
        .replace("In this image, ", "")
        .replace("This image captures ", "")
    )

    return caption


def filter_long_samples(sample):
    return sample.get("input_ids") is not None


class Text2ImageDataset:
    def __init__(
        self,
        train_shards_path_or_url: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int,
        num_train_examples: int,
        per_gpu_batch_size: int,
        global_batch_size: int,
        num_workers: int,
        resolution: int = 256,
        shuffle_buffer_size: int = 1000,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        external_caption_path: Optional[str] = "",
        external_journeydb_caption_path: Optional[str] = "",
        external_laion12m_caption_path: Optional[str] = "",
        external_cc12m_caption_path: Optional[str] = "",
        external_ai2d_caption_path: Optional[str] = "",
        external_clevr_caption_path: Optional[str] = "",
        external_docvqa_caption_path: Optional[str] = "",
        external_geo_caption_path: Optional[str] = "",
        is_captioning: bool = False,
        add_caption_prompt: bool = False,
        long_caption: bool = True,
        shuffle: bool = True,
    ):
        def _normalize_shard_entry(entry: str) -> str:
            text = str(entry)
            if "://" in text or text.startswith("pipe:"):
                return text
            return _resolve_fs_path(text)

        self.long_caption = long_caption
        self.external_caption_path = (
            _resolve_fs_path(external_caption_path) if external_caption_path else ""
        )
        self.external_journeydb_caption_path = (
            _resolve_fs_path(external_journeydb_caption_path)
            if external_journeydb_caption_path
            else ""
        )
        self.external_laion12m_caption_path = (
            _resolve_fs_path(external_laion12m_caption_path)
            if external_laion12m_caption_path
            else ""
        )
        self.external_cc12m_caption_path = (
            _resolve_fs_path(external_cc12m_caption_path)
            if external_cc12m_caption_path
            else ""
        )
        self.is_captioning = is_captioning
        self.add_caption_prompt = add_caption_prompt
        if self.add_caption_prompt:
            questions_path = PROJECT_ROOT / "training" / "questions.json"
            with open(questions_path) as f:
                self.caption_prompt = json.load(f)
                # self.caption_prompt = ['USER: \n' + prompt + ' ASSISTANT:' for prompt in self.caption_prompt]
                self.caption_prompt = [
                    "<|start_header_id|>user<|end_header_id|>\n"
                    + prompt
                    + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
                    for prompt in self.caption_prompt
                ]
        else:
            self.caption_prompt = None

        if self.external_journeydb_caption_path != "":
            with open(self.external_journeydb_caption_path) as file:
                self.journeydb_caption = json.load(file)
        else:
            self.journeydb_caption = None

        if external_ai2d_caption_path != "":
            self.ai2d_caption = pd.read_csv(
                _resolve_fs_path(external_ai2d_caption_path)
            )
        if external_clevr_caption_path != "":
            self.clevr_caption = pd.read_csv(
                _resolve_fs_path(external_clevr_caption_path)
            )
        if external_docvqa_caption_path != "":
            self.docvqa_caption = pd.read_csv(
                _resolve_fs_path(external_docvqa_caption_path)
            )
        if external_geo_caption_path != "":
            self.geo_caption = pd.read_csv(_resolve_fs_path(external_geo_caption_path))

        def tokenize(text):
            if tokenizer is not None:
                text = replace_person_token(text)

                encoding = tokenizer(
                    text,
                    truncation=True,
                    max_length=2 * max_seq_length,
                    padding=False,
                    return_tensors="pt",
                )
                full_input_ids = encoding.input_ids[0]

                if len(full_input_ids) > max_seq_length:
                    return None
                else:
                    return text
            else:
                return text

        if isinstance(train_shards_path_or_url, str):
            train_shards_path_or_url = _normalize_shard_entry(train_shards_path_or_url)
        else:
            train_shards_path_or_url = [
                list(braceexpand(urls)) for urls in train_shards_path_or_url
            ]
            # flatten list using itertools
            train_shards_path_or_url = list(
                itertools.chain.from_iterable(train_shards_path_or_url)
            )
            train_shards_path_or_url = [
                _normalize_shard_entry(url) for url in train_shards_path_or_url
            ]

        if self.external_caption_path != "":
            processing_pipeline = [
                wds.decode("pil", handler=wds.ignore_and_continue),
                wds.map(self.load_external_caption, handler=wds.ignore_and_continue),
                wds.rename(
                    images="jpg;png;jpeg;webp",
                    input_ids="text;txt;caption",
                    handler=wds.warn_and_continue,
                ),
                wds.map(
                    partial(conditional_image_transform, resolution=resolution),
                    handler=wds.warn_and_continue,
                ),
                wds.map(filter_keys(set(["images", "input_ids"]))),
                wds.map_dict(
                    input_ids=tokenize,
                    handler=wds.warn_and_continue,
                ),
                wds.select(filter_long_samples),
            ]
        else:
            processing_pipeline = [
                wds.decode("pil", handler=wds.ignore_and_continue),
                wds.rename(
                    images="jpg;png;jpeg;webp",
                    input_ids="text;txt;caption",
                    handler=wds.warn_and_continue,
                ),
                wds.map(
                    partial(conditional_image_transform, resolution=resolution),
                    handler=wds.warn_and_continue,
                ),
                wds.map(filter_keys(set(["images", "input_ids"]))),
                wds.map_dict(
                    input_ids=tokenize,
                    handler=wds.warn_and_continue,
                ),
                wds.select(filter_long_samples),
            ]

        pipeline = [
            wds.ResampledShards(train_shards_path_or_url),
            tarfile_to_samples_nothrow,
            wds.shuffle(shuffle_buffer_size),
            *processing_pipeline,
            wds.batched(
                per_gpu_batch_size, partial=False, collation_fn=default_collate
            ),
        ]

        num_batches = math.ceil(num_train_examples / global_batch_size)
        num_worker_batches = math.ceil(
            num_train_examples / (global_batch_size * num_workers)
        )  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size

        self._train_dataset = wds.DataPipeline(*pipeline).with_epoch(num_worker_batches)
        self._train_dataloader = wds.WebLoader(
            self._train_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        # add meta-data to dataloader instance for convenience
        self._train_dataloader.num_batches = num_batches
        self._train_dataloader.num_samples = num_samples

    def load_external_caption(self, sample):

        if "SA1B" in sample["__key__"] or "sa" in sample["__key__"]:
            captionf = (
                f"{self.external_caption_path}/{sample['__key__'].split('/')[-1]}.txt"
            )
            if os.path.exists(captionf):
                with open(captionf, "r") as reader:
                    captions = reader.readlines()[0].replace("\n", "")
            else:
                captions = ""

            # for captioning
            if self.is_captioning:
                if self.add_caption_prompt is not None:
                    prompt = random.sample(self.caption_prompt, 1)[0]
                    sample["txt"] = prompt + captions
                else:
                    sample["txt"] = captions
            # for generation
            else:
                # randomly choose short and long captions
                if random.random() < 0.5:
                    sample["txt"] = captions.split(".")[0]
                else:
                    sample["txt"] = captions

                sample["txt"] = remove_prefix(sample["txt"])

            return sample

        elif "laion" in sample["__url__"]:
            url_part = sample["__url__"].split("/")[-1].split(".")[0]
            key = sample["__key__"].split("/")[-1]
            captionf = os.path.join(
                self.external_laion12m_caption_path, url_part, f"{key}.caption"
            )

            if os.path.exists(captionf):
                with open(captionf, "r") as reader:
                    captions = reader.read().strip()
            else:
                captions = ""

            # for captioning
            if self.is_captioning:
                if self.add_caption_prompt is not None:
                    prompt = random.sample(self.caption_prompt, 1)[0]
                    sample["txt"] = prompt + captions
                else:
                    sample["txt"] = captions
            # for generation
            else:
                # randomly choose short and long captions
                if random.random() < 0.5:
                    sample["txt"] = captions.split(".")[0]
                else:
                    sample["txt"] = captions

                sample["txt"] = remove_prefix(sample["txt"])

            return sample

        elif "cc12m" in sample["__url__"]:
            url_part = sample["__url__"].split("/")[-1].split(".")[0]
            key = sample["__key__"].split("/")[-1]
            captionf = os.path.join(
                self.external_cc12m_caption_path, url_part, f"{key}.caption"
            )

            if os.path.exists(captionf):
                with open(captionf, "r") as reader:
                    captions = reader.read().strip()
            else:
                captions = ""

            # for captioning
            if self.is_captioning:
                if self.add_caption_prompt is not None:
                    prompt = random.sample(self.caption_prompt, 1)[0]
                    sample["txt"] = prompt + captions
                else:
                    sample["txt"] = captions
            # for generation
            else:
                # randomly choose short and long captions
                if random.random() < 0.5:
                    sample["txt"] = captions.split(".")[0]
                else:
                    sample["txt"] = captions
                sample["txt"] = remove_prefix(sample["txt"])

            return sample

        elif "ai2d" in sample["__url__"]:
            key = sample["__key__"].split("/")[-1]
            df_row = self.ai2d_caption[
                self.ai2d_caption["image"].astype(str) == key + ".png"
            ]
            if len(df_row) == 0:
                logger.warning("No captions available for key %s", sample["__key__"])
                return sample
            elif len(df_row) > 1:
                df_row = df_row.sample(1)
            question = df_row["question"].values[0]
            solution = df_row["solution"].values[0]
            caption = (
                "<|start_header_id|>user<|end_header_id|>\n"
                "You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here\n"
                f"{question}\n"
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
                f"{solution}"
            )
            sample["txt"] = caption
            return sample

        elif "clevr" in sample["__url__"]:
            key = sample["__key__"].split("/")[-1]
            df_row = self.clevr_caption[
                self.clevr_caption["image"].astype(str) == key + ".jpg"
            ]
            if len(df_row) == 0:
                logger.warning("No captions available for key %s", sample["__key__"])
                return sample
            elif len(df_row) > 1:
                df_row = df_row.sample(1)
            question = df_row["question"].values[0]
            solution = df_row["solution"].values[0]
            caption = (
                "<|start_header_id|>user<|end_header_id|>\n"
                "You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here\n"
                f"{question}\n"
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
                f"{solution}"
            )
            sample["txt"] = caption
            return sample

        elif "docvqa" in sample["__url__"]:
            key = sample["__key__"].split("/")[-1]
            df_row = self.docvqa_caption[
                self.docvqa_caption["image"].astype(str) == key + ".png"
            ]
            if len(df_row) == 0:
                logger.warning("No captions available for key %s", sample["__key__"])
                return sample
            elif len(df_row) > 1:
                df_row = df_row.sample(1)
            question = df_row["question"].values[0]
            solution = df_row["solution"].values[0]
            caption = (
                "<|start_header_id|>user<|end_header_id|>\n"
                "You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here\n"
                f"{question}\n"
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
                f"{solution}"
            )
            sample["txt"] = caption
            return sample

        elif "geo" in sample["__url__"]:
            key = sample["__key__"].split("/")[-1]
            df_row = self.geo_caption[
                self.geo_caption["image"].astype(str) == key + ".jpg"
            ]
            if len(df_row) == 0:
                logger.warning("No captions available for key %s", sample["__key__"])
                return sample
            elif len(df_row) > 1:
                df_row = df_row.sample(1)
            question = df_row["question"].values[0]
            solution = df_row["solution"].values[0]
            caption = (
                "<|start_header_id|>user<|end_header_id|>\n"
                "You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here\n"
                f"{question}\n"
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
                f"{solution}"
            )
            sample["txt"] = caption
            return sample

        elif (
            self.journeydb_caption is not None
            and sample["__key__"] in self.journeydb_caption
        ):
            captions_list = self.journeydb_caption[sample["__key__"]]
            if len(captions_list) == 0:
                logger.warning("No captions available for key %s", sample["__key__"])
                return sample
            sample["txt"] = random.sample(captions_list, 1)[0]
            return sample

        else:
            logger.warning("No caption source matched for sample: %s", sample)
            return sample

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def train_dataloader(self):
        return self._train_dataloader


class SpeechTextDataset(Dataset):
    def __init__(
        self,
        dataset: str,
        subset: str,
        split: Optional[str] = None,
        commonvoice_path: Optional[str] = None,
        local_files_only: Optional[bool] = None,
    ):
        self.dataset_name = _canonical_speech_dataset_name(dataset)
        self.local_files_only = _resolve_local_files_only(local_files_only)

        if self.dataset_name == "gigaspeech":  # subset is either "xs" or "xl"
            self.hgf_dataset: datasets.Dataset = _load_hf_dataset(
                "speechcolab/gigaspeech",
                subset,
                split=split,
                local_files_only=self.local_files_only,
            )

        elif self.dataset_name == "librispeech":
            split = _resolve_librispeech_train_split(split)
            self.hgf_dataset = _load_hf_dataset(
                "openslr/librispeech_asr",
                LIBRISPEECH_HF_CONFIG,
                split=split,
                cache_dir=_default_librispeech_cache_dir(),
                download_mode="reuse_dataset_if_exists",
                local_files_only=self.local_files_only,
            )

        elif self.dataset_name == "commonvoice":
            self.commonvoice_path = (
                commonvoice_path
                or os.getenv("DYNIN_OMNI_COMMONVOICE_PATH")
                or _project_path(
                    "datasets",
                    "audio",
                    "commonvoice",
                    "cv-corpus-22.0-2025-06-20",
                    "en",
                )
            )
            if split is not None:
                warnings.warn(
                    f"Split parameter '{split}' is provided but will not be used for commonvoice dataset."
                )

            self.tsv = pd.read_csv(
                self.commonvoice_path + f"/{subset}.tsv",
                sep="\t",
                usecols=["path", "sentence"],
            )

        else:
            raise ValueError(
                f"Unsupported dataset: {dataset}. "
                "Supported datasets are: gigaspeech/speechcolab/gigaspeech, "
                "librispeech/openslr/librispeech_asr, commonvoice/common_voice."
            )

    def __len__(self):
        if self.dataset_name == "gigaspeech":
            return len(self.hgf_dataset)
        elif self.dataset_name == "librispeech":
            return len(self.hgf_dataset)
        else:  # commonvoice
            return len(self.tsv)

    def __getitem__(self, idx):
        audio_path: str
        text: str

        if self.dataset_name == "gigaspeech":
            sample = self.hgf_dataset[idx]
            audio_path = sample["audio"]["path"]
            text = sample["text"]

        elif self.dataset_name == "librispeech":
            sample = self.hgf_dataset[idx]
            audio_path = sample.get("file") or sample.get("audio", {}).get("path")
            if audio_path and not os.path.isabs(audio_path):
                base = sample.get("file")
                if base:
                    audio_path = os.path.join(os.path.dirname(base), audio_path)
            text = sample.get("text", "")

        else:  # commonvoice
            audio_path = self.commonvoice_path + "/clips/" + self.tsv.iloc[idx]["path"]
            text = self.tsv.iloc[idx]["sentence"]

        return {"audio_path": audio_path, "text": text}


class MixedSpeechTextDataset(Dataset):
    def __init__(
        self,
        dataset_configs: list,
        *,
        preload_audio: bool = False,
        max_audio_duration: Optional[float] = None,
        local_files_only: Optional[bool] = None,
    ):
        """
        Initializes and combines multiple speech datasets.

        Args:
            dataset_configs (list): A list of configuration dictionaries,
                                    where each dict defines a dataset to load.
        """
        self.dataset_metadata = []
        self.dataset_lengths = []
        self._sha1 = hashlib.sha1
        self._missing_root_warned: Set[Path] = set()
        self._token_digest_cache: Dict[Path, Set[str]] = {}
        self.preload_audio = bool(preload_audio)
        self.local_files_only = _resolve_local_files_only(local_files_only)
        if max_audio_duration is not None and max_audio_duration <= 0:
            max_audio_duration = None
        self.max_audio_duration = (
            max_audio_duration if max_audio_duration is not None else None
        )

        # Iterate through the list of dataset configurations from the YAML file
        for config in dataset_configs:
            raw_name = config.get("name") or config.get("dataset_id")
            if raw_name is None:
                raise ValueError(
                    "Each speech dataset config requires 'name' or 'dataset_id'."
                )
            name = _canonical_speech_dataset_name(raw_name)
            subset = config.get("subset", config.get("config"))
            split = config.get("split")
            dataset_local_files_only = _resolve_local_files_only(
                config.get("local_files_only", self.local_files_only)
            )
            use_tokens = bool(config.get("use_precomputed_tokens", False))
            # Allow configs to explicitly require cached tokens; falls back to best-effort when False.
            require_tokens = bool(config.get("require_precomputed_tokens", False))
            token_root = config.get("precomputed_tokens_root")
            token_root_path = Path(token_root).expanduser() if token_root else None

            logger.info(
                "Initializing dataset: %s (source=%s, subset=%s, split=%s)",
                name,
                raw_name,
                subset,
                split,
            )

            # --- Gigaspeech ---
            if name == "gigaspeech":
                hgf_dataset = _load_hf_dataset(
                    "speechcolab/gigaspeech",
                    subset,
                    split=split,
                    local_files_only=dataset_local_files_only,
                )
                if require_tokens and "audio" in hgf_dataset.features:
                    try:
                        hgf_dataset = hgf_dataset.cast_column(
                            "audio", datasets.Audio(decode=False)
                        )
                    except Exception:
                        pass
                self.dataset_metadata.append(
                    {
                        "name": name,
                        "source_name": str(raw_name),
                        "data": hgf_dataset,
                        "use_precomputed_tokens": use_tokens
                        and token_root_path is not None,
                        "precomputed_tokens_root": token_root_path,
                        "require_precomputed_tokens": require_tokens
                        and use_tokens
                        and token_root_path is not None,
                        "_missing_token_warned": set(),
                        "subset": subset,
                        "split": split,
                    }
                )
                metadata = self.dataset_metadata[-1]
                metadata["raw_length"] = len(hgf_dataset)
                if metadata["require_precomputed_tokens"]:
                    cached_indices = self._load_or_create_cached_indices(metadata)
                    metadata["cached_indices"] = cached_indices
                    self.dataset_lengths.append(len(cached_indices))
                else:
                    metadata["cached_indices"] = None
                    self.dataset_lengths.append(len(hgf_dataset))

            # --- LibriSpeech ---
            elif name == "librispeech":
                split = _resolve_librispeech_train_split(split)
                hgf_dataset = _load_hf_dataset(
                    "openslr/librispeech_asr",
                    LIBRISPEECH_HF_CONFIG,
                    split=split,
                    cache_dir=_default_librispeech_cache_dir(),
                    download_mode="reuse_dataset_if_exists",
                    local_files_only=dataset_local_files_only,
                )

                metadata = {
                    "name": name,
                    "source_name": str(raw_name),
                    "data": hgf_dataset,
                    "use_precomputed_tokens": use_tokens
                    and token_root_path is not None,
                    "precomputed_tokens_root": token_root_path,
                    "require_precomputed_tokens": require_tokens
                    and use_tokens
                    and token_root_path is not None,
                    "_missing_token_warned": set(),
                    "subset": subset,
                    "split": split,
                }
                self.dataset_metadata.append(metadata)
                metadata["raw_length"] = len(hgf_dataset)
                if metadata["require_precomputed_tokens"]:
                    cached_indices = self._load_or_create_cached_indices(metadata)
                    metadata["cached_indices"] = cached_indices
                    self.dataset_lengths.append(len(cached_indices))
                else:
                    metadata["cached_indices"] = None
                    self.dataset_lengths.append(len(hgf_dataset))

            # --- Common Voice ---
            elif name == "commonvoice":
                commonvoice_path = (
                    config.get("commonvoice_path")
                    or config.get("data_root")
                    or os.getenv("DYNIN_OMNI_COMMONVOICE_PATH")
                    or _project_path(
                        "datasets",
                        "audio",
                        "commonvoice",
                        "cv-corpus-22.0-2025-06-20",
                        "en",
                    )
                )
                if split is not None:
                    warnings.warn(
                        f"Split parameter '{split}' is provided but will not be used for Common Voice."
                    )

                tsv_path = f"{commonvoice_path}/{subset}.tsv"
                tsv = pd.read_csv(tsv_path, sep="\t", usecols=["path", "sentence"])

                metadata = {
                    "name": name,
                    "source_name": str(raw_name),
                    "data_root": f"{commonvoice_path}/clips/",
                    "tsv": tsv,
                    "use_precomputed_tokens": use_tokens
                    and token_root_path is not None,
                    "precomputed_tokens_root": token_root_path,
                    "require_precomputed_tokens": require_tokens
                    and use_tokens
                    and token_root_path is not None,
                    "_missing_token_warned": set(),
                    "subset": subset,
                    "split": split,
                }
                self.dataset_metadata.append(metadata)
                metadata["raw_length"] = len(tsv)
                metadata["cached_indices"] = None
                self.dataset_lengths.append(len(tsv))

            # --- JSONL (local text <-> speech pairs) ---
            elif name == "jsonl":
                jsonl_path = config.get("jsonl_path")
                if not jsonl_path:
                    raise ValueError("jsonl dataset requires jsonl_path")
                jsonl_path = _resolve_fs_path(str(jsonl_path))
                text_key = config.get("text_key", "text")
                audio_key = config.get("audio_key", "speech")
                audio_root = config.get("audio_root") or config.get("wav_root")
                resolved_root = (
                    Path(_resolve_fs_path(str(audio_root))) if audio_root else None
                )

                records: list[dict] = []
                with Path(jsonl_path).open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            raw = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        text = str(raw.get(text_key, "")).strip()
                        audio_path = (
                            raw.get(audio_key)
                            or raw.get("audio_path")
                            or raw.get("path")
                        )
                        if not audio_path:
                            continue
                        if not os.path.isabs(audio_path) and resolved_root is not None:
                            audio_path = str(resolved_root / audio_path)
                        records.append({"audio_path": audio_path, "text": text})

                metadata = {
                    "name": name,
                    "source_name": str(raw_name),
                    "data": records,
                    "use_precomputed_tokens": use_tokens
                    and token_root_path is not None,
                    "precomputed_tokens_root": token_root_path,
                    "require_precomputed_tokens": require_tokens
                    and use_tokens
                    and token_root_path is not None,
                    "_missing_token_warned": set(),
                    "subset": subset,
                    "split": split,
                }
                self.dataset_metadata.append(metadata)
                metadata["raw_length"] = len(records)
                if metadata["require_precomputed_tokens"]:
                    cached_indices = self._load_or_create_cached_indices(metadata)
                    metadata["cached_indices"] = cached_indices
                    self.dataset_lengths.append(len(cached_indices))
                else:
                    metadata["cached_indices"] = None
                    self.dataset_lengths.append(len(records))

            else:
                raise ValueError(f"Unsupported dataset: {name}.")

        # Calculate cumulative lengths to map a global index to a specific dataset
        self.cumulative_lengths = list(itertools.accumulate(self.dataset_lengths))

    def _sanitize_for_filename(self, value: Optional[str]) -> str:
        if not value:
            return "none"
        return "".join(ch if ch.isalnum() or ch in "-._" else "-" for ch in str(value))

    def _manifest_path(self, metadata: dict) -> Path:
        root: Path = metadata["precomputed_tokens_root"]
        manifest_dir = root / "_manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        subset_part = self._sanitize_for_filename(metadata.get("subset"))
        split_part = self._sanitize_for_filename(metadata.get("split"))
        filename = f"{metadata['name']}_{subset_part}_{split_part}.idx"
        return manifest_dir / filename

    def _collect_token_digests(self, root: Path) -> Set[str]:
        cached = self._token_digest_cache.get(root)
        if cached is not None:
            return cached
        digests: Set[str] = set()
        if not root.exists():
            self._token_digest_cache[root] = digests
            return digests
        with contextlib.suppress(Exception):
            for token_file in root.rglob("*.pt"):
                candidate = token_file.stem
                if len(candidate) == 40:
                    digests.add(candidate)
        self._token_digest_cache[root] = digests
        logger.info("Indexed %d cached token files under %s", len(digests), root)
        return digests

    def _load_or_create_cached_indices(self, metadata: dict) -> List[int]:
        manifest_path = self._manifest_path(metadata)
        if manifest_path.exists():
            try:
                with manifest_path.open("r") as fh:
                    indices = [int(line.strip()) for line in fh if line.strip()]
                logger.info(
                    "Loaded %d cached indices for %s from %s",
                    len(indices),
                    metadata["name"],
                    manifest_path,
                )
                return indices
            except Exception as exc:
                logger.warning(
                    "Failed to read cached index manifest %s: %s", manifest_path, exc
                )

        root: Path = metadata["precomputed_tokens_root"]
        digest_source = self._collect_token_digests(root)
        if not digest_source:
            logger.warning(
                "No precomputed token files found under %s for dataset %s; cached subset will be empty.",
                root,
                metadata["name"],
            )
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.touch(exist_ok=True)
            return []

        dataset = metadata["data"]
        total_len = len(dataset)
        remaining_digests: Set[str] = set(digest_source)
        indices: List[int] = []
        desc = f"Filtering cached {metadata['name']}"
        disable_progress = not logger.isEnabledFor(logging.INFO)
        expected_matches = len(digest_source)
        for idx in tqdm(
            range(total_len), desc=desc, unit="sample", disable=disable_progress
        ):
            try:
                sample = dataset[idx]
            except Exception as exc:
                logger.warning(
                    "Failed to fetch sample %d while building cached index: %s",
                    idx,
                    exc,
                )
                continue
            audio_info = sample.get("audio")
            audio_path = None
            if metadata.get("name") == "librispeech":
                audio_path = sample.get("file") or (
                    audio_info.get("path") if isinstance(audio_info, dict) else None
                )
                if audio_path and not os.path.isabs(audio_path):
                    base = sample.get("file")
                    if base:
                        audio_path = os.path.join(os.path.dirname(base), audio_path)
            else:
                if isinstance(audio_info, dict):
                    audio_path = audio_info.get("path")
                elif isinstance(sample, dict):
                    audio_path = sample.get("audio_path") or sample.get("path")
            if not audio_path:
                continue
            digest = self._sha1(
                _resolve_fs_path(audio_path).encode("utf-8")
            ).hexdigest()
            if digest in remaining_digests:
                indices.append(idx)
                if expected_matches:
                    remaining_digests.discard(digest)
                    if not remaining_digests:
                        break
                if expected_matches and len(indices) >= expected_matches:
                    break

        try:
            with manifest_path.open("w") as fh:
                for idx in indices:
                    fh.write(f"{idx}\n")
            logger.info(
                "Cached index for %s contains %d / %d samples (manifest: %s)",
                metadata["name"],
                len(indices),
                total_len,
                manifest_path,
            )
        except Exception as exc:
            logger.warning(
                "Failed to write cached index manifest %s: %s", manifest_path, exc
            )

        return indices

    def __len__(self):
        """Returns the total number of samples across all datasets."""
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def _resolve_audio_text(
        self, metadata: dict, dataset_idx: int, resolved_idx: int
    ) -> tuple[str, str]:
        dataset_name = metadata["name"]

        if dataset_name == "gigaspeech":
            sample = metadata["data"][resolved_idx]
            audio_path = sample["audio"]["path"]
            text = sample["text"]
            text = (
                text.replace(" <COMMA>", ",")
                .replace(" <PERIOD>", ".")
                .replace(" <QUESTIONMARK>", "?")
                .replace(" <EXCLAMATIONMARK>", "!")
            )
            return audio_path, text

        if dataset_name == "librispeech":
            sample = metadata["data"][resolved_idx]
            audio_path = sample.get("file") or sample.get("audio", {}).get("path")
            if audio_path and not os.path.isabs(audio_path):
                base = sample.get("file")
                if base:
                    audio_path = os.path.join(os.path.dirname(base), audio_path)
            text = sample.get("text", "")
            return audio_path, text

        if dataset_name == "commonvoice":
            row = metadata["tsv"].iloc[resolved_idx]
            audio_path = metadata["data_root"] + row["path"]
            text = row["sentence"].upper()
            return audio_path, text

        if dataset_name == "jsonl":
            sample = metadata["data"][resolved_idx]
            audio_path = sample.get("audio_path") or sample.get("path")
            text = sample.get("text", "")
            return audio_path, text

        raise ValueError(f"Unsupported dataset for resolution: {dataset_name}")

    def _load_audio_waveform(self, audio_path: str) -> Optional[Tuple[np.ndarray, int]]:
        if not self.preload_audio:
            return None
        try:
            with sf.SoundFile(audio_path, mode="r") as handle:
                samplerate = handle.samplerate
                total_frames = len(handle)
                target_frames = total_frames
                if self.max_audio_duration is not None and samplerate > 0:
                    max_frames = int(self.max_audio_duration * samplerate)
                    target_frames = min(total_frames, max_frames)
                handle.seek(0)
                audio = handle.read(
                    frames=target_frames, dtype="float32", always_2d=False
                )
        except Exception as exc:
            logger.warning("Failed to preload audio '%s': %s", audio_path, exc)
            return None
        if audio is None:
            return None
        if isinstance(audio, np.ndarray) and audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        return audio, samplerate

    def __getitem__(self, idx):
        """
        Fetches a sample from the combined dataset.

        It first determines which dataset the global index `idx` belongs to,
        calculates the local index within that dataset, and then retrieves the item.
        """
        if idx >= self.__len__():
            raise IndexError(
                f"Index {idx} is out of bounds for the combined dataset with length {self.__len__()}."
            )

        # Find which dataset the index belongs to
        dataset_idx = bisect.bisect_right(self.cumulative_lengths, idx)

        # Calculate the local index within that dataset
        local_idx = (
            idx - self.cumulative_lengths[dataset_idx - 1] if dataset_idx > 0 else idx
        )

        metadata = self.dataset_metadata[dataset_idx]
        dataset_name = metadata["name"]
        dataset_length = self.dataset_lengths[dataset_idx]
        if dataset_length <= 0:
            raise RuntimeError(
                f"No cached samples available for dataset '{dataset_name}'."
            )

        audio_path: str
        text: str
        audio_tokens: Optional[torch.Tensor]

        max_retry = 5
        retry = 0

        while retry < max_retry:
            try:
                audio_tokens = None
                cached_indices: Optional[List[int]] = metadata.get("cached_indices")
                if cached_indices is not None:
                    if not cached_indices:
                        raise RuntimeError(
                            f"No cached samples available for dataset '{dataset_name}'."
                        )
                    if local_idx >= len(cached_indices):
                        raise IndexError(
                            f"Cached index {local_idx} out of range for dataset '{dataset_name}'."
                        )
                    resolved_idx = cached_indices[local_idx]
                else:
                    resolved_idx = local_idx

                audio_path, text = self._resolve_audio_text(
                    metadata, dataset_idx, resolved_idx
                )

                audio_tokens = self._maybe_load_precomputed_tokens(audio_path, metadata)
                if metadata.get("require_precomputed_tokens") and audio_tokens is None:
                    if dataset_name == "gigaspeech":
                        retry += 1
                        if retry < max_retry:
                            local_idx = random.randint(0, dataset_length - 1)
                            continue
                        fallback = self._fallback_sample(dataset_idx)
                        if fallback is not None:
                            return fallback
                    raise FileNotFoundError(
                        f"Required precomputed tokens missing for {audio_path}"
                    )
                if dataset_name == "gigaspeech" and audio_tokens is None:
                    retry += 1
                    if retry < max_retry:
                        local_idx = random.randint(0, dataset_length - 1)
                        continue
                    fallback = self._fallback_sample(dataset_idx)
                    if fallback is not None:
                        return fallback
                    raise FileNotFoundError(
                        f"GigaSpeech tokens unavailable for {audio_path}"
                    )
                audio_waveform = None
                audio_sample_rate = None
                if audio_tokens is None:
                    loaded = self._load_audio_waveform(audio_path)
                    if loaded is not None:
                        audio_waveform, audio_sample_rate = loaded

                return {
                    "audio_path": audio_path,
                    "text": text,
                    "audio_tokens": audio_tokens,
                    "audio_waveform": audio_waveform,
                    "audio_sample_rate": audio_sample_rate,
                }

            except Exception as exc:
                logger.warning(
                    "[MixedSpeechTextDataset] Failed to load sample from '%s' at local index %s: %r",
                    dataset_name,
                    local_idx,
                    exc,
                )
                retry += 1
                if retry >= max_retry:
                    break
                local_idx = random.randint(0, dataset_length - 1)
                continue

        if dataset_name == "gigaspeech":
            fallback = self._fallback_sample(dataset_idx)
            if fallback is not None:
                return fallback

        raise RuntimeError(
            f"Unable to fetch a valid sample from dataset '{dataset_name}' after {max_retry} retries."
        )

    def _fallback_sample(self, exclude_idx: int) -> Optional[dict]:
        candidate_indices = [
            idx
            for idx, meta in enumerate(self.dataset_metadata)
            if idx != exclude_idx
            and meta.get("name") != "gigaspeech"
            and self.dataset_lengths[idx] > 0
        ]
        if not candidate_indices:
            return None
        random.shuffle(candidate_indices)
        for alt_idx in candidate_indices:
            alt_len = self.dataset_lengths[alt_idx]
            attempts = min(alt_len, 5)
            for _ in range(attempts):
                local_idx = random.randint(0, alt_len - 1)
                try:
                    return self._fetch_fallback(alt_idx, local_idx)
                except RuntimeError:
                    continue
        return None

    def _fetch_fallback(self, dataset_idx: int, local_idx: int) -> dict:
        metadata = self.dataset_metadata[dataset_idx]
        dataset_length = self.dataset_lengths[dataset_idx]
        if dataset_length <= 0:
            raise RuntimeError("Fallback dataset empty")

        cached_indices: Optional[List[int]] = metadata.get("cached_indices")
        if cached_indices is not None:
            if not cached_indices:
                raise RuntimeError("Fallback cache empty")
            if local_idx >= len(cached_indices):
                raise IndexError("Fallback cached index out of range")
            resolved_idx = cached_indices[local_idx]
        else:
            resolved_idx = local_idx

        audio_path, text = self._resolve_audio_text(metadata, dataset_idx, resolved_idx)
        audio_tokens = self._maybe_load_precomputed_tokens(audio_path, metadata)
        if metadata.get("require_precomputed_tokens") and audio_tokens is None:
            raise RuntimeError("Fallback tokens missing")
        audio_waveform = None
        audio_sample_rate = None
        if audio_tokens is None:
            loaded = self._load_audio_waveform(audio_path)
            if loaded is not None:
                audio_waveform, audio_sample_rate = loaded
        return {
            "audio_path": audio_path,
            "text": text,
            "audio_tokens": audio_tokens,
            "audio_waveform": audio_waveform,
            "audio_sample_rate": audio_sample_rate,
        }

    def _maybe_load_precomputed_tokens(
        self, audio_path: str, metadata: dict
    ) -> Optional[torch.Tensor]:
        if not metadata.get("use_precomputed_tokens"):
            return None
        root: Optional[Path] = metadata.get("precomputed_tokens_root")
        if root is None:
            return None
        if not root.exists():
            if metadata.get("require_precomputed_tokens"):
                if root not in self._missing_root_warned:
                    logger.warning("Precomputed token root missing: %s", root)
                    self._missing_root_warned.add(root)
                raise FileNotFoundError(
                    f"Required precomputed token root not found: {root}"
                )
            return None
        key = _resolve_fs_path(audio_path)
        digest = self._sha1(key.encode("utf-8")).hexdigest()
        token_path = root / digest[:2] / digest[2:4] / f"{digest}.pt"
        if not token_path.exists():
            if metadata.get("require_precomputed_tokens"):
                logger.warning("Precomputed audio tokens not found: %s", token_path)
            return None
        try:
            tokens = torch.load(token_path, map_location="cpu")
            if isinstance(tokens, torch.Tensor):
                return tokens.clone()
            if isinstance(tokens, (list, tuple)):
                return torch.tensor(tokens, dtype=torch.long)
            logger.warning(
                "Unexpected token format in %s (type=%s)", token_path, type(tokens)
            )
        except Exception as exc:
            logger.warning("Failed to load precomputed tokens %s: %s", token_path, exc)
        return None


class VideoCaptionDataset(Dataset):
    def __init__(
        self,
        transform,
        tokenizer,
        max_seq_length: int,
        resolution: int = 256,
        openvid1m_path: Optional[str] = None,
        webvid10m_path: Optional[str] = None,
        llavavid_path: Optional[str] = None,
        dataset_name="openvid1m",
        llavavid_local_files_only: bool = False,
        local_files_only: Optional[bool] = None,
        llavavid_skip_configs: Optional[Sequence[str]] = None,
        llavavid_skip_video_patterns: Optional[Sequence[str]] = None,
        llavavid_max_samples: Optional[int] = None,
        llavavid_sample_seed: int = 42,
        sample_method="uniform",
        num_frames: int = 8,
        vq_model=None,
        max_video_seconds: Optional[float] = None,
    ):
        openvid1m_path = openvid1m_path or _env_or_project_path(
            "DYNIN_OMNI_OPENVID1M_PATH",
            "datasets",
            "video",
            "openvid1m",
            "video",
        )
        webvid10m_path = webvid10m_path or _env_or_project_path(
            "DYNIN_OMNI_WEBVID10M_PATH",
            "datasets",
            "video",
            "webvid10m",
        )
        llavavid_path = llavavid_path or _env_or_project_path(
            "DYNIN_OMNI_LLAVAVID_PATH",
            "datasets",
            "video",
            "LLaVA-Video-178K",
        )

        available_datasets = ["openvid1m", "webvid10m", "llavavid"]
        if dataset_name not in available_datasets:
            raise ValueError(
                f"Invalid dataset name: {dataset_name}. Available datasets: {available_datasets}"
            )

        self.max_seq_length = max_seq_length
        self.transform = transform
        self.vq_model = vq_model
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.sample_method = sample_method
        self.dataset_name = dataset_name
        self.num_frames = num_frames
        self.local_files_only = _resolve_local_files_only(local_files_only)
        self.llavavid_local_files_only = self.local_files_only or bool(
            llavavid_local_files_only
        )
        self.llavavid_skip_configs = set(llavavid_skip_configs or [])
        self.llavavid_skip_video_patterns = tuple(llavavid_skip_video_patterns or [])
        self.llavavid_max_samples = (
            int(llavavid_max_samples) if llavavid_max_samples else None
        )
        self.llavavid_sample_seed = int(llavavid_sample_seed)
        self.max_video_seconds = (
            float(max_video_seconds)
            if max_video_seconds and max_video_seconds > 0
            else None
        )
        self.caption_prompt = V2T_INSTRUCTION
        self.caption_prompt = [
            "<|start_header_id|>user<|end_header_id|>\n"
            + prompt
            + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            for prompt in self.caption_prompt
        ]

        self.webvid10m_path = webvid10m_path

        if dataset_name == "webvid10m":
            self.vid_data = self._collect_webvid10m(webvid10m_path)
            self.dataset_root = webvid10m_path
        elif dataset_name == "openvid1m":
            self.vid_data = self._collect_openvid1m(openvid1m_path)
            self.dataset_root = openvid1m_path
        elif dataset_name == "llavavid":
            self.vid_data = self._collect_llavavid(llavavid_path)
            self.dataset_root = Path(llavavid_path)
            self.llavavid_video_root = Path(llavavid_path)

        else:
            raise ValueError(
                f"Invalid dataset name: {dataset_name}. Available datasets: openvid1m, webvid10m, llavavid"
            )

    def _get_caption_prompt(self):
        """
        Get a random caption prompt from the list of caption prompts.
        """
        return np.random.choice(self.caption_prompt)

    def _tokenize(self, text):
        if self.tokenizer is not None:
            input_ids = self.tokenizer(
                text,
                truncation=True,
                max_length=2 * self.max_seq_length,
                padding=False,
                return_tensors="pt",
            )[0]

            if len(input_ids) > self.max_seq_length:
                return None
            else:
                return input_ids
        else:
            raise ValueError("Tokenizer is not provided.")

    def _collect_webvid10m(self, root_path):
        logger.info("Loading videos from WebVid10m dataset...")
        resolved_root = _resolve_fs_path(str(root_path))

        def _find_csv(base_dir: str) -> Optional[str]:
            preferred = osp.join(base_dir, "webvid-10M-train.csv")
            if osp.isfile(preferred):
                return preferred
            candidates = sorted(glob(osp.join(base_dir, "*train*.csv")))
            if candidates:
                return candidates[0]
            fallback = sorted(glob(osp.join(base_dir, "*.csv")))
            if fallback:
                return fallback[0]
            return None

        csv_path = _find_csv(resolved_root) if osp.isdir(resolved_root) else None
        if csv_path is None:
            raw_root = str(root_path).strip()

            def _looks_like_hf_repo_id(value: str) -> bool:
                if not value or osp.isabs(value):
                    return False
                if value.startswith((".", "~")):
                    return False
                parts = [part for part in value.split("/") if part]
                if len(parts) != 2:
                    return False
                if any(part in {".", ".."} for part in parts):
                    return False
                return True

            repo_id = (
                raw_root if _looks_like_hf_repo_id(raw_root) else "TempoFunk/webvid-10M"
            )
            if snapshot_download is None:
                raise RuntimeError(
                    "huggingface_hub is not available, but WebVid metadata is not found locally."
                )
            snapshot_dir = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                cache_dir=_default_hf_cache_dir(),
                allow_patterns=["*.csv", "*.json", "*.jsonl", "README*"],
                local_files_only=self.local_files_only,
            )
            resolved_root = snapshot_dir
            csv_path = _find_csv(resolved_root)

        if csv_path is None:
            raise FileNotFoundError(
                f"Could not locate WebVid metadata CSV under '{resolved_root}'."
            )

        webvid_pd = pd.read_csv(csv_path)
        self.dataset_length = len(webvid_pd)
        self.webvid10m_path = resolved_root
        logger.info("%d videos loaded from WebVid10m.", len(webvid_pd))

        return webvid_pd

    def _collect_openvid1m(self, root_path):
        csv_path = osp.join(root_path, "OpenVid-1M.csv")
        openvid_pd = pd.read_csv(csv_path)
        self.dataset_length = len(openvid_pd)
        logger.info("%d videos loaded from OpenVid-1M.", len(openvid_pd))

        return openvid_pd

    def _collect_llavavid(
        self,
        root_path="lmms-lab/LLaVA-Video-178K",
        cache_dir: Optional[str] = None,
    ):
        """
        Collect all available (and locally cached) subsets of the LLaVA-Video-178K dataset.
        Handles both on-disk exports (each config stored as subfolders of splits) and remote configs.
        Returns a single flattened HuggingFace Dataset that concatenates every successfully loaded config.
        """
        if cache_dir is None:
            cache_dir = _default_hf_cache_dir()

        DATASET_NAME = root_path

        local_root = Path(DATASET_NAME)
        configs: list[str]
        using_local_dirs = local_root.exists()

        # Fast path: if root is a saved HF dataset (dataset_info.json present), load directly
        if using_local_dirs and (local_root / "dataset_info.json").exists():
            try:
                ds = load_from_disk(str(local_root))
                count = len(ds)
                ds = _add_config_column(ds, "llavavid", count)
                return ds, count
            except Exception:
                pass

        configs = []
        if using_local_dirs:
            for p in sorted(local_root.iterdir()):
                if not p.is_dir():
                    continue
                if p.name.startswith("."):
                    continue
                split_exists = any(
                    (p / split_name).exists()
                    for split_name in ("open_ended", "caption", "multi_choice")
                )
                raw_json_exists = bool(list(p.glob("*_processed.json")))
                if not split_exists and not raw_json_exists:
                    continue
                configs.append(p.name)
            if not configs:
                using_local_dirs = False

        if not configs:
            try:
                configs = get_dataset_config_names(DATASET_NAME)
                using_local_dirs = False
            except Exception as e:
                raise RuntimeError(f"Failed to fetch configs for {DATASET_NAME}: {e}")

        skip_configs = getattr(self, "llavavid_skip_configs", set())
        if skip_configs:
            existing = [cfg for cfg in configs if cfg in skip_configs]
            if existing:
                logger.info("LLaVA-Vid: skipping configs %s", existing)
            configs = [cfg for cfg in configs if cfg not in skip_configs]

        if not configs:
            raise RuntimeError(
                "All LLaVA-Video configs were skipped; nothing left to load."
            )

        def _add_config_column(dataset: HFDataset, cfg_name: str, row_count: int):
            """Attach the originating config name so downstream can locate videos."""
            if dataset is None or not cfg_name:
                return dataset
            if "llavavid_config" in dataset.column_names:
                return dataset
            return dataset.add_column("llavavid_config", [cfg_name] * row_count)

        def _flatten_dataset(ds_obj, label: str, cfg_name: str):
            """Convert DatasetDicts into a single Dataset and report the row count."""
            if ds_obj is None:
                return None, 0
            if isinstance(ds_obj, DatasetDict):
                splits = [split for split in ds_obj.values()]
                if not splits:
                    logger.warning("Skipping %s: dataset dict has no splits.", label)
                    return None, 0
                total_rows = sum(len(split) for split in splits)
                if len(splits) == 1:
                    return splits[0], total_rows
                try:
                    merged = concatenate_datasets(splits)
                except Exception as merge_err:
                    logger.warning(
                        "Skipping %s: failed to concatenate splits: %s",
                        label,
                        merge_err,
                    )
                    return None, 0
                dataset = merged
            else:
                dataset = ds_obj
                try:
                    total_rows = len(dataset)
                except Exception as len_err:
                    logger.warning(
                        "Skipping %s: unable to compute dataset length (%s).",
                        label,
                        len_err,
                    )
                    return None, 0

            dataset = _add_config_column(dataset, cfg_name, total_rows)
            return dataset, total_rows

        def _load_local_config(cfg_name: str):
            """Attempt to read a single config from disk, handling split sub-directories if needed."""
            cfg_root = local_root / cfg_name
            if not cfg_root.exists():
                return None, 0

            # First try loading the directory directly (Dataset or DatasetDict exports).
            try:
                ds_direct = load_from_disk(str(cfg_root))
            except Exception as direct_err:
                logger.warning(
                    "Failed to load config %s via load_from_disk: %s",
                    cfg_name,
                    direct_err,
                )
            else:
                ds_flat, ds_count = _flatten_dataset(ds_direct, cfg_name, cfg_name)
                if ds_flat is not None and ds_count > 0:
                    return ds_flat, ds_count

            # Fallback: iterate over split sub-directories (caption/open_ended/multi_choice, etc.).
            split_dirs = [p for p in sorted(cfg_root.iterdir()) if p.is_dir()]
            if not split_dirs:
                split_dirs = []

            split_datasets = []
            for split_dir in split_dirs:
                try:
                    split_ds = load_from_disk(str(split_dir))
                except Exception as split_err:
                    logger.warning(
                        "Skipping %s/%s: %s", cfg_name, split_dir.name, split_err
                    )
                    continue
                split_datasets.append(split_ds)

            if not split_datasets:
                # Final fallback: load raw *_processed.json files if present.
                json_files = sorted(cfg_root.glob("*_processed.json"))
                for json_path in json_files:
                    try:
                        split_ds = _load_hf_dataset(
                            "json",
                            data_files=str(json_path),
                            split="train",
                            local_files_only=self.llavavid_local_files_only,
                        )
                    except Exception as json_err:
                        logger.warning(
                            "Skipping %s/%s: %s", cfg_name, json_path.name, json_err
                        )
                        continue
                    split_datasets.append(split_ds)

            if not split_datasets:
                return None, 0

            split_total = sum(len(split_ds) for split_ds in split_datasets)
            if len(split_datasets) == 1:
                dataset = split_datasets[0]
            else:
                try:
                    dataset = concatenate_datasets(split_datasets)
                except Exception as merge_err:
                    logger.warning(
                        "Skipping %s: failed to concatenate split datasets: %s",
                        cfg_name,
                        merge_err,
                    )
                    return None, 0

            dataset = _add_config_column(dataset, cfg_name, split_total)
            return dataset, split_total

        datasets_loaded = []
        total_count = 0

        for cfg in configs:
            ds = None
            cfg_count = 0

            if using_local_dirs:
                ds, cfg_count = _load_local_config(cfg)

            if ds is None or cfg_count == 0:
                download_cfg = None
                if self.llavavid_local_files_only:
                    download_cfg = DownloadConfig(local_files_only=True)
                try:
                    remote_ds = _load_hf_dataset(
                        DATASET_NAME,
                        name=cfg,
                        cache_dir=cache_dir,
                        verification_mode="no_checks",
                        download_config=download_cfg,
                        local_files_only=self.llavavid_local_files_only,
                    )
                except Exception as remote_err:
                    logger.warning("Skipping %s: %s", cfg, remote_err)
                    continue

                ds, cfg_count = _flatten_dataset(remote_ds, cfg, cfg)
                if ds is None or cfg_count == 0:
                    logger.warning("Skipping %s: dataset empty after flattening.", cfg)
                    continue

            datasets_loaded.append(ds)
            total_count += cfg_count

        if not datasets_loaded:
            raise RuntimeError("No valid configs could be loaded!")

        if len(datasets_loaded) == 1:
            global_dataset = datasets_loaded[0]
        else:
            try:
                global_dataset = concatenate_datasets(datasets_loaded)
            except Exception as merge_err:
                logger.warning(
                    "Failed to concatenate configs in one step: %s. Trying pairwise concatenation.",
                    merge_err,
                )
                try:
                    combined = datasets_loaded[0]
                    for ds_next in datasets_loaded[1:]:
                        combined = concatenate_datasets([combined, ds_next])
                    global_dataset = combined
                except Exception as pair_err:
                    raise RuntimeError(
                        f"Unable to merge LLaVA-Video configs: {pair_err}"
                    ) from pair_err

        # Filter out samples whose video path matches known-bad patterns (e.g., missing shareVideoGPTV frames)
        skip_patterns = getattr(self, "llavavid_skip_video_patterns", tuple())
        if skip_patterns:

            def _matches_skip(entry: dict[str, Any]) -> bool:
                video_entry = entry.get("video")
                if not isinstance(video_entry, str):
                    return False
                return any(pattern in video_entry for pattern in skip_patterns)

            def _filter_dataset(ds_obj):
                if isinstance(ds_obj, list):
                    filtered_list = []
                    removed_total = 0
                    for item in ds_obj:
                        filtered_item, removed_item = _filter_dataset(item)
                        removed_total += removed_item
                        if filtered_item is None:
                            continue
                        filtered_list.append(filtered_item)
                    return filtered_list, removed_total
                elif isinstance(ds_obj, HFDataset):
                    before = len(ds_obj)
                    filtered = ds_obj.filter(lambda ex: not _matches_skip(ex))
                    removed = before - len(filtered)
                    return filtered, removed
                elif isinstance(ds_obj, dict):
                    return (None, 1) if _matches_skip(ds_obj) else (ds_obj, 0)
                else:
                    return ds_obj, 0

            global_dataset, removed_samples = _filter_dataset(global_dataset)
            if removed_samples > 0:
                total_count -= removed_samples
                logger.info(
                    "LLaVA-Vid: skipped %d samples matching patterns %s.",
                    removed_samples,
                    skip_patterns,
                )

        if (
            self.llavavid_max_samples is not None
            and total_count > self.llavavid_max_samples
        ):
            target = self.llavavid_max_samples
            rng = random.Random(self.llavavid_sample_seed)
            if isinstance(global_dataset, HFDataset):
                global_dataset = global_dataset.shuffle(
                    seed=self.llavavid_sample_seed
                ).select(range(target))
            else:
                try:
                    total_len = len(global_dataset)
                    indices = rng.sample(range(total_len), target)
                    if hasattr(global_dataset, "select"):
                        global_dataset = global_dataset.select(indices)
                    else:
                        global_dataset = [global_dataset[i] for i in indices]
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to subsample LLaVA-Vid to {target} rows: {exc}"
                    ) from exc
            total_count = len(global_dataset)
            logger.info(
                "LLaVA-Vid: randomly subsampled to %s rows (seed=%s).",
                f"{total_count:,}",
                self.llavavid_sample_seed,
            )

        logger.info("LLaVA-Vid: %d configs loaded.", len(datasets_loaded))
        logger.info("LLaVA-Vid: %s total samples loaded.", f"{total_count:,}")

        self.dataset_length = total_count
        return global_dataset

    def __len__(self):
        return len(self.vid_data)

    def __getitem__(self, idx):
        max_try_count = 50

        for try_count in range(max_try_count):
            try:
                data = self._sample_data(idx)
            except Exception as exc:
                logger.warning(
                    "VideoCaptionDataset failed to fetch index %s on attempt %s/%s: %s",
                    idx,
                    try_count + 1,
                    max_try_count,
                    exc,
                )
                idx = random.randint(0, self.dataset_length - 1)
                continue

            if data is not None:
                return {
                    "video": data["video"],
                    "caption": data["caption"],
                }

            idx = random.randint(0, self.dataset_length - 1)

        logger.warning(
            "VideoCaptionDataset exhausted %s attempts without a valid sample; returning None.",
            max_try_count,
        )
        return None

    def _sample_data_webvid10m(self):
        store_path = osp.join(self.webvid10m_path, "video_store")
        os.makedirs(store_path, exist_ok=True)

        row = self.vid_data.sample(1).iloc[0]
        url = (
            row.get("contentUrl")
            or row.get("content_url")
            or row.get("url")
            or row.get("video_url")
        )
        if not isinstance(url, str) or not url.strip():
            raise ValueError("WebVid row is missing a valid URL column.")
        video_id = (
            row.get("videoid")
            or row.get("video_id")
            or row.get("id")
            or hashlib.sha1(url.encode("utf-8")).hexdigest()
        )
        caption = row.get("name") or row.get("caption") or row.get("title") or ""
        video_id = str(video_id)
        caption = str(caption)

        video_path = osp.join(store_path, f"{video_id}.mp4")
        if not osp.exists(video_path):  # not downloaded yet
            success = download_video_url(url, video_path)
            if not success:
                raise RuntimeError(f"Failed to download WebVid video from {url}")

        return video_path, caption

    def _sample_data(self, idx):
        if self.dataset_name == "webvid10m":
            # currently randomly sample from the dataset
            video_path, caption = self._sample_data_webvid10m()
        elif self.dataset_name == "openvid1m":
            data_row = self.vid_data.iloc[idx]
            video_path = osp.join(self.dataset_root, "video", data_row["video"])
            caption = data_row["caption"]
        elif self.dataset_name == "llavavid":
            data_row = self.vid_data[idx]
            video_entry = data_row["video"]
            cfg_name = (
                data_row.get("llavavid_config") if isinstance(data_row, dict) else None
            )
            caption = data_row["conversations"]  # this is a list of turns in llavavid

            resolved_video_path = None
            if isinstance(video_entry, str):
                candidate_paths = []
                video_path_obj = Path(video_entry)
                if video_path_obj.is_absolute() and video_path_obj.exists():
                    resolved_video_path = video_path_obj
                else:
                    if hasattr(self, "llavavid_video_root"):
                        base_root = Path(self.llavavid_video_root)
                        if cfg_name:
                            candidate_paths.append(base_root / cfg_name / video_entry)
                        candidate_paths.append(base_root / video_entry)
                    # Also allow treating the stored value as relative to current dir.
                    candidate_paths.append(Path(video_entry))

                    for candidate in candidate_paths:
                        if candidate.exists():
                            resolved_video_path = candidate
                            break

            if resolved_video_path is None:
                logger.warning(
                    "LLaVA-Video sample missing video file: %s (config=%s)",
                    video_entry,
                    cfg_name,
                )
                return None

            if resolved_video_path.suffix.lower() == ".mkv":
                logger.warning(
                    "LLaVA-Video skipping MKV file: %s (config=%s)",
                    resolved_video_path,
                    cfg_name,
                )
                return None

            video_path = str(resolved_video_path)
        else:
            raise ValueError(
                f"Invalid dataset name: {self.dataset_name}. Available datasets: openvid1m, webvid10m, llavavid"
            )

        try:
            frames = load_video_mp4(
                video_path=video_path,
                sample_method=self.sample_method,
                num_frames=self.num_frames,
                resolution=self.resolution,
                transform=self.transform,
                max_duration=self.max_video_seconds,
                strict=False,
            )
        except Exception as exc:
            logger.warning(
                "LLaVA-Video sample failed to load (%s): %s",
                video_path,
                exc,
            )
            return None
        if frames is None:
            logger.warning(
                "LLaVA-Video sample timed out while reading frames (%s); skipping sample.",
                video_path,
            )
            return None

        return {
            "video": frames,  # torch tensor (T, C, H, W)
            "caption": caption,  # input_ids (seq_len); str
        }


def download_video_url(url: str, save_path, timeout=10, max_retries=3) -> bool:
    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(save_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            return True  # Success

        except Exception as e:
            logger.warning(
                "[Attempt %d/%d] Download failed: %s", attempt, max_retries, e
            )
            if attempt < max_retries:
                sleep_time = 2 ** (attempt - 1)  # exponential backoff: 1,2,4,8,...
                time.sleep(sleep_time)
            else:
                return False  # all attempts failed

    return False


def load_video_mp4(
    video_path,
    sample_method: str = "uniform",
    num_frames: int = 8,
    resolution: int = 256,
    transform=None,
    *,
    max_duration: Optional[float] = None,
    per_frame_timeout: float = 1.5,
    read_retry_interval: float = 0.05,
    strict: bool = True,
):
    """
    Load video frames and return them as a list of PIL images.

    Args:
        video_path: Path to the video file.
        sample_method: Sampling method, 'uniform', 'random', or 'uniform_sequential'.
        num_frames: Number of frames to sample from the video.
        per_frame_timeout: Max seconds to block while seeking/reading a frame.
        read_retry_interval: Delay between read retries while waiting for a frame.
        strict: When False, return None on timeout/seek failure instead of raising.

    Returns:
        List[Image.Image] | None (if strict=False and a timeout/seek failure occurs)
    """

    with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file {video_path}")

        if per_frame_timeout <= 0:
            per_frame_timeout = 0.1
        if read_retry_interval <= 0:
            read_retry_interval = 0.01

        def _read_frame_with_timeout(frame_index: Optional[int] = None):
            if frame_index is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            deadline = time.monotonic() + per_frame_timeout
            while True:
                ret, frame = cap.read()
                if ret and frame is not None:
                    return frame
                if time.monotonic() >= deadline:
                    return None
                time.sleep(
                    min(read_retry_interval, max(deadline - time.monotonic(), 0.0))
                )

        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except Exception:
            frame_count = -1
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if not np.isfinite(fps) or fps <= 0:
                fps = None
        except Exception:
            fps = None

        max_frames_allowed: Optional[int] = None
        if max_duration is not None and max_duration > 0 and fps:
            max_frames_allowed = max(1, int(max_duration * fps))

        if frame_count is None or frame_count <= 0:
            # Fallback: attempt to read sequentially but stop early on failure
            frames = []
            try:
                while len(frames) < num_frames:
                    if (
                        max_frames_allowed is not None
                        and len(frames) >= max_frames_allowed
                    ):
                        break
                    frame = _read_frame_with_timeout()
                    if frame is None:
                        break
                    frames.append(
                        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    )
            finally:
                cap.release()

            if len(frames) < num_frames:
                msg = f"Video {video_path} has insufficient frames ({len(frames)})."
                if strict:
                    raise ValueError(msg)
                if not frames:
                    logger.warning("%s Skipping sample.", msg)
                    return None
                logger.warning("%s Padding by repeating frames to %d.", msg, num_frames)
                frames.extend([frames[-1]] * (num_frames - len(frames)))
            selected = frames
        else:
            effective_count = frame_count
            if max_frames_allowed is not None and effective_count > max_frames_allowed:
                effective_count = max_frames_allowed

            allow_repeat = False
            if effective_count < num_frames:
                cap.release()
                msg = f"Video {video_path} has insufficient frames ({effective_count})."
                if strict:
                    raise ValueError(msg)
                logger.warning("%s Padding by repeating frames to %d.", msg, num_frames)
                allow_repeat = True

            sequential_methods = {
                "uniform_sequential",
                "uniform-sequential",
                "sequential",
            }
            sequential_sampling = False
            if sample_method in sequential_methods:
                indices = np.linspace(0, effective_count - 1, num_frames).astype(int)
                sequential_sampling = True
            elif sample_method == "uniform":
                indices = np.linspace(0, effective_count - 1, num_frames).astype(int)
            elif sample_method == "random":
                indices = np.sort(
                    np.random.choice(effective_count, num_frames, replace=allow_repeat)
                )
            else:
                cap.release()
                raise ValueError(f"Sampling method {sample_method} not supported.")

            selected = []
            try:
                if sequential_sampling:
                    current_position: Optional[int] = None
                    for idx_value in indices:
                        target_idx = int(idx_value)
                        if current_position is None:
                            frame = _read_frame_with_timeout(target_idx)
                            if frame is None:
                                msg = (
                                    f"Timed out ({per_frame_timeout:.2f}s) seeking initial frame "
                                    f"{target_idx} in {video_path}"
                                )
                                if strict:
                                    raise TimeoutError(msg)
                                logger.warning("%s. Skipping sample.", msg)
                                return None
                            current_position = target_idx
                        else:
                            if target_idx <= current_position:
                                continue
                            skip = target_idx - current_position - 1
                            next_idx = current_position + 1
                            while skip > 0:
                                skipped_frame = _read_frame_with_timeout(None)
                                if skipped_frame is None:
                                    msg = (
                                        f"Timed out ({per_frame_timeout:.2f}s) while sequentially "
                                        f"advancing to frame {next_idx} in {video_path}"
                                    )
                                    if strict:
                                        raise TimeoutError(msg)
                                    logger.warning("%s. Skipping sample.", msg)
                                    return None
                                current_position += 1
                                next_idx = current_position + 1
                                skip -= 1
                            frame = _read_frame_with_timeout(None)
                            if frame is None:
                                msg = (
                                    f"Timed out ({per_frame_timeout:.2f}s) reading frame "
                                    f"{target_idx} sequentially in {video_path}"
                                )
                                if strict:
                                    raise TimeoutError(msg)
                                logger.warning("%s. Skipping sample.", msg)
                                return None
                            current_position = target_idx
                        selected.append(
                            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        )
                else:
                    for idx in indices:
                        frame = _read_frame_with_timeout(int(idx))
                        if frame is None:
                            msg = f"Timed out ({per_frame_timeout:.2f}s) seeking frame {idx} in {video_path}"
                            if strict:
                                raise TimeoutError(msg)
                            logger.warning("%s. Skipping sample.", msg)
                            return None
                        selected.append(
                            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        )
            finally:
                cap.release()
            if allow_repeat and selected:
                if len(selected) < num_frames:
                    selected.extend([selected[-1]] * (num_frames - len(selected)))

        sampled_frames = []
        for frame in selected:
            if transform:
                frame = transform(frame, resolution=resolution)
            sampled_frames.append(frame)

        return sampled_frames


FRAME_EXTS_DEFAULT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _natural_key(value: str) -> list[Any]:
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", value)
    ]


def load_video_frames_dir(
    frame_dir: str,
    sample_method: str = "uniform",
    num_frames: int = 8,
    resolution: int = 256,
    transform=None,
    *,
    frame_exts: Optional[Sequence[str]] = None,
    strict: bool = True,
):
    if not os.path.isdir(frame_dir):
        msg = f"Frame directory not found: {frame_dir}"
        if strict:
            raise FileNotFoundError(msg)
        logger.warning("%s. Skipping sample.", msg)
        return None

    exts = tuple(ext.lower() for ext in (frame_exts or FRAME_EXTS_DEFAULT))
    frame_paths = []
    for entry in os.scandir(frame_dir):
        if not entry.is_file():
            continue
        if entry.name.lower().endswith(exts):
            frame_paths.append(entry.path)

    if not frame_paths:
        msg = f"No frames found in {frame_dir}"
        if strict:
            raise FileNotFoundError(msg)
        logger.warning("%s. Skipping sample.", msg)
        return None

    frame_paths.sort(key=lambda p: _natural_key(os.path.basename(p)))
    frame_count = len(frame_paths)
    allow_repeat = False
    if frame_count < num_frames:
        msg = f"Frame directory {frame_dir} has insufficient frames ({frame_count})."
        if strict:
            raise ValueError(msg)
        logger.warning("%s Padding by repeating frames to %d.", msg, num_frames)
        allow_repeat = True

    sequential_methods = {"uniform_sequential", "uniform-sequential", "sequential"}
    if sample_method in sequential_methods or sample_method == "uniform":
        indices = np.linspace(0, frame_count - 1, num_frames).astype(int)
    elif sample_method == "random":
        indices = np.sort(
            np.random.choice(frame_count, num_frames, replace=allow_repeat)
        )
    else:
        raise ValueError(f"Sampling method {sample_method} not supported.")

    sampled_frames = []
    for idx in indices:
        frame_path = frame_paths[int(idx)]
        try:
            frame = Image.open(frame_path).convert("RGB")
        except Exception as exc:
            if strict:
                raise RuntimeError(f"Failed to read frame {frame_path}: {exc}") from exc
            logger.warning(
                "Failed to read frame %s (%s). Skipping sample.", frame_path, exc
            )
            return None
        if transform:
            frame = transform(frame, resolution=resolution)
        sampled_frames.append(frame)

    return sampled_frames


def _strip_video_placeholder(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    return (
        text.replace("<video>\r\n", "")
        .replace("<video>\n", "")
        .replace("<video>", "")
        .strip()
    )


class ShareGPTVideoSFTDataset(Dataset):
    def __init__(
        self,
        *,
        jsonl_path: str,
        transform=None,
        resolution: int = 256,
        num_frames: int = 8,
        sample_method: str = "uniform",
        strip_video_token: bool = True,
        frame_exts: Optional[Sequence[str]] = None,
        require_video: bool = True,
    ) -> None:
        self.transform = transform
        self.resolution = resolution
        self.num_frames = num_frames
        self.sample_method = sample_method or "uniform"
        self.strip_video_token = strip_video_token
        self.frame_exts = frame_exts
        self.require_video = require_video

        path = Path(jsonl_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"ShareGPTVideo jsonl not found: {path}")

        self.items: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    continue
                video_path = record.get("video")
                if not isinstance(video_path, str) or not video_path:
                    continue
                if self.require_video and not os.path.exists(video_path):
                    continue
                question = record.get("question") or ""
                answer = record.get("answer") or ""
                if self.strip_video_token:
                    question = _strip_video_placeholder(question)
                if not isinstance(answer, str):
                    answer = str(answer)
                self.items.append(
                    {
                        "video": video_path,
                        "question": question,
                        "answer": answer,
                    }
                )

        if not self.items:
            raise RuntimeError(f"ShareGPTVideo dataset is empty after loading {path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Optional[dict[str, Any]]:
        item = self.items[idx]
        video_path = item["video"]
        frames = None
        if os.path.isdir(video_path):
            frames = load_video_frames_dir(
                frame_dir=video_path,
                sample_method=self.sample_method,
                num_frames=self.num_frames,
                resolution=self.resolution,
                transform=self.transform,
                frame_exts=self.frame_exts,
                strict=False,
            )
        else:
            frames = load_video_mp4(
                video_path=video_path,
                sample_method=self.sample_method,
                num_frames=self.num_frames,
                resolution=self.resolution,
                transform=self.transform,
                strict=False,
            )

        if frames is None:
            return None

        caption = [
            {"from": "human", "value": item["question"]},
            {"from": "gpt", "value": item["answer"]},
        ]
        if os.environ.get("DEBUG_FIRST_VIDEO_QA") and not getattr(
            self, "_debug_printed", False
        ):
            logger.info("[DEBUG_FIRST_VIDEO_QA] question: %s", item["question"])
            logger.info("[DEBUG_FIRST_VIDEO_QA] answer: %s", item["answer"])
            self._debug_printed = True
        return {
            "video": frames,
            "caption": caption,
        }


class VideoSpeechDataset(Dataset):
    """Loads paired video clips and speech audio paths or pre-tokenized speech."""

    def __init__(
        self,
        *,
        transform=None,
        resolution: int = 256,
        num_frames: int = 8,
        video_root: Optional[str] = None,
        audio_root: Optional[str] = None,
        speech_dir_name: str = "openvid-speech-trunc",
        index_path: Optional[str] = None,
        sample_method: str = "uniform",
        precomputed_tokens_root: Optional[str] = None,
        validate_paths: bool = False,
        index_cache_path: Optional[str] = None,
        max_video_seconds: Optional[float] = None,
        preload_audio: bool = False,
        max_audio_duration: Optional[float] = None,
    ) -> None:
        self.transform = transform
        self.resolution = resolution
        self.num_frames = num_frames
        self.sample_method = sample_method or "uniform"
        valid_sample_methods = {
            "uniform",
            "random",
            "uniform_sequential",
            "uniform-sequential",
            "sequential",
        }
        if self.sample_method not in valid_sample_methods:
            logger.warning(
                "Unknown sample_method '%s', defaulting to 'uniform'. Valid options: %s",
                self.sample_method,
                ", ".join(sorted(valid_sample_methods)),
            )
            self.sample_method = "uniform"

        video_root = video_root or _env_or_project_path(
            "DYNIN_OMNI_VIDEO_SPEECH_VIDEO_ROOT",
            "datasets",
            "video",
            "openvid1m",
            "video",
            "video",
        )
        audio_root = audio_root or _env_or_project_path(
            "DYNIN_OMNI_VIDEO_SPEECH_AUDIO_ROOT",
            "datasets",
            "video-speech",
        )
        index_path = index_path or _env_or_project_path(
            "DYNIN_OMNI_VIDEO_SPEECH_INDEX_PATH",
            "datasets",
            "video-speech",
            "openvid-speech.csv",
        )

        self.video_root = Path(video_root).expanduser().resolve()
        audio_base = Path(audio_root).expanduser()
        if speech_dir_name:
            audio_base = audio_base / speech_dir_name
        self.audio_root = audio_base.resolve()

        self.index_path = Path(index_path).expanduser().resolve()
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"VideoSpeechDataset index not found: {self.index_path}"
            )

        self.validate_paths = validate_paths

        self.precomputed_tokens_root = (
            Path(precomputed_tokens_root).expanduser().resolve()
            if precomputed_tokens_root
            else None
        )
        if (
            self.precomputed_tokens_root is not None
            and not self.precomputed_tokens_root.exists()
        ):
            logger.warning(
                "Precomputed speech token root %s missing; falling back to raw audio paths.",
                self.precomputed_tokens_root,
            )
            self.precomputed_tokens_root = None

        self._samples: list[tuple[Path, Path]] = []
        self._token_cache: Dict[str, torch.Tensor] = {}
        self._token_cache_limit = 4096
        self.index_cache_path = (
            Path(index_cache_path).expanduser().resolve() if index_cache_path else None
        )
        self.max_video_seconds = (
            max_video_seconds if max_video_seconds and max_video_seconds > 0 else None
        )
        self.preload_audio = bool(preload_audio)
        if max_audio_duration is not None and max_audio_duration <= 0:
            max_audio_duration = None
        self.max_audio_duration = (
            max_audio_duration if max_audio_duration is not None else None
        )

        if not self._load_index_from_cache():
            self._load_index()
            self._store_index_cache()

        if not self._samples:
            raise RuntimeError(
                f"VideoSpeechDataset found no valid samples in {self.index_path}"
            )

    def _load_index(self) -> None:
        missing = 0
        with self.index_path.open("r", newline="") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if not row:
                    continue
                base = row[0].strip()
                if not base:
                    continue
                if base.lower().endswith(".wav"):
                    base = base[:-4]
                video_path = (self.video_root / f"{base}.mp4").resolve()
                audio_path = (self.audio_root / f"{base}.wav").resolve()
                if self.validate_paths:
                    if not video_path.is_file() or not audio_path.is_file():
                        missing += 1
                        continue
                self._samples.append((video_path, audio_path))
        if missing:
            logger.info(
                "VideoSpeechDataset skipped %d entries missing media (index=%s)",
                missing,
                self.index_path,
            )

    def _load_index_from_cache(self) -> bool:
        if self.index_cache_path is None or not self.index_cache_path.exists():
            return False
        try:
            cached = torch.load(self.index_cache_path, map_location="cpu")
            if not isinstance(cached, list):
                raise ValueError("Cache payload is not a list")
            restored: list[tuple[Path, Path]] = []
            for entry in cached:
                if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                    raise ValueError("Invalid cache entry format")
                video_str, audio_str = entry
                restored.append((Path(video_str), Path(audio_str)))
            if self.validate_paths:
                filtered: list[tuple[Path, Path]] = []
                missing = 0
                for video_path, audio_path in restored:
                    if video_path.is_file() and audio_path.is_file():
                        filtered.append((video_path, audio_path))
                    else:
                        missing += 1
                if missing:
                    logger.info(
                        "VideoSpeechDataset cache skipped %d entries missing media (cache=%s)",
                        missing,
                        self.index_cache_path,
                    )
                self._samples = filtered
            else:
                self._samples = restored
            logger.info(
                "VideoSpeechDataset loaded %d samples from cache %s",
                len(self._samples),
                self.index_cache_path,
            )
            return True
        except Exception as exc:
            logger.warning(
                "Failed to load VideoSpeechDataset cache %s: %s",
                self.index_cache_path,
                exc,
            )
            return False

    def _store_index_cache(self) -> None:
        if self.index_cache_path is None or not self._samples:
            return
        try:
            payload = [(str(v), str(a)) for v, a in self._samples]
            cache_dir = self.index_cache_path.parent
            cache_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = self.index_cache_path.with_suffix(
                self.index_cache_path.suffix + f".tmp.{os.getpid()}"
            )
            torch.save(payload, tmp_path)
            os.replace(tmp_path, self.index_cache_path)
            logger.info(
                "VideoSpeechDataset cached %d samples to %s",
                len(self._samples),
                self.index_cache_path,
            )
        except Exception as exc:
            logger.warning(
                "Failed to write VideoSpeechDataset cache %s: %s",
                self.index_cache_path,
                exc,
            )
            with contextlib.suppress(OSError):
                if "tmp_path" in locals() and tmp_path.exists():
                    tmp_path.unlink()

    def __len__(self) -> int:
        return len(self._samples)

    def _transform_frame(self, image: Image.Image, resolution: int) -> torch.Tensor:
        if self.transform is None:
            return utils_image_transform(image, resolution)
        try:
            return self.transform(image, resolution=resolution)
        except TypeError:
            return self.transform(image)

    def _resolve_token_path(self, audio_path: Path) -> Optional[Path]:
        if self.precomputed_tokens_root is None:
            return None
        digest = hashlib.sha1(
            _resolve_fs_path(str(audio_path)).encode("utf-8")
        ).hexdigest()
        return self.precomputed_tokens_root / digest[:2] / digest[2:4] / f"{digest}.pt"

    def _get_precomputed_tokens(self, audio_path: Path) -> Optional[torch.Tensor]:
        cache_key = _resolve_fs_path(str(audio_path))
        cached = self._token_cache.get(cache_key)
        if cached is not None:
            return cached.clone()

        token_path = self._resolve_token_path(audio_path)
        if token_path is None or not token_path.exists():
            return None
        try:
            tokens = torch.load(token_path, map_location="cpu")
        except Exception as exc:
            logger.warning(
                "Failed to load precomputed speech tokens %s: %s", token_path, exc
            )
            return None
        if not isinstance(tokens, torch.Tensor):
            return None
        tokens = tokens.to(dtype=torch.long, copy=False)
        if len(self._token_cache) < self._token_cache_limit:
            self._token_cache[cache_key] = tokens
        return tokens.clone()

    def _load_audio_waveform(
        self, audio_path: Path
    ) -> Optional[Tuple[np.ndarray, int]]:
        if not self.preload_audio:
            return None
        try:
            with sf.SoundFile(str(audio_path), mode="r") as handle:
                samplerate = handle.samplerate
                total_frames = len(handle)
                frames = total_frames
                if self.max_audio_duration is not None and samplerate > 0:
                    max_frames = int(self.max_audio_duration * samplerate)
                    frames = min(total_frames, max_frames)
                handle.seek(0)
                audio = handle.read(frames=frames, dtype="float32", always_2d=False)
        except Exception as exc:
            logger.warning(
                "VideoSpeechDataset failed to preload audio %s: %s", audio_path, exc
            )
            return None
        if isinstance(audio, np.ndarray) and audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        return audio, samplerate

    def _prepare_speech_entry(self, audio_path: Path):
        tokens = self._get_precomputed_tokens(audio_path)
        if tokens is not None:
            return tokens
        if self.preload_audio:
            loaded = self._load_audio_waveform(audio_path)
        else:
            loaded = None
        if loaded is not None:
            waveform, samplerate = loaded
        else:
            waveform, samplerate = None, None
        if waveform is None:
            return str(audio_path)
        return {
            "audio_path": str(audio_path),
            "audio_array": waveform,
            "audio_sample_rate": samplerate,
            "audio_tokens": None,
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video_path, audio_path = self._samples[idx]
        max_attempts = 2
        last_error: Optional[Exception] = None
        frames: Optional[list[torch.Tensor]] = None

        for attempt in range(1, max_attempts + 1):
            try:
                frames = load_video_mp4(
                    str(video_path),
                    sample_method=self.sample_method,
                    num_frames=self.num_frames,
                    resolution=self.resolution,
                    transform=self._transform_frame,
                    max_duration=self.max_video_seconds,
                    strict=False,
                )
            except Exception as exc:
                last_error = exc
                frames = None
                logger.warning(
                    "VideoSpeechDataset decode attempt %d/%d failed for %s: %s",
                    attempt,
                    max_attempts,
                    video_path,
                    exc,
                )
            if frames:
                break

        if not frames:
            if last_error is not None:
                logger.warning(
                    "VideoSpeechDataset dropping %s after %d failed decode attempts: %s",
                    video_path,
                    max_attempts,
                    last_error,
                )
            else:
                logger.warning(
                    "VideoSpeechDataset dropping %s because decoding produced no frames.",
                    video_path,
                )
            return None

        speech_entry = self._prepare_speech_entry(audio_path)
        return {
            "video": frames,
            "speech": speech_entry,
        }


class TextImageInterleavedDataset:
    """
    HF-backed dataset that yields rows of:
      {
        "image_paths": [str, ...],     # absolute paths (no decoding)
        "user_text": str,
        "assistant_text": str,
      }
    """

    def __init__(
        self,
        *,
        configs: Union[str, Sequence[str], None] = None,  # default: all configs
        split: str = "train",
        data_root: Optional[str] = None,
        max_images: Optional[int] = None,
        filter_empty: bool = True,
        resolution: int = 256,
        # sampling controls
        per_config_fraction: float = 1 / 7,  # ← sample 1/7 PER CONFIG
        sample_seed: int = 42,
        # kept for compatibility, not used in this 1/7-per-config version
        max_samples: Optional[int] = 1_000_000,
        local_data_root: Optional[str] = None,
        local_data_files: Optional[Dict[str, Any]] = None,
        local_files_only: Optional[bool] = None,
    ):
        data_root = data_root or _env_or_project_path(
            "DYNIN_OMNI_MANTIS_DATA_ROOT",
            "datasets",
            "TIGER-Lab",
            "Mantis-Instruct",
        )
        self.data_root = _resolve_fs_path(data_root)
        self.split = self._normalize_split(split)
        self.max_images = max_images
        self.filter_empty = filter_empty
        self.resolution = resolution
        self.local_data_root = (
            _resolve_fs_path(local_data_root) if local_data_root else None
        )
        self.local_data_files = local_data_files or {}
        self.local_files_only = _resolve_local_files_only(local_files_only)
        self._download_config = _hf_download_config(self.local_files_only)

        # cache transforms
        self._tfm_crop = transforms.Compose(
            [
                transforms.Resize(
                    resolution, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop((resolution, resolution)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self._tfm_squash = transforms.Compose(
            [
                transforms.Resize(
                    (resolution, resolution),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # ---- resolve configs ----
        if configs is None or configs == "all":
            cfgs = self._resolve_configs_from_local()
            if not cfgs:
                cfgs = sorted(get_dataset_config_names("TIGER-Lab/Mantis-Instruct"))
        elif isinstance(configs, str):
            cfgs = [configs]
        else:
            cfgs = list(configs)
        self.configs = cfgs

        rng = np.random.default_rng(sample_seed)
        per_cfg_ds: List[HFDataset] = []

        acc = Accelerator()

        for cfg in cfgs:
            base_ds = self._load_base_dataset(cfg, acc)
            if base_ds is None:
                continue
            # --- SAMPLE 1/7 OF BASE ROWS *PER CONFIG* (before any map/expansion) ---
            n = len(base_ds)
            if n == 0:
                continue
            k = max(1, int(np.floor(n * per_config_fraction)))
            # reproducible uniform sample without replacement
            sel_idx = rng.choice(n, size=k, replace=False)
            base_ds = base_ds.select(list(sel_idx))

            # locate image dir for this (cfg, split)
            img_dir = self._resolve_img_dir(cfg, self.split)
            if img_dir is None:
                raise FileNotFoundError(
                    f"No image dir for config='{cfg}', split='{self.split}'"
                )

            # (1) attach constants
            def add_const_cols(batch):
                m = len(next(iter(batch.values()))) if batch else 0
                return {"config": [cfg] * m, "img_dir": [img_dir] * m}

            ds = base_ds.map(add_const_cols, batched=True)

            # (2) normalize image column → absolute string paths
            image_key = self._guess_image_key(ds.column_names)

            def make_abs_paths(batch):
                bases = batch["img_dir"]  # list[str] per row
                rels = batch[image_key]  # per-row: list[dict]|dict|list[str]|str|None

                def dict_to_rel(d: Dict[str, Any]) -> Optional[str]:
                    # typical HF Image: {"path": "...", "bytes": ...}
                    for k in ("path", "file_name", "filepath", "image_path", "name"):
                        v = d.get(k)
                        if isinstance(v, str) and v:
                            return v
                    # nested
                    img = d.get("image")
                    if isinstance(img, dict):
                        v = img.get("path")
                        if isinstance(v, str) and v:
                            return v
                    return None

                out_paths = []
                for base, r in zip(bases, rels):
                    # normalize r → list[str]
                    if r is None:
                        row = []
                    elif isinstance(r, str):
                        row = [r]
                    elif isinstance(r, dict):
                        s = dict_to_rel(r)
                        row = [s] if s else []
                    elif isinstance(r, list):
                        tmp = []
                        for x in r:
                            if isinstance(x, str):
                                tmp.append(x)
                            elif isinstance(x, dict):
                                s = dict_to_rel(x)
                                if s:
                                    tmp.append(s)
                        row = tmp
                    else:
                        row = []

                    # join to absolute (keep absolute if already)
                    abs_paths = [
                        p if os.path.isabs(p) else os.path.join(base, p)
                        for p in row
                        if isinstance(p, str)
                    ]

                    # cap if requested
                    if self.max_images is not None and len(abs_paths) > self.max_images:
                        abs_paths = abs_paths[: self.max_images]

                    out_paths.append(abs_paths)

                return {"image_paths": out_paths}

            ds = ds.map(make_abs_paths, batched=True)

            # (3) expand conversation: one row per (user → assistant) turn
            conv_key = "conversation"

            def expand_turns(batch):
                image_paths_list = batch["image_paths"]
                conversations = batch.get(conv_key, [[]] * len(image_paths_list))

                out_img_paths, out_user, out_assistant = [], [], []

                for img_paths, conv in zip(image_paths_list, conversations):
                    conv = conv or []
                    # walk adjacent pairs
                    i = 0
                    while i < len(conv) - 1:
                        a, b = conv[i], conv[i + 1]
                        if (
                            isinstance(a, dict)
                            and isinstance(b, dict)
                            and a.get("role") == "user"
                            and b.get("role") == "assistant"
                        ):
                            user_text = (a.get("content") or "").strip()
                            assistant_text = (b.get("content") or "").strip()
                            if (not self.filter_empty) or assistant_text:
                                out_img_paths.append(img_paths)
                                out_user.append(user_text)
                                out_assistant.append(assistant_text)
                            i += 2
                        else:
                            i += 1

                return {
                    "image_paths": out_img_paths,
                    "user_text": out_user,
                    "assistant_text": out_assistant,
                }

            ds = ds.map(expand_turns, batched=True, remove_columns=ds.column_names)

            if self.filter_empty:
                ds = ds.filter(lambda e: bool(e["assistant_text"]))

            per_cfg_ds.append(ds)

        if not per_cfg_ds:
            raise ValueError(
                "Empty dataset after per-config sampling and preprocessing."
            )

        self.dataset = (
            concatenate_datasets(per_cfg_ds) if len(per_cfg_ds) > 1 else per_cfg_ds[0]
        )
        self.dataset = self.dataset.with_format("python")
        logger.info(
            "[HF Dataset] per-config 1/7 sampled; configs=%s, split='%s', rows=%d",
            self.configs,
            self.split,
            len(self.dataset),
        )

    # ---- public API ----
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        start_idx = idx
        attempts = 0
        max_attempts = 10

        while attempts < max_attempts:
            ex = self.dataset[idx]

            text = (
                "<|start_header_id|>user<|end_header_id|>\n"
                f"{ex['user_text']}\n"
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
                f"{ex['assistant_text']}"
            )

            paths = ex["image_paths"]
            imgs: list[torch.Tensor] = []
            for path in paths:
                img = self._load_and_transform_one(path)
                if img is not None:
                    imgs.append(img)

            if imgs:
                return {
                    "images": imgs,
                    "text": text,
                }

            attempts += 1
            idx = (idx + 1) % len(self.dataset)
            if idx == start_idx:
                break

        raise RuntimeError(
            "TextImageInterleavedDataset: no valid images found after retries."
        )

    # ---- helpers ----
    @staticmethod
    def _normalize_split(split: str) -> str:
        s = split.lower()
        return {"val": "validation", "dev": "validation"}.get(s, s)

    def _resolve_img_dir(self, cfg: str, split: str) -> Optional[str]:
        # Typical local layout:
        #   {data_root}/{cfg}/{split}_images
        #   {data_root}/{cfg}/images
        cand1 = os.path.join(self.data_root, cfg, f"{split}_images")
        cand2 = os.path.join(self.data_root, cfg, "images")
        for c in (cand1, cand2):
            if os.path.isdir(c):
                return c
        return None

    def _load_and_transform_one(self, path: str):
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
        except FileNotFoundError:
            return None
        except Exception:
            return None

        return self._tfm_crop(im)

    @staticmethod
    def _guess_image_key(cols: List[str]) -> str:
        for k in ("images", "image_paths", "imgs", "paths", "image"):
            if k in cols:
                return k
        raise KeyError(f"Cannot find image column among {cols}")

    def _resolve_configs_from_local(self) -> List[str]:
        cfgs: List[str] = []

        if self.local_data_root:
            root = Path(self.local_data_root)
            if root.is_dir():
                for entry in sorted(root.iterdir()):
                    if not entry.is_dir():
                        continue
                    if self._has_split_data(entry):
                        cfgs.append(entry.name)

        if not cfgs and self.local_data_files:
            cfgs = [k for k in sorted(self.local_data_files.keys()) if k != "default"]

        return cfgs

    def _has_split_data(self, cfg_path: Path) -> bool:
        split_dir = cfg_path / self.split
        if split_dir.is_dir():
            return True

        alt_dirs = [
            cfg_path / f"{self.split}.dataset",
            cfg_path / f"{self.split}.arrow",
        ]
        for candidate in alt_dirs:
            if candidate.is_dir():
                return True

        patterns = [
            cfg_path / self.split / "*.arrow",
            cfg_path / self.split / "*.parquet",
            cfg_path / f"{self.split}/*.arrow",
            cfg_path / f"{self.split}/*.parquet",
            cfg_path / f"{self.split}*.arrow",
            cfg_path / f"{self.split}*.parquet",
        ]
        for pattern in patterns:
            if glob(str(pattern)):
                return True

        return False

    def _load_base_dataset(self, cfg: str, acc: Accelerator) -> Optional[HFDataset]:
        base_ds: Optional[HFDataset] = None

        if self.local_data_root is not None:
            base_ds = self._load_from_local_root(cfg)

        if base_ds is None and self.local_data_files:
            base_ds = self._load_from_local_data_files(cfg)

        if base_ds is not None:
            return base_ds

        kwargs = {}
        if self._download_config is not None:
            kwargs["download_config"] = self._download_config

        if acc.num_processes > 1:
            acc.wait_for_everyone()
        try:
            base_ds = _load_hf_dataset(
                "TIGER-Lab/Mantis-Instruct",
                cfg,
                split=self.split,
                local_files_only=self.local_files_only,
                **kwargs,
            )
        except Exception as exc:
            if self._download_config is not None:
                raise RuntimeError(
                    f"Failed to load local dataset for config='{cfg}'. "
                    "Ensure that the dataset is cached or provide 'local_data_root'."
                ) from exc
            raise
        finally:
            if acc.num_processes > 1:
                acc.wait_for_everyone()

        return base_ds

    def _load_from_local_root(self, cfg: str) -> Optional[HFDataset]:
        cfg_root = os.path.join(self.local_data_root, cfg)
        if not os.path.exists(cfg_root):
            return None

        candidates = [
            cfg_root,
            os.path.join(cfg_root, self.split),
            os.path.join(cfg_root, f"{self.split}.dataset"),
        ]

        for path in candidates:
            if not os.path.isdir(path):
                continue
            try:
                loaded = load_from_disk(path)
                if isinstance(loaded, DatasetDict):
                    if self.split in loaded:
                        return loaded[self.split]
                    continue
                return loaded
            except Exception:
                continue

        patterns = [
            os.path.join(cfg_root, f"{self.split}.parquet"),
            os.path.join(cfg_root, f"{self.split}/*.parquet"),
            os.path.join(cfg_root, f"{self.split}_*.parquet"),
            os.path.join(cfg_root, f"{self.split}.json"),
            os.path.join(cfg_root, f"{self.split}.jsonl"),
            os.path.join(cfg_root, f"{self.split}/*.jsonl"),
            os.path.join(cfg_root, f"{self.split}.arrow"),
            os.path.join(cfg_root, f"{self.split}/*.arrow"),
        ]

        for pattern in patterns:
            files = sorted(glob(pattern))
            if files:
                return self._load_from_files(files)

        return None

    def _load_from_local_data_files(self, cfg: str) -> Optional[HFDataset]:
        spec = self.local_data_files.get(cfg) or self.local_data_files.get("default")
        if spec is None:
            return None

        if isinstance(spec, str):
            entries = [spec]
            loader = None
        elif isinstance(spec, dict):
            loader = spec.get("type") or spec.get("loader") or spec.get("format")
            files = spec.get(self.split) or spec.get("files")
            if files is None:
                return None
            entries = files if isinstance(files, list) else [files]
        else:
            entries = list(spec)
            loader = None

        resolved_files: list[str] = []
        for entry in entries:
            if not entry:
                continue
            matched = _glob_paths(str(entry))
            if matched:
                resolved_files.extend(matched)
            else:
                candidate = _resolve_fs_path(str(entry))
                if os.path.exists(candidate):
                    resolved_files.append(candidate)

        if not resolved_files:
            return None

        return self._load_from_files(resolved_files, loader_hint=loader)

    def _load_from_files(
        self, files: list[str], loader_hint: Optional[str] = None
    ) -> Optional[HFDataset]:
        if not files:
            return None

        ext = Path(files[0]).suffix.lower()
        loader = loader_hint
        if loader is None:
            if ext in (".parquet",):
                loader = "parquet"
            elif ext in (".json", ".jsonl"):
                loader = "json"
            elif ext in (".arrow", ".feather"):
                loader = "arrow"

        if loader == "parquet":
            return _load_hf_dataset(
                "parquet",
                data_files={self.split: files},
                split=self.split,
                local_files_only=self.local_files_only,
            )
        if loader in {"json", "jsonl"}:
            return _load_hf_dataset(
                "json",
                data_files={self.split: files},
                split=self.split,
                local_files_only=self.local_files_only,
            )
        if loader == "arrow":
            datasets = [HFDataset.from_file(path) for path in files]
            return concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]

        return None


class HFInstructionTextDataset(Dataset):
    """Mixed instruction-following text dataset sourced from multiple HF corpora."""

    HF_SOURCES = (
        {
            "name": "garage-bAInd/Open-Platypus",
            "config": None,
            "split": "train",
            "user_key": "instruction",
            "assistant_key": "output",
        },
        {
            "name": "open-r1/OpenR1-Math-220k",
            "config": "all",
            "split": "train",
            "user_key": "problem",
            "assistant_key": "solution",
        },
        {
            "name": "teknium/OpenHermes-2.5",
            "config": None,
            "split": "train",
            "type": "conversation",
            "conv_key": "conversations",
        },
        {
            "name": "Magpie-Align/Magpie-Pro-300K-Filtered",
            "config": None,
            "split": "train",
            "type": "conversation",
            "conv_key": "conversations",
        },
    )

    def __init__(
        self,
        *,
        split: str = "train",
        max_samples_per_source: Optional[int] = None,
        max_total_samples: Optional[int] = None,
        seed: int = 42,
        local_files_only: Optional[bool] = None,
        sources: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        self.split = split
        self.seed = seed
        self.local_files_only = _resolve_local_files_only(local_files_only)
        self.samples: List[str] = []
        raw_source_list = (
            list(sources) if sources is not None else list(self.HF_SOURCES)
        )
        source_list: List[Dict[str, Any]] = []
        for raw_source in raw_source_list:
            plain_source = _to_plain(raw_source)
            if isinstance(plain_source, dict):
                source_list.append(plain_source)
            else:
                logger.warning(
                    "[HFInstructionTextDataset] Skipping non-dict source entry: %r",
                    raw_source,
                )

        rng = random.Random(seed)

        for source in source_list:
            desired_split = source.get("split", split)
            try:
                dataset_name = (
                    source.get("dataset_id")
                    or source.get("name")
                    or source.get("repo_id")
                )
                if not dataset_name:
                    logger.warning(
                        "[HFInstructionTextDataset] Source missing dataset_id/name: %s",
                        source,
                    )
                    continue
                dataset_config = source.get("config")
                if dataset_config is not None:
                    hf_ds = _load_hf_dataset(
                        dataset_name,
                        dataset_config,
                        split=desired_split,
                        local_files_only=self.local_files_only,
                    )
                else:
                    hf_ds = _load_hf_dataset(
                        dataset_name,
                        split=desired_split,
                        local_files_only=self.local_files_only,
                    )
            except Exception as exc:
                logger.warning(
                    "[HFInstructionTextDataset] Failed to load %s: %s",
                    dataset_name,
                    exc,
                )
                continue

            if (
                max_samples_per_source is not None
                and len(hf_ds) > max_samples_per_source
            ):
                hf_ds = hf_ds.shuffle(seed=seed).select(range(max_samples_per_source))

            src_type = source.get("type", "pair")
            if src_type == "conversation":
                conv_key = source.get("conv_key", "conversations")
                for example in hf_ds:
                    conv = example.get(conv_key)
                    if not isinstance(conv, list):
                        continue
                    current_user = None
                    for turn in conv:
                        role = str(turn.get("from", "")).lower()
                        text = str(turn.get("value", "")).strip()
                        if role == "human":
                            current_user = text
                        elif role in {"gpt", "assistant"} and current_user:
                            assistant_raw = text
                            formatted = self._format_dialogue(
                                current_user, assistant_raw
                            )
                            if formatted:
                                self.samples.append(formatted)
                            current_user = None
            else:
                user_key = source["user_key"]
                assistant_key = source["assistant_key"]

                for example in hf_ds:
                    user_raw = str(example.get(user_key, "")).strip()
                    assistant_raw = str(example.get(assistant_key, "")).strip()
                    if not user_raw or not assistant_raw:
                        continue

                    formatted = self._format_dialogue(user_raw, assistant_raw)
                    if formatted:
                        self.samples.append(formatted)

        if not self.samples:
            raise ValueError("HFInstructionTextDataset loaded zero valid samples.")

        rng.shuffle(self.samples)

        if max_total_samples is not None:
            self.samples = self.samples[:max_total_samples]

    @staticmethod
    def _format_dialogue(user_text: str, assistant_text: str) -> str:
        return (
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{user_text}\n"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            f"{assistant_text}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, str]:
        return {"input_ids": self.samples[index]}

    @staticmethod
    def collate_fn(batch: List[Dict[str, str]]) -> Dict[str, List[str]]:
        return {"input_ids": [example["input_ids"] for example in batch]}


class ReasoningSFTCSVDataset(Dataset):
    """Loads a CSV with problem/solution columns and formats chat-style samples."""

    def __init__(
        self,
        *,
        csv_path: str,
        seed: int = 42,
        max_total_samples: Optional[int] = None,
    ) -> None:
        self.csv_path = csv_path
        self.samples: List[str] = []

        if not csv_path or not os.path.exists(csv_path):
            raise FileNotFoundError(f"ReasoningSFTCSVDataset missing file: {csv_path}")

        self._load_csv(csv_path)
        if not self.samples:
            raise ValueError("ReasoningSFTCSVDataset loaded zero valid samples.")

        rng = random.Random(seed)
        rng.shuffle(self.samples)
        if max_total_samples is not None:
            self.samples = self.samples[:max_total_samples]

    @staticmethod
    def _format_dialogue(user_text: str, assistant_text: str) -> str:
        return (
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{user_text}\n"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            f"{assistant_text}"
        )

    def _load_csv(self, csv_path: str) -> None:
        with open(csv_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                problem = (row.get("problem") or "").strip()
                solution = (row.get("solution") or "").strip()
                if not problem or not solution:
                    continue
                if not solution.lstrip().startswith("<think>"):
                    continue
                formatted = self._format_dialogue(f"{problem} /think", solution)
                self.samples.append(formatted)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, str]:
        return {"input_ids": self.samples[index]}

    @staticmethod
    def collate_fn(batch: List[Dict[str, str]]) -> Dict[str, List[str]]:
        return {"input_ids": [example["input_ids"] for example in batch]}


class PickAPicV2Dataset(Dataset):
    """Loads Min-Jaewon/pickapic-v2 for text-to-image fine-tuning."""

    def __init__(
        self,
        split: str = "train",
        resolution: int = 256,
        dataset_name: str = "Min-Jaewon/pickapic-v2",
        cache_dir: str | None = None,
        local_files_only: Optional[bool] = None,
    ) -> None:
        self.resolution = resolution
        self.dataset_name = dataset_name
        self.cache_dir = _resolve_fs_path(cache_dir) if cache_dir else None
        self.local_files_only = _resolve_local_files_only(local_files_only)

        self._dataset = _load_hf_dataset(
            dataset_name,
            split=split,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self._dataset[idx]

        prompt = sample.get("prompt", "") or sample.get("caption", "")
        image_field = sample.get("image")
        if image_field is None:
            raise KeyError("Expected 'image' field in pickapic-v2 sample")

        if isinstance(image_field, Image.Image):
            image = image_field.convert("RGB")
        elif isinstance(image_field, bytes):
            image = Image.open(BytesIO(image_field)).convert("RGB")
        else:
            image = Image.fromarray(np.array(image_field)).convert("RGB")

        image_tensor = utils_image_transform(image, self.resolution)

        return {
            "input_prompt": prompt,
            "output_prompt": None,
            "edit_prompt": None,
            "inverse_prompt": None,
            "input_image": image_tensor,
            "output_image": image_tensor,
        }


class PromptImageJsonlDataset(Dataset):
    """Loads local JSONL rows with prompt + image_path for T2I training."""

    def __init__(
        self,
        jsonl_path: Union[str, Sequence[str]],
        resolution: int = 256,
        prompt_keys: Sequence[str] = ("prompt", "query"),
        image_keys: Sequence[str] = ("image_path", "image"),
        skip_missing: bool = True,
        cache_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self.resolution = resolution
        self.prompt_keys = tuple(prompt_keys)
        self.image_keys = tuple(image_keys)
        self.skip_missing = skip_missing
        self.cache_path = str(cache_path) if cache_path else None
        self.samples: list[dict[str, str]] = []

        if self.cache_path and os.path.isfile(self.cache_path):
            self._load_cache(self.cache_path)
        else:
            paths = self._resolve_paths(jsonl_path)
            if not paths:
                raise FileNotFoundError(
                    "PromptImageJsonlDataset could not resolve any jsonl files."
                )

            for path in paths:
                with open(path, "r", encoding="utf-8") as fh:
                    base_dir = os.path.dirname(path)
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        prompt = self._pick_first(record, self.prompt_keys)
                        image_path = self._pick_first(record, self.image_keys)
                        if not prompt or not image_path:
                            continue
                        image_path = self._resolve_record_path(
                            image_path, base_dir=base_dir
                        )
                        if skip_missing and not os.path.isfile(image_path):
                            continue
                        self.samples.append(
                            {
                                "prompt": str(prompt).strip(),
                                "image_path": image_path,
                            }
                        )

            if self.cache_path:
                self._write_cache(self.cache_path)

        if not self.samples:
            raise ValueError("PromptImageJsonlDataset loaded zero valid samples.")

        rng = random.Random(seed)
        rng.shuffle(self.samples)
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    @staticmethod
    def _pick_first(record: dict[str, Any], keys: Sequence[str]) -> Optional[str]:
        for key in keys:
            value = record.get(key)
            if value is not None:
                value = str(value).strip()
                if value:
                    return value
        return None

    @staticmethod
    def _resolve_paths(entries: Union[str, Sequence[str]]) -> list[str]:
        if isinstance(entries, (str, Path)):
            entries = [str(entries)]
        resolved: list[str] = []
        for entry in entries:
            if not entry:
                continue
            entry_path = _resolve_fs_path(str(entry))
            if os.path.isdir(entry_path):
                resolved.extend(
                    sorted(str(p) for p in Path(entry_path).glob("*.jsonl"))
                )
                continue
            matched = _glob_paths(str(entry))
            if matched:
                resolved.extend(matched)
            elif os.path.exists(entry_path):
                resolved.append(entry_path)
        return resolved

    @staticmethod
    def _resolve_record_path(path: str, *, base_dir: str) -> str:
        return _resolve_fs_path(path, base_dir=base_dir)

    def _load_cache(self, cache_path: str) -> None:
        base_dir = os.path.dirname(cache_path)
        with open(cache_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt = record.get("prompt")
                image_path = record.get("image_path")
                if not prompt or not image_path:
                    continue
                self.samples.append(
                    {
                        "prompt": str(prompt).strip(),
                        "image_path": self._resolve_record_path(
                            str(image_path).strip(), base_dir=base_dir
                        ),
                    }
                )

    def _write_cache(self, cache_path: str) -> None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as fh:
            for sample in self.samples:
                fh.write(json.dumps(sample, ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        image_tensor = utils_image_transform(image, self.resolution)
        return {
            "input_prompt": sample["prompt"],
            "output_prompt": None,
            "edit_prompt": None,
            "inverse_prompt": None,
            "input_image": image_tensor,
            "output_image": image_tensor,
        }


class BasicEditJsonlDataset(Dataset):
    """Loads local JSONL rows with prompt + image_path for I2I editing."""

    def __init__(
        self,
        jsonl_path: Union[str, Sequence[str]],
        resolution: int = 256,
        prompt_keys: Sequence[str] = ("prompt", "query"),
        image_keys: Sequence[str] = ("image_path", "image"),
        skip_missing: bool = True,
        cache_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self.resolution = resolution
        self.prompt_keys = tuple(prompt_keys)
        self.image_keys = tuple(image_keys)
        self.skip_missing = skip_missing
        self.cache_path = str(cache_path) if cache_path else None
        self.samples: list[dict[str, str]] = []

        if self.cache_path and os.path.isfile(self.cache_path):
            self._load_cache(self.cache_path)
        else:
            paths = PromptImageJsonlDataset._resolve_paths(jsonl_path)
            if not paths:
                raise FileNotFoundError(
                    "BasicEditJsonlDataset could not resolve any jsonl files."
                )

            for path in paths:
                with open(path, "r", encoding="utf-8") as fh:
                    base_dir = os.path.dirname(path)
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        prompt = PromptImageJsonlDataset._pick_first(
                            record, self.prompt_keys
                        )
                        src_path = record.get("source_image_path")
                        tgt_path = record.get("target_image_path")
                        if not src_path or not tgt_path:
                            image_path = PromptImageJsonlDataset._pick_first(
                                record, self.image_keys
                            )
                            src_path = image_path
                            tgt_path = image_path
                        if not prompt or not src_path or not tgt_path:
                            continue
                        src_path = PromptImageJsonlDataset._resolve_record_path(
                            str(src_path), base_dir=base_dir
                        )
                        tgt_path = PromptImageJsonlDataset._resolve_record_path(
                            str(tgt_path), base_dir=base_dir
                        )
                        if skip_missing and (
                            not os.path.isfile(src_path) or not os.path.isfile(tgt_path)
                        ):
                            continue
                        self.samples.append(
                            {
                                "prompt": str(prompt).strip(),
                                "source_image_path": src_path,
                                "target_image_path": tgt_path,
                            }
                        )

            if self.cache_path:
                self._write_cache(self.cache_path)

        if not self.samples:
            raise ValueError("BasicEditJsonlDataset loaded zero valid samples.")

        rng = random.Random(seed)
        rng.shuffle(self.samples)
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def _load_cache(self, cache_path: str) -> None:
        base_dir = os.path.dirname(cache_path)
        with open(cache_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt = record.get("prompt")
                src_path = record.get("source_image_path") or record.get("image_path")
                tgt_path = record.get("target_image_path") or record.get("image_path")
                if not prompt or not src_path or not tgt_path:
                    continue
                self.samples.append(
                    {
                        "prompt": str(prompt).strip(),
                        "source_image_path": PromptImageJsonlDataset._resolve_record_path(
                            str(src_path).strip(), base_dir=base_dir
                        ),
                        "target_image_path": PromptImageJsonlDataset._resolve_record_path(
                            str(tgt_path).strip(), base_dir=base_dir
                        ),
                    }
                )

    def _write_cache(self, cache_path: str) -> None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as fh:
            for sample in self.samples:
                fh.write(json.dumps(sample, ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        src_image = Image.open(sample["source_image_path"]).convert("RGB")
        tgt_image = Image.open(sample["target_image_path"]).convert("RGB")
        src_tensor = utils_image_transform(src_image, self.resolution)
        tgt_tensor = utils_image_transform(tgt_image, self.resolution)
        return {
            "input_prompt": None,
            "output_prompt": None,
            "edit_prompt": sample["prompt"],
            "inverse_prompt": None,
            "input_image": src_tensor,
            "output_image": tgt_tensor,
        }


class FluxReasonDataset(Dataset):
    """Loads LucasFang/FLUX-Reason-6M with composition/detail filtering for text-to-image."""

    def __init__(
        self,
        split: str = "train",
        resolution: int = 256,
        dataset_name: str = "LucasFang/FLUX-Reason-6M",
        cache_dir: str | None = None,
        score_threshold: float = 8.0,
        local_files_only: Optional[bool] = None,
        indices_cache_path: str | None = None,
        progress_interval: int | None = None,
    ) -> None:
        local_files_only = _resolve_local_files_only(local_files_only)
        cache_dir = _resolve_fs_path(cache_dir) if cache_dir else None
        # Force offline mode if requested
        if local_files_only:
            os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        self.resolution = resolution
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.score_threshold = score_threshold
        self.progress_interval = (
            progress_interval if progress_interval and progress_interval > 0 else None
        )
        dl_cfg = _hf_download_config(local_files_only)

        self._dataset = _load_hf_dataset(
            dataset_name,
            split=split,
            cache_dir=cache_dir,
            download_config=dl_cfg,
            local_files_only=local_files_only,
        )

        # Optional cached index to avoid re-scanning on every run
        cache_file: Path | None = None
        if indices_cache_path:
            cache_file = Path(indices_cache_path)
        elif cache_dir:
            safe_name = dataset_name.replace("/", "_")
            cache_file = (
                Path(cache_dir)
                / f"{safe_name}_idx_score{int(self.score_threshold*10)}.json"
            )
        else:
            safe_name = dataset_name.replace("/", "_")
            cache_file = (
                Path(_default_hf_cache_dir())
                / f"{safe_name}_idx_score{int(self.score_threshold*10)}.json"
            )

        self._indices: list[int] = []
        if cache_file and cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as rf:
                    self._indices = [int(x) for x in json.load(rf)]
            except Exception:
                self._indices = []

        if not self._indices:
            meta_view = self._dataset.with_format(
                type="python",
                columns=[
                    "bool_caption_composition",
                    "score_composition",
                    "caption_composition",
                    "bool_caption_detail",
                    "caption_detail",
                ],
                output_all_columns=False,
            )
            for idx, sample in enumerate(meta_view):
                comp_ok = (
                    bool(sample.get("bool_caption_composition"))
                    and sample.get("score_composition", 0) is not None
                )
                if comp_ok:
                    try:
                        comp_ok = float(
                            sample.get("score_composition", 0)
                        ) > self.score_threshold and bool(
                            sample.get("caption_composition")
                        )
                    except Exception:
                        comp_ok = False
                detail_ok = bool(sample.get("bool_caption_detail")) and bool(
                    sample.get("caption_detail")
                )
                if comp_ok or detail_ok:
                    self._indices.append(idx)
                if self.progress_interval and (idx + 1) % self.progress_interval == 0:
                    logger.info(
                        "[FluxReasonDataset] scanned %s samples, kept %s",
                        f"{idx + 1:,}",
                        f"{len(self._indices):,}",
                    )

            if cache_file:
                try:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(cache_file, "w", encoding="utf-8") as wf:
                        json.dump(self._indices, wf)
                except Exception:
                    pass

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        real_idx = self._indices[idx]
        sample = self._dataset[real_idx]

        comp_ok = bool(sample.get("bool_caption_composition"))
        if comp_ok:
            try:
                comp_ok = float(
                    sample.get("score_composition", 0)
                ) > self.score_threshold and bool(sample.get("caption_composition"))
            except Exception:
                comp_ok = False
        detail_ok = bool(sample.get("bool_caption_detail"))

        prompt = None
        if comp_ok:
            prompt = sample.get("caption_composition", "")
        elif detail_ok:
            prompt = sample.get("caption_detail", "")
        if not prompt:
            raise IndexError("Filtered sample without valid caption")

        image_field = sample.get("image")
        if image_field is None:
            raise IndexError("Expected 'image' field in FLUX-Reason sample")

        try:
            if isinstance(image_field, Image.Image):
                image = image_field.convert("RGB")
            elif isinstance(image_field, bytes):
                image = Image.open(BytesIO(image_field)).convert("RGB")
            else:
                image = Image.fromarray(np.array(image_field)).convert("RGB")
        except Exception as exc:
            raise IndexError(f"Failed to decode FLUX-Reason image: {exc}") from exc

        image_tensor = utils_image_transform(image, self.resolution)

        return {
            "input_prompt": prompt,
            "output_prompt": None,
            "edit_prompt": None,
            "inverse_prompt": None,
            "input_image": image_tensor,
            "output_image": image_tensor,
        }


class JourneyDBDataset(Dataset):
    """Loads JourneyDB JSONL for text-conditioned image (editing-style: prompt + target image)."""

    def __init__(
        self,
        jsonl_path: str,
        image_root: str,
        split: str = "train",
        resolution: int = 256,
        cache_dir: str | None = None,
        local_files_only: bool = True,
    ) -> None:
        self.resolution = resolution
        self.image_root = image_root
        if not os.path.isfile(jsonl_path):
            raise FileNotFoundError(f"JourneyDB jsonl not found: {jsonl_path}")

        # Manually load and keep only prompt / img_path to avoid schema cast errors
        records: list[dict[str, str]] = []
        missing_paths = 0
        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                prompt = obj.get("prompt")
                img_rel = obj.get("img_path")
                if not prompt or not img_rel:
                    continue
                img_path = img_rel
                if not os.path.isabs(img_path):
                    img_path = os.path.join(self.image_root, img_rel.lstrip("./"))
                if not os.path.isfile(img_path):
                    missing_paths += 1
                    continue
                records.append({"prompt": str(prompt), "img_path": img_path})

        if missing_paths:
            logger.warning(
                "JourneyDBDataset skipped %d missing image files.", missing_paths
            )

        self._dataset = records

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self._dataset[idx]
        prompt = sample.get("prompt", "")
        img_path = sample.get("img_path")
        if not img_path:
            raise KeyError("Expected 'img_path' in JourneyDB sample")
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as exc:
            raise IndexError(
                f"Failed to load JourneyDB image at {img_path}: {exc}"
            ) from exc
        image_tensor = utils_image_transform(image, self.resolution)

        return {
            "input_prompt": prompt,
            "output_prompt": None,
            "edit_prompt": None,
            "inverse_prompt": prompt,
            "input_image": image_tensor,
            "output_image": image_tensor,
        }


class UltraEditDataset(Dataset):
    """Loads BleachNick/UltraEdit_500k for image editing."""

    def __init__(
        self,
        split: str = "FreeForm",
        resolution: int = 256,
        dataset_name: str = "BleachNick/UltraEdit_500k",
        cache_dir: str | None = None,
        local_files_only: Optional[bool] = None,
    ) -> None:
        local_files_only = _resolve_local_files_only(local_files_only)
        self.resolution = resolution
        self.dataset_name = dataset_name
        self.cache_dir = _resolve_fs_path(cache_dir) if cache_dir else None
        dl_cfg = _hf_download_config(local_files_only)

        self._dataset = _load_hf_dataset(
            dataset_name,
            split=split,
            cache_dir=self.cache_dir,
            download_config=dl_cfg,
            local_files_only=local_files_only,
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self._dataset[idx]

        source_caption = sample.get("source_caption", "")
        target_caption = sample.get("target_caption", "")
        edit_prompt = sample.get("edit_prompt", "")

        input_image_field = sample.get("source_image")
        output_image_field = sample.get("edited_image")

        if input_image_field is None or output_image_field is None:
            raise KeyError(
                "Expected 'source_image' and 'edited_image' fields in UltraEdit sample"
            )

        def to_image(x):
            if isinstance(x, Image.Image):
                return x.convert("RGB")
            if isinstance(x, bytes):
                return Image.open(BytesIO(x)).convert("RGB")
            return Image.fromarray(np.array(x)).convert("RGB")

        input_img = utils_image_transform(to_image(input_image_field), self.resolution)
        output_img = utils_image_transform(
            to_image(output_image_field), self.resolution
        )

        return {
            "input_prompt": source_caption,
            "output_prompt": target_caption,
            "edit_prompt": edit_prompt,
            "inverse_prompt": source_caption,
            "input_image": input_img,
            "output_image": output_img,
        }


class CambrianInterleavedDataset(Dataset):
    """
    JSONL-backed MMU dataset (e.g., Cambrian7M_withsystemprompt.jsonl).
    Uses ONLY the first human→gpt turn. Keeps <image> placeholder in the user text.
    """

    def __init__(
        self,
        jsonl_path: Optional[str] = None,
        image_root: Optional[str] = None,
        resolution: int = 256,
        filter_empty: bool = False,
        local_files_only: Optional[bool] = True,
        max_invalid_image_retries: int = 10,
        max_samples: int | None = None,
        seed: int = 42,
        balance_by_folder: bool = False,
        answer_noise_prob: float = 0.0,
        answer_noise_seed: int = 0,
        answer_noise_strategy: str = "swap",
    ) -> None:
        jsonl_path = jsonl_path or _env_or_project_path(
            "DYNIN_OMNI_CAMBRIAN_JSONL_PATH",
            "datasets",
            "Cambrian-10M",
            "jsons",
            "Cambrian7M_withsystemprompt.jsonl",
        )
        image_root = image_root or _env_or_project_path(
            "DYNIN_OMNI_CAMBRIAN_IMAGE_ROOT",
            "datasets",
            "Cambrian-10M",
        )
        jsonl_path = _resolve_fs_path(jsonl_path)
        image_root = _resolve_fs_path(image_root)
        self.local_files_only = _resolve_local_files_only(local_files_only)
        if not (os.path.isfile(jsonl_path) or os.path.isdir(jsonl_path)):
            raise FileNotFoundError(f"jsonl_path not found: {jsonl_path}")
        self.image_root = image_root
        self.resolution = resolution
        self.filter_empty = filter_empty
        self.max_invalid_image_retries = max(0, max_invalid_image_retries)
        self.max_samples = int(max_samples) if max_samples is not None else None
        self.seed = int(seed)
        self.balance_by_folder = bool(balance_by_folder)
        self.answer_noise_prob = max(0.0, float(answer_noise_prob))
        self.answer_noise_seed = int(answer_noise_seed)
        self.answer_noise_strategy = (answer_noise_strategy or "swap").lower()
        self._tfm = transforms.Compose(
            [
                transforms.Resize(
                    (resolution, resolution),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # Allow using pre-cached Arrow dataset (save_to_disk) or raw JSONL.
        base: HFDataset
        if os.path.isdir(jsonl_path):
            # Expect an Arrow dataset saved via save_to_disk
            try:
                base = load_from_disk(jsonl_path)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load MMU cache at '{jsonl_path}'. "
                    "Rebuild the cache (or point jsonl_path to the raw JSONL)."
                ) from exc
        else:
            records: list[dict[str, Any]] = []
            progress_interval = 100000
            with open(jsonl_path, "r", encoding="utf-8") as fh:
                for idx, line in enumerate(fh, 1):
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    records.append(
                        {
                            "image": obj.get("image", ""),
                            "conversations": obj.get("conversations", []),
                        }
                    )
                    if progress_interval and idx % progress_interval == 0:
                        logger.info(
                            "[MMU jsonl] read %d lines from %s", idx, jsonl_path
                        )
            base = HFDataset.from_list(records)
            logger.info(
                "[MMU jsonl] finished reading %d rows from %s", len(records), jsonl_path
            )

        preprocessed = {"image_paths", "user_text", "assistant_text"}.issubset(
            base.column_names
        )

        if preprocessed:
            logger.info(
                "[MMU cache] detected preprocessed dataset; skipping extraction map"
            )
            ds = base
        else:

            def extract_first_turn(batch):
                out_image = []
                out_user = []
                out_asst = []
                for img_rel, conv in zip(
                    batch.get("image", []), batch.get("conversations", [])
                ):
                    user_text = None
                    asst_text = None
                    conv = conv or []
                    for i in range(len(conv) - 1):
                        a, b = conv[i], conv[i + 1]
                        if (
                            isinstance(a, dict)
                            and isinstance(b, dict)
                            and a.get("from") == "human"
                            and b.get("from") in {"gpt", "assistant"}
                        ):
                            user_text = (a.get("value") or "").strip()
                            asst_text = (b.get("value") or "").strip()
                            break
                    if not user_text or not asst_text:
                        continue
                    img_path = img_rel or ""
                    if img_path and not os.path.isabs(img_path):
                        img_path = os.path.join(image_root, img_path)
                    if not os.path.isfile(img_path):
                        continue
                    out_image.append([img_path])  # keep list for consistency
                    out_user.append(user_text)
                    out_asst.append(asst_text)
                return {
                    "image_paths": out_image,
                    "user_text": out_user,
                    "assistant_text": out_asst,
                }

            ds = base.map(
                extract_first_turn, batched=True, remove_columns=base.column_names
            )
            logger.info("[MMU map] after first-turn extraction: %d rows", len(ds))

        if filter_empty:
            ds = ds.filter(lambda e: bool(e["assistant_text"]))
            logger.info(
                "[MMU filter] after dropping empty assistant_text: %d rows", len(ds)
            )
        if self.max_samples is not None and len(ds) > self.max_samples:
            ds = self._subsample_dataset(ds)
            logger.info("[MMU sample] after subsampling: %d rows", len(ds))
        if len(ds) == 0:
            raise ValueError("CambrianInterleavedDataset produced zero samples.")
        self.dataset = ds.with_format("python")

    def __len__(self) -> int:
        return len(self.dataset)

    def _rng_for_idx(self, idx: int) -> random.Random:
        return random.Random(self.answer_noise_seed + idx * 1000003)

    def _should_corrupt(self, idx: int) -> bool:
        if self.answer_noise_prob <= 0:
            return False
        rng = self._rng_for_idx(idx)
        return rng.random() < self.answer_noise_prob

    def _swap_answer(self, idx: int, current: str) -> str:
        if len(self.dataset) <= 1:
            return current
        rng = self._rng_for_idx(idx)
        for _ in range(5):
            other_idx = rng.randrange(len(self.dataset) - 1)
            if other_idx >= idx:
                other_idx += 1
            other = self.dataset[other_idx]
            other_ans = (other.get("assistant_text") or "").strip()
            if other_ans and other_ans != current:
                return other_ans
        return current

    def _path_to_folder(self, path: str) -> str:
        if not path:
            return ""
        p = Path(path)
        if p.is_absolute():
            try:
                p = p.relative_to(self.image_root)
            except Exception:
                pass
        parts = p.parts
        return parts[0] if parts else ""

    def _subsample_dataset(self, ds: HFDataset) -> HFDataset:
        rng = random.Random(self.seed)

        if not self.balance_by_folder:
            indices = list(range(len(ds)))
            rng.shuffle(indices)
            return ds.select(indices[: self.max_samples])

        indices_by_folder: dict[str, list[int]] = {}
        for idx in range(len(ds)):
            ex = ds[idx]
            image_paths = ex.get("image_paths") or []
            path = ""
            if isinstance(image_paths, list) and image_paths:
                first = image_paths[0]
                if isinstance(first, list):
                    path = first[0] if first else ""
                else:
                    path = first
            key = self._path_to_folder(path)
            indices_by_folder.setdefault(key, []).append(idx)

        folders = [k for k, v in indices_by_folder.items() if v]
        if not folders:
            indices = list(range(len(ds)))
            rng.shuffle(indices)
            return ds.select(indices[: self.max_samples])

        rng.shuffle(folders)
        per_folder = self.max_samples // len(folders)
        remainder = self.max_samples % len(folders)
        selected: list[int] = []

        for key in folders:
            candidates = indices_by_folder[key]
            rng.shuffle(candidates)
            take = min(per_folder, len(candidates))
            selected.extend(candidates[:take])
            indices_by_folder[key] = candidates[take:]

        extra_folders = [k for k in folders if indices_by_folder[k]]
        while remainder > 0 and extra_folders:
            for key in list(extra_folders):
                if remainder <= 0:
                    break
                if indices_by_folder[key]:
                    selected.append(indices_by_folder[key].pop())
                    remainder -= 1
                if not indices_by_folder[key]:
                    extra_folders.remove(key)

        if len(selected) < self.max_samples:
            leftovers = [i for items in indices_by_folder.values() for i in items]
            rng.shuffle(leftovers)
            need = self.max_samples - len(selected)
            selected.extend(leftovers[:need])

        rng.shuffle(selected)
        return ds.select(selected[: self.max_samples])

    def __getitem__(self, idx):
        attempts = self.max_invalid_image_retries + 1
        current_idx = idx
        for attempt in range(attempts):
            ex = self.dataset[current_idx]
            user_text = ex["user_text"]
            assistant_text = ex["assistant_text"]
            if self._should_corrupt(current_idx):
                if self.answer_noise_strategy in {"swap", "random_sample"}:
                    assistant_text = self._swap_answer(current_idx, assistant_text)
            text = (
                "<|start_header_id|>user<|end_header_id|>\n"
                f"{user_text}\n"
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
                f"{assistant_text}"
            )
            imgs = []
            for p in ex["image_paths"]:
                if not p:
                    continue
                path = p[0] if isinstance(p, list) else p
                try:
                    img = Image.open(path).convert("RGB")
                    imgs.append(self._tfm(img))
                except Exception:
                    continue
            if imgs:
                return {"images": imgs, "text": text}

            if attempt < attempts - 1:
                logger.warning(
                    "CambrianInterleavedDataset: no loadable images for idx=%d (attempt %d/%d); resampling.",
                    current_idx,
                    attempt + 1,
                    attempts,
                )
                current_idx = random.randint(0, len(self.dataset) - 1)
                continue
        raise IndexError(
            "Failed to load image for idx=%d after %d attempts"
            % (current_idx, attempts)
        )


class HQEditX2IDataset(Dataset):

    def __init__(
        self,
        split: str = "train",
        resolution: int = 256,
        dataset_name: str = "UCSC-VLAA/HQ-Edit",
        cache_dir: Optional[str] = None,
        local_files_only: Optional[bool] = None,
    ):
        local_files_only = _resolve_local_files_only(local_files_only)
        self.resolution = resolution
        self.cache_dir = (
            _resolve_fs_path(cache_dir) if cache_dir else _default_hf_cache_dir()
        )

        self._dataset = _load_hf_dataset(
            dataset_name,
            split=split,
            cache_dir=self.cache_dir,
            local_files_only=local_files_only,
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self._dataset[idx]

        input_tensor = utils_image_transform(
            sample["input_image"].convert("RGB"), self.resolution
        )
        output_tensor = utils_image_transform(
            sample["output_image"].convert("RGB"), self.resolution
        )

        return {
            "input_prompt": sample["input"],
            "output_prompt": sample["output"],
            "edit_prompt": sample["edit"],
            "inverse_prompt": sample["inverse_edit"],
            "input_image": input_tensor,
            "output_image": output_tensor,
        }


class CombinedX2IDataset(Dataset):
    """Round-robin combination of multiple x2i-style datasets."""

    def __init__(self, datasets: Sequence[Dataset]):
        if not datasets:
            raise ValueError("CombinedX2IDataset requires at least one dataset.")
        self.datasets = list(datasets)
        self.lengths = [len(ds) for ds in self.datasets]
        if any(length == 0 for length in self.lengths):
            raise ValueError("Underlying x2i dataset has zero length.")
        self.cumulative = list(itertools.accumulate(self.lengths))
        self.total_length = self.cumulative[-1]

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= self.total_length:
            raise IndexError(
                f"Index {idx} out of bounds for CombinedX2IDataset of length {self.total_length}"
            )

        dataset_idx = bisect.bisect_right(self.cumulative, idx)
        prev = self.cumulative[dataset_idx - 1] if dataset_idx > 0 else 0
        local_idx = idx - prev
        return self.datasets[dataset_idx][local_idx]


if __name__ == "__main__":
    pass
