import os
import sys
import json
import torch
import warnings
import numpy as np
import pandas as pd
import string
from PIL import Image
from transformers import AutoTokenizer, AutoConfig

os.environ["USE_COT"] = "0"
repo_root = os.path.realpath(os.path.join(os.path.dirname(__file__), "../../../../.."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

try:
    from models import MAGVITv2, DyninOmniModelLM
    from training.prompting_utils import UniversalPrompting
    from training.utils import image_transform
except ImportError as e:
    warnings.warn(f"Failed to import MMaDA modules: {e}")
    # Retry with the local repository root explicitly prioritized.
    if repo_root in sys.path:
        sys.path.remove(repo_root)
    sys.path.insert(0, repo_root)
    from models import MAGVITv2, DyninOmniModelLM
    from training.prompting_utils import UniversalPrompting
    from training.utils import image_transform

from .utils import load_mmada_image, build_mmada_prompt, reorganize_mmada_prompt
from .dataset_configs import get_dataset_config, merge_configs, DEFAULT_KWARGS
from ..base import BaseModel
from ...dataset import DATASET_TYPE, DATASET_MODALITY
from ...smp import *

from .utils import (build_multi_choice_prompt,
                    build_mcq_cot_prompt,
                    build_qa_cot_prompt)


def _load_video_token_merger_weights(model, model_path: str) -> None:
    if not hasattr(model, "video_token_merger") or model.video_token_merger is None:
        return
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    safetensors_path = os.path.join(model_path, "model.safetensors")
    state_dict = {}
    try:
        from safetensors.torch import load_file
    except Exception as exc:
        warnings.warn(f"safetensors not available; cannot load video_token_merger weights: {exc}")
        return

    if os.path.isfile(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
        except Exception as exc:
            warnings.warn(f"Failed to read safetensors index: {exc}")
            return
        weight_map = index.get("weight_map", {})
        shard_files = {
            fname for key, fname in weight_map.items() if key.startswith("video_token_merger.")
        }
        for shard in shard_files:
            shard_path = os.path.join(model_path, shard)
            try:
                tensors = load_file(shard_path, device="cpu")
            except Exception as exc:
                warnings.warn(f"Failed to load shard {shard_path}: {exc}")
                continue
            for key, value in tensors.items():
                if key.startswith("video_token_merger."):
                    state_dict[key] = value
    elif os.path.isfile(safetensors_path):
        try:
            tensors = load_file(safetensors_path, device="cpu")
        except Exception as exc:
            warnings.warn(f"Failed to load {safetensors_path}: {exc}")
            return
        for key, value in tensors.items():
            if key.startswith("video_token_merger."):
                state_dict[key] = value
    else:
        warnings.warn("No safetensors weights found for video_token_merger.")
        return

    if not state_dict:
        warnings.warn("No video_token_merger weights found in checkpoint.")
        return
    merger_state = {
        key.replace("video_token_merger.", ""): value
        for key, value in state_dict.items()
        if key.startswith("video_token_merger.")
    }
    missing, unexpected = model.video_token_merger.load_state_dict(merger_state, strict=False)
    if missing:
        warnings.warn(f"Missing video_token_merger weights: {missing}")
    if unexpected:
        warnings.warn(f"Unexpected video_token_merger weights: {unexpected}")


class DyninOmni(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    
    def __init__(self,
                 model_path='./work_dirs/mmada/',
                 tokenizer_path=None,
                 vq_model_path=None,
                 vq_model_type='magvitv2',
                 resolution=256,  # Changed to match training default
                 max_new_tokens=16,
                 steps=16,
                 block_length=16,
                 temperature=0.8,
                 top_k=1,
                 use_config_file=True,
                 custom_configs=None,
                 vid_merge: bool = False,
                 vid_merge_stride: int = 2,
                 **kwargs):
        self.use_cot = (os.getenv('USE_COT') == '1')
        print(f"use_cot: {self.use_cot}")
        self.cot_prompt = "You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here"
        
        self.model_path = model_path
        self.resolution = resolution
        self.max_new_tokens = max_new_tokens
        self.steps = steps
        self.block_length = block_length
        self.temperature = temperature
        self.top_k = top_k
        self.use_config_file = use_config_file
        self.custom_configs = custom_configs or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if tokenizer_path is None:
            tokenizer_path = model_path
        
        try:
            tok_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                padding_side="left",
                trust_remote_code=True,
                config=tok_config,
            )
        except Exception as e:
            warnings.warn(f"Failed to load tokenizer from {tokenizer_path}: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                padding_side="left",
                trust_remote_code=True,
            )
        
        self.uni_prompting = UniversalPrompting(
            self.tokenizer, 
            max_text_len=2048,
            special_tokens=(
                                        "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>",
                                        "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>", 
                                        "<|i2i|>", "<|ti2ti|>", "<|v2t|>", "<|v2s|>", "<|s2t|>", "<|t2s|>", "<|s2s|>", "<|soa|>", "<|eoa|>", "<think>", "</think>"
                                    ),
            ignore_id=-100, 
            cond_dropout_prob=0.0, 
            use_reserved_token=True
        )
        
        if vq_model_type == "magvitv2":
            if vq_model_path is None:
                vq_model_path = "multimodalart/MAGVIT2"
            
            self.vq_model = MAGVITv2.from_pretrained(vq_model_path).to(self.device)
            self.vq_model.requires_grad_(False)
            self.vq_model.eval()
        else:
            raise ValueError(f"VQ model type {vq_model_type} not supported.")
        
        # self.model = MMadaModelLM.from_pretrained(
        #     model_path, 
        #     trust_remote_code=True, 
        #     torch_dtype=torch.bfloat16
        # ).to(self.device)
        config_path = os.path.join(model_path, "config.json")
        if not os.path.isfile(config_path):
            warnings.warn(f"[ERROR] Config file not found at {config_path}, loading model without config.")
            config_path = None
        self.model = DyninOmniModelLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            config=config_path,
        ).to(self.device)
        if vid_merge and hasattr(self.model, "enable_video_merge"):
            self.model.enable_video_merge(int(vid_merge_stride))
            _load_video_token_merger_weights(self.model, model_path)
        elif vid_merge:
            warnings.warn("vid_merge requested but model does not support enable_video_merge().")
        self.model.eval()
        self.mask_token_id = self.model.config.mask_token_id if hasattr(self.model.config, 'mask_token_id') else None
        
        warnings.warn(f'Dynin-Omni initialized with model_path: {model_path}')

    def get_generation_kwargs(self, dataset=None):
        base_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "steps": self.steps,
            "block_length": self.block_length,
            "temperature": self.temperature,
            "top_k": self.top_k
        }
        min_new_tokens = os.getenv("MIN_NEW_TOKENS")
        if min_new_tokens is not None:
            try:
                base_kwargs["min_new_tokens"] = int(min_new_tokens)
            except ValueError:
                pass
        if os.getenv("DEBUG_GEN_KWARGS") == "1":
            print(f"Generation_kwargs({dataset}): {base_kwargs}")
        
        if not self.use_config_file:
            if dataset in self.custom_configs:
                base_kwargs.update(self.custom_configs[dataset])
            return base_kwargs
        
        if dataset is None:
            return base_kwargs
            
        dataset_config = get_dataset_config(dataset)
        final_config = merge_configs(
            DEFAULT_KWARGS,
            dataset_config,
            self.custom_configs.get(dataset, {})
        )
        
        return final_config

   
    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if dataset in [
            'atomic_dataset', 'electro_dataset', 'mechanics_dataset',
            'optics_dataset', 'quantum_dataset', 'statistics_dataset'
        ]:
            return False
        if listinstr(['MMDU', 'MME-RealWorld', 'MME-RealWorld-CN', 'WeMath_COT', 'MMAlignBench'], dataset):
            # For Multi-Turn we don't have custom prompt
            return False
        if DATASET_MODALITY(dataset) == 'VIDEO':
            # For Video benchmarks we don't have custom prompt at here
            return False
        else:
            return True
        return True

    def build_prompt(self, line, dataset=None):
        """Build prompt for MMaDA evaluation"""
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        
        tgt_path = self.dump_image(line, dataset)
        
        if dataset is not None and DATASET_TYPE(dataset) == 'Y/N':
            question = line['question']
            if listinstr(['MME'], dataset):
                prompt = question + ' Answer the question using a single word or phrase.'
            elif listinstr(['HallusionBench', 'AMBER'], dataset):
                prompt = question + ' Please answer yes or no. Answer the question using a single word or phrase.'
            else:
                prompt = question
        elif dataset is not None and DATASET_TYPE(dataset) == 'MCQ':
            prompt = build_multi_choice_prompt(line, dataset)
            if os.getenv('USE_COT') == '1':
                prompt = build_mcq_cot_prompt(line, prompt, self.cot_prompt)
        elif dataset is not None and DATASET_TYPE(dataset) == 'VQA':
            question = line['question']
            if listinstr(['LLaVABench', 'WildVision'], dataset):
                prompt = question + '\nAnswer this question in detail.'
            elif listinstr(['OCRVQA', 'TextVQA', 'ChartQA', 'DocVQA', 'InfoVQA', 'OCRBench',
                            'DUDE', 'SLIDEVQA', 'GQA', 'MMLongBench_DOC'], dataset):
                prompt = question + '\nAnswer the question using a single word or phrase.'
            elif listinstr(['MathVista', 'MathVision', 'VCR', 'MTVQA', 'MMVet', 'MathVerse',
                            'MMDU', 'CRPE', 'MIA-Bench', 'MM-Math', 'DynaMath', 'QSpatial',
                            'WeMath', 'LogicVista', 'MM-IFEval', 'ChartMimic'], dataset):
                prompt = question
                if os.getenv('USE_COT') == '1':
                    prompt = build_qa_cot_prompt(line, prompt, self.cot_prompt)
            else:
                prompt = question + '\nAnswer the question using a single word or phrase.'
        else:
            # VQA_ex_prompt: OlympiadBench, VizWiz
            prompt = line['question']
            if os.getenv('USE_COT') == '1':
                prompt = build_qa_cot_prompt(line, prompt, self.cot_prompt)

        print(prompt)
        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])

        return message

    def set_max_num(self, dataset):
        """Set maximum number of images based on dataset"""
        self.total_max_num = 16  # Conservative limit for MMaDA
        if dataset is None:
            self.max_num = 1  # MMaDA typically works with single images
            return None
        
        if DATASET_MODALITY(dataset) == 'VIDEO':
            # Allow multiple frames for video evaluation.
            self.max_num = self.total_max_num
        else:
            self.max_num = 1  # Start with single image support

    @torch.no_grad()
    def generate_mmada(self, message, dataset=None):
        """Generate response using MMaDA model"""
        image_num = len([x for x in message if x['type'] == 'image'])
        
        if image_num == 0:
            prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
            return "I need an image to provide a meaningful response."
        
        image_paths = [x['value'] for x in message if x['type'] == 'image']
        prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
        is_video = dataset is not None and DATASET_MODALITY(dataset) == 'VIDEO'
        
        if is_video:
            # Video: stack frames and use v2t tokenization path (matches training/inference).
            frames = []
            for path in image_paths:
                try:
                    img = Image.open(path).convert("RGB")
                except Exception as e:
                    warnings.warn(f"Failed to load frame from {path}: {e}")
                    img = Image.new("RGB", (self.resolution, self.resolution), (255, 255, 255))
                frames.append(image_transform(img, resolution=self.resolution))
            if len(frames) == 0:
                return "I need video frames to provide a meaningful response."
            video_tensor = torch.stack(frames).to(self.device)  # (T, C, H, W)
            video_tokens = self.vq_model.get_code(video_tensor) + len(self.uni_prompting.text_tokenizer)
            video_tokens = video_tokens.view(1, -1)
        else:
            if image_num > 1:
                warnings.warn(f"Multiple images ({image_num}) detected, using the first one.")
            image_path = image_paths[0]
            try:
                image_ori = Image.open(image_path).convert("RGB")
            except Exception as e:
                warnings.warn(f"Failed to load image from {image_path}: {e}")
                image_ori = Image.new("RGB", (self.resolution, self.resolution), (255, 255, 255))
            
            image = image_transform(image_ori, resolution=self.resolution).to(self.device)
            image = image.unsqueeze(0)
            image_tokens = self.vq_model.get_code(image) + len(self.uni_prompting.text_tokenizer)
        
        input_text = build_mmada_prompt(prompt, self.uni_prompting.sptids_dict)
        input_ids = self.uni_prompting.text_tokenizer([input_text])['input_ids']
        input_ids = torch.tensor(input_ids).to(self.device)
        
        # Construct the full input
        input_ids = torch.cat([
            (torch.ones(input_ids.shape[0], 1) * self.uni_prompting.sptids_dict['<|v2t|>' if is_video else '<|mmu|>']).to(self.device),
            (torch.ones(input_ids.shape[0], 1) * self.uni_prompting.sptids_dict['<|soi|>']).to(self.device),
            video_tokens if is_video else image_tokens,
            (torch.ones(input_ids.shape[0], 1) * self.uni_prompting.sptids_dict['<|eoi|>']).to(self.device),
            (torch.ones(input_ids.shape[0], 1) * self.uni_prompting.sptids_dict['<|sot|>']).to(self.device),
            input_ids
        ], dim=1).long()
        
        generation_kwargs = self.get_generation_kwargs(dataset)
        
        if dataset:
            warnings.warn(f"Using generation config for {dataset}: {generation_kwargs}")
        
        output_ids = self.model.mmu_generate(
            input_ids, 
            **generation_kwargs  
        )
        
        response_text = self.uni_prompting.text_tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1]:], 
            skip_special_tokens=True
        )[0]
        
        response_text = self.post_process_response(response_text, dataset)
        print(response_text)
        return response_text.strip()

    def post_process_response(self, response, dataset=None):
        if dataset is None:
            return response
            
        if DATASET_TYPE(dataset) == 'Y/N':
            response_lower = response.lower()
            if 'yes' in response_lower and 'no' not in response_lower:
                return 'Yes'
            elif 'no' in response_lower and 'yes' not in response_lower:
                return 'No'
            elif response_lower.strip().startswith('yes'):
                return 'Yes'
            elif response_lower.strip().startswith('no'):
                return 'No'
                
        elif DATASET_TYPE(dataset) == 'MCQ':
            import re
            matches = re.findall(r'\b([A-E])\b', response)
            if matches:
                return matches[-1]
                
        if listinstr(['MME'], dataset):
            if len(response) > 10:
                words = response.split()
                if len(words) > 3:
                    return ' '.join(words[:3])
                    
        return response

    def generate_inner(self, message, dataset=None):
        """Main generation function called by VLMEvalKit"""
        self.set_max_num(dataset)
        return self.generate_mmada(message, dataset) 
