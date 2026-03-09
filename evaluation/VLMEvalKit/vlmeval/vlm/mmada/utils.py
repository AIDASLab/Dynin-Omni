import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
import pandas as pd
import string
from ..base import BaseModel
from ...dataset import DATASET_TYPE, DATASET_MODALITY
from ...dataset.dynamath import preprocess
from ...smp import *


def image_transform(image, resolution=256, normalize=True):
    """Transform image for MMaDA model - matches original training/utils.py implementation"""
    image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC)(image)
    image = transforms.CenterCrop((resolution, resolution))(image)
    image = transforms.ToTensor()(image)
    if normalize:
        image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
    return image

def image_transform_squash(image, resolution=256, normalize=True):
    """Alternative transform that squashes image to fit resolution"""
    image = transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC)(image)
    image = transforms.ToTensor()(image)
    if normalize:
        image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
    return image

def load_mmada_image(image_path, resolution=512, device='cuda'):
    """Load and process image for MMaDA"""
    image = Image.open(image_path).convert("RGB")
    image_tensor = image_transform(image, resolution=resolution)
    return image_tensor.unsqueeze(0).to(device)

def get_vq_model_class(model_type):
    """Get VQ model class based on type"""
    if model_type == "magvitv2":
        try:
            from models import MAGVITv2
            return MAGVITv2
        except ImportError:
            # Try alternative import path
            sys.path.append('/path/to/MMaDA')
            from models import MAGVITv2
            return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")

def build_mmada_prompt(question, special_tokens_dict=None):
    """Build prompt for MMaDA model"""
    # Based on infer_mmu.py line 90
    input_text = f'<|start_header_id|>user<|end_header_id|>\n{question}<eot_id><|start_header_id|>assistant<|end_header_id|>\n'
    return input_text

def reorganize_mmada_prompt(message, image_num=1):
    """Reorganize prompt for MMaDA similar to dmllm utils"""
    if image_num == 1:
        # Simple case with one image
        prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
        return prompt
    else:
        # Multiple images case
        prompt = ''
        image_idx = 1
        for x in message:
            if x['type'] == 'text':
                prompt += x['value']
            elif x['type'] == 'image':
                prompt += f'<Image-{image_idx}>'
                image_idx += 1
        return prompt 


def build_qa_cot_prompt(line, prompt, cot_prompt=None):
    if cot_prompt is None:
        cot_prompt = (
            "Answer the preceding question. The last line of your response should follow this format: "
            "'Answer: \\boxed{$FINAL_ANSWER}' (without quotes), where 'FINAL_ANSWER' is your conclusion "
            "based on the reasoning provided. If you are uncertain or the problem is too complex, make "
            "a reasoned guess based on the information provided. Avoid repeating steps indefinitely—"
            "provide your best guess even if unsure. Think step by step logically, considering all "
            "relevant information before answering."
        )
    prompt = prompt + '\n' + cot_prompt

    return prompt


def build_multi_choice_prompt(line, dataset=None):
    question = line['question']
    hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
    if hint is not None:
        question = hint + '\n' + question

    options = {
        cand: line[cand]
        for cand in string.ascii_uppercase
        if cand in line and not pd.isna(line[cand])
    }
    for key, item in options.items():
        question += f'\n{key}. {item}'
    prompt = question

    if len(options):
        prompt += '\n请直接回答选项字母。' if cn_string(
            prompt) else "\nAnswer with the option's letter from the given choices directly."
    else:
        prompt += '\n请直接回答问题。' if cn_string(prompt) else '\nAnswer the question directly.'

    return prompt


def build_mcq_cot_prompt(line, prompt, cot_prompt=None):
    if cot_prompt is None:
        cot_prompt = (
            "Answer the preceding multiple choice question. The last line of your response should follow "
            "this format: 'Answer: \\boxed{$LETTER}' (without quotes), where LETTER is one of the options. "
            "If you are uncertain or the problem is too complex, make a reasoned guess based on the "
            "information provided. Avoid repeating steps indefinitely—provide your best guess even if "
            "unsure. Think step by step logically, considering all relevant information before answering."
        )
    prompt = prompt.replace("Answer with the option's letter from the given choices directly.", '').strip()
    prompt = prompt + '\n' + cot_prompt

    return prompt


def reorganize_prompt(message, image_num, dataset=None):
    if dataset is not None and listinstr(['MUIRBench'], dataset):
        prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
        images_to_remove = ' '.join(['<image>'] * image_num)
        prompt = prompt.replace(images_to_remove, '')
        for i in range(image_num):
            prompt = prompt.replace('<image>', f'<Image-{i + 1}>', 1)
        prompt = ''.join([f'Image-{i + 1}: <image>\n' for i in range(image_num)]) + prompt
    elif image_num == 1:
        prompt = '<image>\n' + '\n'.join([x['value'] for x in message if x['type'] == 'text'])
    else:
        prompt, image_idx = '', 1
        for x in message:
            if x['type'] == 'text':
                prompt += x['value']
            elif x['type'] == 'image':
                prompt += f'<Image-{image_idx}>'
                image_idx += 1
        prompt = ''.join([f'Image-{i + 1}: <image>\n' for i in range(image_num)]) + prompt
        images_to_remove = ''.join([f'<Image-{i + 1}>' for i in range(image_num)])
        prompt = prompt.replace(images_to_remove, '')
    return prompt