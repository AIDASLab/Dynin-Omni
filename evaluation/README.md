# Dynin-Omni Evaluation

## 1. VLM Evaluation

`VLMEvalKit` provides VLM evaluation, including multimodal understanding tasks such as video-to-text and image-to-text.

### 1.1 Installation

```bash
cd evaluation/VLMEvalKit
pip install -e .
```

<br>

### 1.2 Model Path Configuration

Model paths are defined in `evaluation/VLMEvalKit/vlmeval/config.py` as follows:

```python
dynin_omni = {
    "DyninOmni": partial(
        DyninOmni, 
        model_path="snu-aidas/Dynin-Omni",
        tokenizer_path="snu-aidas/Dynin-Omni",
        vq_model_path="snu-aidas/magvitv2",
        vq_model_type="magvitv2",
        resolution=480,
    ),
}
```

<br>

### 1.3 Dataset Configuration

`max_new_tokens`, `steps`, and `block_length` are configured per dataset in `evaluation/VLMEvalKit/vlmeval/vlm/dynin_omni/dataset_configs.py`. Example:

```python
DATASET_CONFIGS = {
    "MathVista_MINI": {
        "max_new_tokens": 96,
        "steps": 96,
        "block_length": 48,
    },

    "MathVerse_MINI_Vision_Only": {
        "max_new_tokens": 256,
        "steps": 128,
        "block_length": 32,
    },

    "MMVet": {
        "max_new_tokens": 512,
        "steps": 256,
        "block_length": 128,
    },
}
```

<br>

### 1.4 VLM Evaluation Execution

VLM evaluation is executed with the provided scripts:

```bash
cd evaluation

# Single-GPU configuration
bash scripts/vlmeval.sh --gpu 0 --data MMStar --model DyninOmni

# Multi-GPU configuration
bash scripts/vlmeval.sh --gpu 0,1,2,3 --data MMMU_DEV_VAL --model DyninOmni

# Full sweep across supported datasets and models
bash scripts/vlmeval.sh --gpu 0,1,2,3 --data all --model all
```

<br>

## 2. LLM Evaluation

`lm-eval` is used for evaluating text-only LLM capability through Dynin-Omni Model.

### 2.1 Installation

```bash
pip install -U lm-eval accelerate
```

<br>

### 2.2 LM Eval Execution

All evaluations are executed via `scripts/lmeval.sh` for consistent deployment and benchmarking.

```bash
cd evaluation

bash scripts/lmeval.sh \
  --model-path /path/to/unwrapped_model

# Example: task specification + output path
bash scripts/lmeval.sh \
  --model-path /path/to/unwrapped_model \
  --task hendrycks_math500 \
  --num-fewshot 0 \
  --gen-length 1024 \
  --block-length 16 \
  --output-dir ./results/lm

# Example: fast-dLLM v1 path
bash scripts/lmeval.sh \
  --model-path /path/to/unwrapped_model \
  --param use_fastdllm_v1=True \
  --param factor=3.0
```

<br>

## 3. Image Generation Evaluation

`GenEval` and `DPGBench (ELLA)` are used for evaluating text-to-image generation capability.

### 3.1 Installation

```bash
cd evaluation

# clone GenEval
git clone https://github.com/djghosh13/geneval.git

# clone DPGBench
git clone https://github.com/TencentQQGYLab/ELLA.git
```

<br>

### 3.2 GenEval and DPGBench Execution

Benchmark-specific procedures are documented in [GenEval](https://github.com/djghosh13/geneval) and [DPGBench](https://github.com/TencentQQGYLab/ELLA).

<br>

## 4. Image Editing Evaluation

`ImgEdit` is used for evaluating image-editing capability (image-to-image).

### 4.1 Installation

```bash
cd evaluation
git clone https://github.com/PKU-YuanGroup/ImgEdit.git
```

<br>

### 4.2 ImgEdit Benchmark Execution

Benchmark-specific procedures are documented in [ImgEdit](https://github.com/PKU-YuanGroup/ImgEdit).

<br>

## 5. Speech Evaluation

`Libri-Light` and `Seed-TTS-Eval` are used for evaluating speech capability, including ASR and TTS.

### 5.1 Installation

```bash
cd evaluation

# clone Libri-light
git clone https://github.com/facebookresearch/libri-light.git

# clone Seed-TTS-Eval
git clone https://github.com/BytedanceSpeech/seed-tts-eval.git
```

<br>

### 5.2 Libri-Light and Seed-TTS-Eval Execution

Benchmark-specific procedures are documented in [Libri-Light](https://github.com/facebookresearch/libri-light) and [Seed-TTS-Eval](https://github.com/BytedanceSpeech/seed-tts-eval).
