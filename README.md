# 🧪 Reinforcement Fine-Tuning with GRPO + LoRA/QLoRA on Qwen inspired from DeepSeek-R1

This repository contains two experimental notebooks focused on fine-tuning large language models (LLMs) and Visual Language models(VLMs) for math reasoning and chart data using [🤗 TRL](https://github.com/huggingface/trl), [LoRA/QLoRA](https://github.com/huggingface/peft), and [GRPO (Generalized Reinforcement Preference Optimization)](https://huggingface.co/docs/trl/main/en/grpo) and SFT (Supervised Finetuneing). 

---

## 📁 Contents
### `lora-qwen-vl-trl-quantization.ipynb`
"""# Fine-Tuning Qwen2.5-VL-3B with QLoRA and TRL (Quantized)

## Overview
This notebook demonstrates how to fine-tune the [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) model — a vision-language model — using the LoRA (Low-Rank Adaptation) technique, in combination with quantization-aware training via BitsAndBytes 4-bit inference and the Hugging Face `trl` library (`SFTTrainer`).

## Key Features

- **Model:** `Qwen/Qwen2.5-VL-3B-Instruct` (a multimodal model for visual question answering)
- **Training Method:** Supervised Fine-Tuning (SFT) using `SFTTrainer` from Hugging Face TRL
- **Parameter-Efficient Fine-Tuning:** Utilizes **LoRA** via `peft`
- **Quantization:** Applies 4-bit quantization using `BitsAndBytes` (`bnb_4bit`)
- **Dataset:** Uses a small 1% subset of [ChartQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA)
- **Input Format:** Multimodal prompts — image + text query → textual answer
- **Device:** Optimized for CUDA (GPU)

## Techniques Used

- ✅ **LoRA** for memory-efficient fine-tuning (`peft.LoraConfig`)
- ✅ **QLoRA-style setup** using 4-bit quantization (`BitsAndBytesConfig`)
- ✅ Hugging Face `transformers`, `trl`, and `datasets` ecosystem
- ✅ Multi-turn chat-style formatting for VLM data with `"system"`, `"user"`, and `"assistant"` roles

## Dependencies Installed

- `trl`, `peft`, `transformers`, `bitsandbytes`, `vllm`, `triton`
- Uses `uv pip` to manage system installs in a Kaggle-like environment
"""


### `sft-trl-qlora-qwen.ipynb`
- Supervised fine-tuning (SFT) of **Qwen models** using QLoRA and TRL’s `SFTTrainer`.
- Prepares the base model for structured math output (e.g., XML tags like `<answer>`).
- Designed to bootstrap downstream GRPO reward-based training.

### `trl_LoRa_grpo_qwen_0_5b.ipynb`
- Trains a **Qwen2.5** model using `GRPOTrainer`.
- Incorporates reward functions to encourage XML-style reasoning:
  - Presence of `<answer>` and `<reasoning>` tags
  - Correct numeric outputs
  - Structured format compliance

---
