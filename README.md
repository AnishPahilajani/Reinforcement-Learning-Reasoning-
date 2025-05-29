# 🧪 Reinforcement Fine-Tuning with GRPO + LoRA/QLoRA on Qwen and DeepSeek-R1

This repository contains two experimental notebooks focused on fine-tuning large language models (LLMs) for math reasoning using [🤗 TRL](https://github.com/huggingface/trl), [LoRA/QLoRA](https://github.com/huggingface/peft), and [GRPO (Generalized Reinforcement Preference Optimization)](https://huggingface.co/docs/trl/main/en/grpo).

---

## 📁 Contents

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

## 🎯 Objective

I aim to train instruction-following LLMs to reason through multi-step math problems and output in a consistent, machine-readable format:

<reasoning>
Step-by-step explanation...
</reasoning>
<answer>
Final numeric answer
</answer>

---
