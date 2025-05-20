# 🧪 Reinforcement Fine-Tuning with GRPO + LoRA/QLoRA on Qwen and DeepSeek-R1

This repository contains two experimental notebooks focused on fine-tuning large language models (LLMs) for math reasoning using [🤗 TRL](https://github.com/huggingface/trl), [LoRA/QLoRA](https://github.com/huggingface/peft), and [GRPO (Generalized Reinforcement Preference Optimization)](https://huggingface.co/docs/trl/main/en/grpo).

---

## 📁 Contents

### `sft-trl-qlora-qwen.ipynb`
- Supervised fine-tuning (SFT) of **Qwen models** using QLoRA and TRL’s `SFTTrainer`.
- Prepares the base model for structured math output (e.g., XML tags like `<answer>`).
- Designed to bootstrap downstream GRPO reward-based training.

### `grpo-trl-lora-qwen-deepseek-r1.ipynb`
- Trains a **Qwen2.5 or DeepSeek R1** model using `GRPOTrainer`.
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

⚠️ Current Status
The GRPO-based training in grpo-trl-lora-qwen-deepseek-r1.ipynb is currently facing a cold start problem:

The model does not yet generate outputs in the expected XML-style format (e.g., <answer>...</answer>).

As a result, most reward functions return zero, leading to no gradient updates and a training loss of 0.0.

We're actively experimenting with lenient reward shaping (e.g., partial credit for including <answer> tags) to bootstrap learning and improve signal flow.

✅ Reward functions in GRPO phase:
easy_reward_func: Rewards basic format usage

xmlcount_reward_func: Gives fractional credit for XML compliance

correctness_reward_func: Rewards exact matches to gold answers (currently unreachable)

int_reward_func: Rewards presence of numeric output

strict_format_reward_func / soft_format_reward_func: Enforces full XML structure (unused early)

