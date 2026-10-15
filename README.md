<div align="center">

# Multilingual Safety Alignment via Self-Distillation

**NeurIPS 2026**

Ruiyang Qin\* · Qingzhuo Wang\* · Dongrui Liu · Qiang Li · Zhihua Wei · Wen Shen

Tongji University · Shanghai AI Laboratory  
<sub>\* Equal contribution</sub>

[![Paper](https://img.shields.io/badge/arXiv-2605.02971-b31b1b.svg)](https://arxiv.org/abs/2605.02971)
[![Conference](https://img.shields.io/badge/NeurIPS-2026-6846A5.svg)](https://neurips.cc/virtual/2026/loc/sydney/poster/151163)
[![GitHub](https://img.shields.io/badge/GitHub-MSD-181717.svg?logo=github)](https://github.com/weiliang822/MSD)

[Overview](#overview) · [Quick Start](#quick-start) · [Data](#data) · [Models](#models) · [Evaluation](#evaluation) · [Citation](#citation)

</div>

## Overview

**Multilingual Self-Distillation (MSD)** transfers an LLM's safety capabilities from high-resource to low-resource languages using parallel multilingual queries. The teacher and student start from the same model: the student receives the target-language query, while the frozen teacher also receives its English translation and instructions to reason about safety. Training requires no externally supplied responses.

<p align="center">
  <img src="assets/overview.svg" alt="Figure 2: MSD with student or teacher response sampling and Dual-Perspective Safety Weighting" width="100%">
</p>

MSD supports **on-policy** sampling from the student and **off-policy** sampling from the teacher. **Dual-Perspective Safety Weighting (DPSW)** combines teacher confidence with student disagreement to emphasize safety-critical tokens in the distillation loss. At inference time, the trained student receives only the original query.

## Quick Start

Use **Linux, Python 3.12, and an NVIDIA GPU**. Our training setup uses a single **RTX PRO 6000 (96 GB)**. The default configuration performs full fine-tuning with BF16 and colocated vLLM generation.

```bash
git clone https://github.com/weiliang822/MSD.git
cd MSD
conda create -n msd python=3.12 -y
conda activate msd
pip install -r requirements.txt
```

Prepare the datasets as described in [Data](#data). For example, start Qwen3-8B training with:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_qwen3_8b.sh
```

Each model has a dedicated training script with its training budget listed in [Models](#models). Additional arguments can be appended to the command.

For off-policy MSD, use `--sampling_policy off_policy` and a separate output directory. The final model and tokenizer are saved to `<output_dir>/final`; intermediate checkpoints are saved every 400 steps.

| Setting | Default |
| :--- | :--- |
| Learning rate | `2e-5` |
| Batch size / gradient accumulation | `1` / `32` |
| Prompt / completion length | `512` / `512` tokens |
| Sampling temperature | `1.0` |
| Distillation objective | Reverse KL with DPSW, top-K = `32` |
| Teacher | Frozen copy of the initial model |

<details>
<summary>Training options</summary>

- Set `--alpha 0` for forward KL or `--alpha 0.5` for generalized Jensen-Shannon divergence; `--alpha 1` selects reverse KL.
- Use `--use_peft --lora_r 16 --lora_alpha 32` for LoRA fine-tuning. The final directory then contains an adapter.
- Use `--no-use_vllm` for Transformers generation, and `--precision fp16` on GPUs without BF16 support.
- Resume with `--resume_from_checkpoint outputs/msd-qwen3-8b/checkpoint-400`.
- The trainer retains multimodal processor support for custom datasets. The supplied CLI and benchmarks use text queries.

Run `python main.py --help` for all command-line options. Logging stays local by default; install `wandb` and add `--report_to wandb` to enable it.

</details>

## Data

| Source | Use |
| :--- | :--- |
| [XSafety](https://github.com/Jarviswang94/Multilingual_safety_benchmark) | Training |
| [MultiJail](https://github.com/DAMO-NLP-SG/multilingual-safety-for-LLMs) | Safety evaluation |
| [PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF) | Safety evaluation |

Download the datasets from the sources above, process them according to the experimental settings in the paper, and place the prepared CSV files in `data/`: `train.csv` for training, and `MultiJail.csv` and `PKU-SafeRLHF.csv` for evaluation.

Each row should contain aligned translations of the same query. The training CSV uses `en` for teacher context and `zh, ar, bn, de, fr, hi, ja, ru, sp` for the target languages (`sp` denotes Spanish). Evaluation CSVs require a unique `id` and the language columns `en, zh, it, vi, ar, ko, th, bn, sw, jv`.

## Models

MSD starts from the following Hugging Face backbones. Training budgets are measured in optimizer steps and are set by the corresponding scripts.

| Model | Hugging Face | Training steps | Training script |
| :--- | :--- | ---: | :--- |
| Qwen3-8B | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | 300 | [`train_qwen3_8b.sh`](scripts/train_qwen3_8b.sh) |
| Qwen2.5-7B-Instruct | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | 60 | [`train_qwen2_5_7b.sh`](scripts/train_qwen2_5_7b.sh) |
| Llama-3-8B-Instruct | [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | 20 | [`train_llama3_8b.sh`](scripts/train_llama3_8b.sh) |
| Llama-2-7B-Chat | [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) | 20 | [`train_llama2_7b.sh`](scripts/train_llama2_7b.sh) |

## Evaluation

**1. Generate responses** on both safety benchmarks:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_generate.sh \
  --model_path outputs/msd-qwen3-8b/final \
  --model_name msd-qwen3-8b \
  --datasets multijail,pku \
  --output_dir results
```

Generation uses each model's chat template, greedy decoding, and up to 1,024 new tokens. Use `--languages en,zh` for a language subset or `--tensor_parallel_size 2` for two-GPU inference. For LoRA checkpoints, pass the original backbone as `--model_path` and the adapter directory as `--adapter_path`.

**2. Judge response safety** with GPT-4o:

```bash
read -rsp "OpenAI API key: " OPENAI_API_KEY; echo
export OPENAI_API_KEY
python scripts/run_safety_judge.py \
  --results_dir results \
  --model_name msd-qwen3-8b \
  --judge_model gpt-4o
```

The judge assigns `safe`, `unsafe`, or `invalid`. Following the paper, **ASR (%) = 100 × (unsafe + invalid) / total**; lower is better. API errors are reported separately, and complete ASR is produced only after every response has a valid judgment.

Responses and judgments are saved under `results/<dataset>/<model>/<language>/`. `results/safety_summary.json` contains per-language results and the mean ASR over the evaluated languages. Both stages can be rerun: matching generation outputs are reused, and judging retries failed or changed records. `OPENAI_BASE_URL` optionally selects a compatible API endpoint.

## Citation

```bibtex
@inproceedings{qin2026multilingual,
  title     = {Multilingual Safety Alignment via Self-Distillation},
  author    = {Qin, Ruiyang and Wang, Qingzhuo and Liu, Dongrui and Li, Qiang and Wei, Zhihua and Shen, Wen},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  url       = {https://arxiv.org/abs/2605.02971}
}
```
