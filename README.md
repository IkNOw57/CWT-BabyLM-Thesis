<div align="center">

# CWT-BabyLM

### Contrastive Weight Tying for Sample-Efficient Language Model Pretraining

*Master's Thesis — MA Linguistics (Text Mining), Vrije Universiteit Amsterdam*

[![Paper](https://img.shields.io/badge/paper-arXiv%3A2309.08351-b31b1b.svg)](https://arxiv.org/abs/2309.08351)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](#requirements)
[![SLURM](https://img.shields.io/badge/scheduler-SLURM-orange.svg)](#running-on-a-slurm-cluster)

[Overview](#overview) •
[Results](#results) •
[Installation](#installation) •
[Usage](#usage) •
[Repository Structure](#repository-structure) •
[Citation](#citation)

</div>

---

## Overview

This repository contains the training and evaluation code accompanying the thesis
**["Learning with Less: Contrastive Weight Tying on the BabyLM Challenge"](https://github.com/inovdwouw/Sample-Efficient-Language-Modelling/blob/main/docs/thesis.pdf)**
(Ino van de Wouw, VU Amsterdam, 2025).

The project studies **headless language models**, models pretrained with
**Contrastive Weight Tying (CWT)** ([Godey et al., 2024](https://arxiv.org/abs/2309.08351)) instead of a
standard cross-entropy prediction head, on the
**[BabyLM Challenge](https://babylm.github.io/)**, a shared task focused on
sample-efficient pretraining from developmentally plausible, human-scale corpora
(≤ 100M words).

CWT removes the vocabulary projection head during pretraining and instead learns
representations via in-batch contrastive learning between input embeddings and
output hidden states. This eliminates the most computationally expensive part of
a standard forward pass, trading the vocabulary-sized softmax for a much cheaper
contrastive loss.

**Research question:** How does a headless (CWT) language model trained on the
BabyLM 10M/100M-word datasets compare to a standard prediction-headed model, and
under what conditions might it outperform its traditional counterpart?

Both **Masked Language Model (MLM, BERT-style)** and **Generative (GPT-style)**
architectures are trained and evaluated head-to-head against vanilla baselines on
the **GLUE** and **BLiMP** benchmarks.

## Results

*(600M and 2B pretraining-token checkpoints; full breakdown in the [thesis](https://github.com/inovdwouw/Sample-Efficient-Language-Modelling/blob/main/docs/thesis.pdf), Chapter 4)*

| Dataset | Architecture | GLUE avg. (headless) | GLUE avg. (vanilla) | Training speed-up (headless) |
|---|---|---|---|---|
| STRICT-SMALL (10M words) | MLM | 58.8% | 59.9% | **32% faster** |
| STRICT (100M words) | MLM | 70.98% | 71.51% | **34% faster** |
| STRICT (100M words) | GPT | 35.83% | 36.23% | **53% faster** |

**Key findings**

- Headless (CWT) models train **32–53% faster** than vanilla models at every
  scale tested, with the largest gains on GPT-style architectures.
- On GLUE, vanilla models retain a small edge, but the gap **narrows as training
  data increases** (from ~1.1 pp at 10M words to ~0.5 pp at 100M words) —
  headless models scale more favourably with data.
- On BLiMP, MLM architectures perform close to random chance regardless of
  training objective, while GPT-style models reach ~68% accuracy — suggesting
  autoregressive objectives acquire grammatical competence more effectively
  from constrained corpora than masked objectives do.
- Absolute performance limits appear to stem primarily from the **constrained
  size of the BabyLM corpora**, not from the headless architecture itself.

See the [thesis PDF](https://github.com/inovdwouw/Sample-Efficient-Language-Modelling/blob/main/docs/thesis.pdf) for full task-by-task results, standard
deviations across random seeds, and BLiMP phenomenon-level analysis.

## Repository Structure

```
.
├── readings/
│   └── thesis.pdf              # Full thesis write-up
├── scripts/
│   ├── preprocess.py           # Tokenize & pack a HF dataset for pretraining
│   ├── mlm_headless.py         # Pretrain an MLM (BERT-style), headless or vanilla
│   ├── gpt_headless.py         # Pretrain a GPT (decoder-only), headless or vanilla
│   ├── ft_gpt_headless.py      # Add a head to & fine-tune a headless GPT checkpoint
│   ├── hf_publisher.py         # Push a trained checkpoint to the HF Hub
│   ├── glue_finetuning.py      # Fine-tune / evaluate a checkpoint on GLUE
│   ├── glue_test.py            # Evaluate a checkpoint on GLUE (multi-GPU aware)
│   ├── publish_dataset.py      # Push a preprocessed dataset to the HF Hub
│   └── *.sh                    # SLURM submission scripts matching each .py file
├── configs/                    # Model / training configs (add your own JSON files here)
├── requirements.txt
└── README.md
```

> **Note:** The scripts are unmodified from the original experiment code and
> import from an `engine` package (data modules, Lightning tasks, model
> variants) that is not included in this snapshot of the repository. See
> [Requirements](#requirements) below.

## Installation

### Requirements

- Python ≥ 3.9
- CUDA ≥ 11.2 (for GPU training)
- The private `engine` package referenced by the training scripts
  (`engine.data`, `engine.tasks.*`, `engine.lit.lightning_module`, `engine.models.*`) —
  contact the author or check the main project repository for access.

### Setup

```bash
git clone <this-repository-url>
cd cwt-babylm
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

All scripts accept `--config` pointing to a JSON file with model/training
hyperparameters (see `configs/` for examples you should populate for your setup).
Every Python script has a matching `.sh` file for submission to a SLURM cluster.

### 1. Preprocess a dataset

```bash
python scripts/preprocess.py --config configs/preprocess_baby_lm_100M.json
```

Tokenizes and packs a Hugging Face dataset into fixed-length sequences, ready
for pretraining.

### 2. Pretrain an encoder (MLM / BERT-style)

```bash
python scripts/mlm_headless.py \
    --config configs/mlm_headless.json \
    --num_nodes <gpu-node-count> \
    --global_bs <accumulated-batch-size> \
    --gpu_bs <per-device-batch-size> \
    --dataset <preprocessed-output>.hf \
    --hf_tokenizer <tokenizer-name> \
    --hf_path <model-architecture-on-hf> \
    --model_max_seq_len <max-position-embeddings> \
    --run_name <run-name> \
    --saved_ckpt_path <checkpoint-output-dir>
```

Whether the model is trained **headless** (CWT) or **vanilla** is controlled
by the training objective specified in `--config`. Other useful flags:
`--accelerator` (`hf`, `xformers`, or `flash_attention`) and `--ckpt_every`.

### 3. Pretrain a decoder (GPT-style)

```bash
python scripts/gpt_headless.py \
    --config configs/gpt_headless_70m.json \
    --num_nodes <gpu-node-count> \
    --global_bs <accumulated-batch-size> \
    --gpu_bs <per-device-batch-size> \
    --dataset <preprocessed-output>.hf \
    --hf_tokenizer <tokenizer-name> \
    --hf_path <model-architecture-on-hf> \
    --model_max_seq_len <max-position-embeddings> \
    --run_name <run-name> \
    --saved_ckpt_path <checkpoint-output-dir>
```

> `--accelerator xformers` may error depending on floating-point precision;
> `flash_attention` or `hf` are the safer defaults.

### 4. Publish a checkpoint to the Hugging Face Hub

```bash
python scripts/hf_publisher.py \
    --hf_name <your-hf-id>/<model-name> \
    --model_ckpt <path-to>.ckpt \
    --mode mlm   # or: add_head, lm
```

Use `--mode add_head` when publishing a **headless GPT** checkpoint that needs
a randomly-initialised LM head to be able to generate text.

### 5. Fine-tune a headless GPT for generation

A headless GPT is not trained to predict tokens, so it needs a head added and
fine-tuned before it can generate text:

```bash
python scripts/ft_gpt_headless.py \
    --ckpt_path <headless-model>.ckpt \
    --config configs/gpt_vanilla_ft.json \
    --num_nodes <gpu-nodes> \
    --global_bs <accumulated-batch-size> \
    --gpu_bs <per-device-batch-size> \
    --dataset <preprocessed-output>.hf \
    --run_name <run-name> \
    --saved_ckpt_path <checkpoint-output-dir>
```

Then publish the fine-tuned checkpoint (no `add_head` needed this time):

```bash
python scripts/hf_publisher.py \
    --hf_name <your-hf-id>/<model-name> \
    --model_ckpt <path-to-finetuned>.ckpt \
    --mode lm
```

### 6. Evaluate on GLUE

```bash
python scripts/glue_finetuning.py \
    --model_ckpt <path-to>.ckpt \
    --mode lm \
    --train_batch_size 16 \
    --run_name <run-name>
```

`scripts/glue_test.py` is a multi-GPU-aware variant that also reports GPU
memory availability before loading the checkpoint.

### 7. Publish a preprocessed dataset

```bash
python scripts/publish_dataset.py
```

Uploads the folder in `dataset_storage/` to the Hugging Face Hub. Requires the
`HF_TOKEN` environment variable to be set.

## Running on a SLURM Cluster

All experiments were run on a SLURM-managed HPC cluster. Each Python script has
a matching shell script (e.g. `preprocess.sh` for `preprocess.py`) that loads
the required modules, activates the virtual environment, and submits the job:

```bash
sbatch scripts/preprocess.sh
```

Adjust the `#SBATCH` directives (partition, GPU count, wall time, notification
email) and `module load` lines to match your cluster's environment before
submitting.

## Citation

If you use this code, please cite both the thesis and the original CWT paper
it builds on:

```bibtex
@mastersthesis{vandewouw2025cwtbabylm,
  title  = {Learning with Less: Contrastive Weight Tying on the BabyLM Challenge},
  author = {van de Wouw, Ino},
  school = {Vrije Universiteit Amsterdam},
  year   = {2025},
  type   = {MA Thesis (Text Mining)}
}

@misc{godey2023headless,
  title         = {Headless Language Models: Learning without Predicting with Contrastive Weight Tying},
  author        = {Godey, Nathan and de la Clergerie, Éric and Sagot, Benoît},
  year          = {2023},
  eprint        = {2309.08351},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```

A machine-readable citation file is also provided in [`CITATION.cff`](CITATION.cff).

## Acknowledgments

Supervised by Prof. Dr. Antske Fokkens, with Dr. Pia Sommerauer as second
reader, at the Computational Linguistics and Text-Mining Lab, VU Amsterdam.
Built on the original [headless-lm](https://arxiv.org/abs/2309.08351) codebase
by Godey et al.

## Contact

Ino van de Wouw — inovandewouw@gmail.com

## License

Released under the [MIT License](LICENSE) unless otherwise noted (see
[NOTICE](#) for third-party components, e.g. the `engine` package, which may
carry their own license).
