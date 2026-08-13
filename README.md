# Log Diagnosis Baseline Reproduction Workspace

**Research code for reproducing and adapting log-failure diagnosis baselines, including LogKG-style experiments and a lightweight LogInsight pipeline.**

This repository is best understood as an **experimental baseline workspace** used to study log preprocessing, failure classification, LLM-based diagnosis, LoRA adaptation, zero-shot prompting, and evaluation workflows.

> **Important scope note:** this repository is **not** presented as the implementation of LogRover. It contains baseline reproduction / adaptation work used to understand and compare log-diagnosis methods.

## Why This Repository Exists

Industrial CI and system logs are noisy, long, and heterogeneous. Before designing a new diagnosis framework, it is useful to reproduce representative baselines and understand where they fail.

This repository explores that baseline layer through several complementary paths:

```text
raw logs
   ↓
preprocessing / dataset adaptation
   ↓
log representation or summary
   ↓
classifier / LLM / LoRA / zero-shot diagnosis
   ↓
evaluation and error analysis
```

The emphasis is on **reproducibility, data adaptation, hierarchical classification, and comparable evaluation**, rather than on claiming the baseline methods themselves as novel.

## Repository Overview

```text
logkg/
├── code/
│   ├── LogKG_mobile_exp.ipynb
│   ├── LogKG_mobile_exp_OS.ipynb
│   ├── d1_adapter.py
│   ├── hierarchical_os_classifier.py
│   ├── logkg_d1_enhanced.py
│   ├── process/
│   ├── model/
│   ├── result/
│   └── loginsight/
│       ├── configs/
│       ├── data/
│       ├── scripts/
│       ├── src/
│       └── README.md
└── loginsight_zeroshot/
    ├── configs/
    ├── scripts/
    ├── src/
    └── CLASSIFICATION_FLOW.md
```

## 1. LogKG-Style Experiments

The `code/` directory contains notebooks and utilities for adapting log datasets to LogKG-style experiments.

Representative files include:

- `LogKG_mobile_exp.ipynb` — notebook-based experiment workflow;
- `LogKG_mobile_exp_OS.ipynb` — OS-log experiment variant;
- `d1_adapter.py` — dataset adaptation utility;
- `hierarchical_os_classifier.py` — hierarchical fault classification experiments;
- `logkg_d1_enhanced.py` — extended LogKG-oriented experimental code;
- `process/` — preprocessing adapters for different datasets;
- `result/` — saved experiment outputs.

These files are useful for understanding how a baseline must be modified when its expected input schema does not match a new log dataset.

## 2. Lightweight LogInsight Reproduction

The most structured subproject is [`code/loginsight/`](code/loginsight/README.md).

Its pipeline is:

```text
raw dataset
   ↓
case normalization
   ↓
FOLS-style log summarization
   ↓
instruction-data construction
   ↓
LoRA training / inference
   ↓
classification evaluation
   ↓
ablation analysis
```

The implementation supports:

- multiple dataset adapters;
- log cleanup and normalization;
- FOLS-style representative-line selection;
- instruction-data generation;
- LoRA fine-tuning;
- inference with parsed fault labels and explanations;
- micro / macro / weighted F1 evaluation;
- confusion-matrix export;
- ablations such as removing FOLS or replacing the clustering strategy;
- a single-case `LogInsightAgent` interface for interactive diagnosis.

See the detailed subproject documentation in [`code/loginsight/README.md`](code/loginsight/README.md).

## 3. Zero-Shot Diagnosis Track

The `loginsight_zeroshot/` directory contains a separate zero-shot path for studying diagnosis without supervised adapter training.

It includes:

- major / minor fault-classification configurations;
- prompt construction;
- preprocessing;
- single-run inference;
- agent-style diagnosis;
- evaluation scripts;
- a documented classification flow.

This track is useful for comparing **prompt-only inference** against adaptation methods that use LoRA or structured instruction data.

## Research Questions Explored

This workspace supports questions such as:

1. How much preprocessing is required before an LLM can reliably classify long logs?
2. Does log summarization preserve enough evidence for diagnosis?
3. How do zero-shot and LoRA-adapted approaches differ?
4. When does hierarchical classification help with major/minor fault labels?
5. Which preprocessing or summarization components contribute most to final F1?
6. How portable are published log-diagnosis methods across datasets with different schemas?

## Evaluation

The structured LogInsight path can export:

```text
micro F1
macro F1
weighted F1
validity rate
per-class metrics
confusion matrix
```

The repository also retains intermediate outputs and ablation files so that diagnosis behavior can be inspected beyond a single final score.

## Quick Start: LogInsight

The commands below are run from `code/loginsight`.

### Install dependencies

```bash
pip install torch transformers peft datasets scikit-learn pandas numpy pyyaml tqdm
```

For the optional embedding / retrieval baseline:

```bash
pip install sentence-transformers
```

### Prepare data

```bash
cd code/loginsight
python scripts/prepare_data.py --config configs/dataset.yaml
```

### Build summarized instruction data

```bash
python scripts/build_instruction_data.py --config configs/base.yaml
```

### Train LoRA

```bash
python scripts/train_lora.py --config configs/model.yaml
```

### Run inference and evaluation

```bash
python scripts/run_inference.py --config configs/base.yaml
python scripts/eval_all.py --config configs/base.yaml
```

### Run ablations

```bash
python scripts/run_ablation.py --config configs/base.yaml
```

## What This Repository Demonstrates

For research / graduate-application review, the repository is most useful as evidence of:

- baseline reproduction;
- adapting research code to new dataset schemas;
- log preprocessing and noise reduction;
- hierarchical failure classification;
- LLM prompting and structured output parsing;
- LoRA-based adaptation;
- zero-shot diagnosis;
- experiment configuration and ablation design;
- F1-based evaluation and confusion-matrix analysis;
- failure-diagnosis engineering beyond toy text classification.

## Research Integrity

The repository includes implementations inspired by or reproducing prior log-diagnosis methods. Those methods should be credited to their original authors.

The purpose of this workspace is to document **reproduction, adaptation, and evaluation work**. Novel research frameworks that use these methods as baselines should be documented and evaluated in their own repositories or manuscripts rather than conflated with this baseline code.