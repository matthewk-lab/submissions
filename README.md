# Multi-Label Toxic Comment Classification

Deep learning coursework (Deep Learning Applications module) — detecting toxic online
comments across six labels using a ladder of models from a classical baseline to a
fine-tuned transformer, with responsible-AI bias analysis and explainability.

**Project option:** Hate Speech and Toxic Comment Classification on Social Media
**Dataset:** [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge)

## Task

Each comment may carry any subset of six **independent** labels:
`toxic, severe_toxic, obscene, threat, insult, identity_hate` (multi-label, not
multi-class). The data is heavily imbalanced (`toxic` ≈ 10%, `threat` ≈ 0.3%).

## Models (comparison ladder)

1. **TF-IDF + One-vs-Rest Logistic Regression** — classical baseline (full data).
2. **1D-CNN** and **BiLSTM** with pretrained GloVe embeddings — classic deep learning.
3. **DistilBERT** fine-tuned — transformer.

Evaluation on the full, correctly-filtered test set (63,978 scored rows) with per-label
ROC-AUC / PR-AUC, mean column-wise ROC-AUC (the official metric), and macro/micro-F1 at
validation-tuned thresholds. Includes an identity-term **bias probe** (per-subgroup false
positive rates) and **LIME**-based explainability.

## Setup

This project targets **Python 3.12** (the ML stack does not yet have reliable wheels for
newer interpreters). Using [`uv`](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Or with plain `pip`:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## Data

The dataset is **not** committed (large; competition rules). Fetch it with:

```bash
python download_data.py
```

This requires a Kaggle account that has accepted the competition rules and a Kaggle API
token at `~/.kaggle/kaggle.json`. Alternatively, download `train.csv`, `test.csv`, and
`test_labels.csv` manually from the competition "Data" tab, gzip them, and place them in
`./data/` as `train.csv.gz`, `test.csv.gz`, `test_labels.csv.gz`.

## Run

Open and run the notebook top-to-bottom:

```bash
jupyter notebook deep.ipynb
```

All figures and metrics are produced inline. Heavy training (transformer) runs on a
stratified subsample and is intended to be run with other memory-heavy apps (browser)
closed on a CPU-only machine.

## Reproducibility

- Pinned dependencies in `requirements.txt`.
- Global random seeds set in the notebook setup cell.
- Design rationale in [`docs/DESIGN.md`](docs/DESIGN.md).


# Submissions
