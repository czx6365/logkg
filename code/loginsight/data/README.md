# LogInsight Data Directory

This directory stores **generated experiment artifacts**, not source datasets.

Large JSONL files are intentionally excluded from Git so the repository stays lightweight and reproducible. The pipeline regenerates them from the configured input dataset.

## Regeneration flow

From `code/loginsight`:

```bash
python scripts/prepare_data.py --config configs/dataset.yaml
python scripts/build_instruction_data.py --config configs/base.yaml
```

Typical generated files include:

```text
data/processed/*_preprocessed*.jsonl
data/processed/*_fols.jsonl
data/instruction/*_instruction_all.jsonl
data/instruction/*_instruction_train.jsonl
data/instruction/*_instruction_val.jsonl
```

These paths are ignored by the root `.gitignore`.

Compact evaluation summaries such as curated CSV metrics may remain versioned when they are useful as research evidence.

## Data policy

- Do not commit raw/private industrial logs.
- Do not commit large generated JSONL intermediates.
- Keep configuration, preprocessing code, evaluation code, and small reproducibility summaries under version control.
- Store private or licensed datasets outside the repository and point `configs/dataset.yaml` to the local copy.
