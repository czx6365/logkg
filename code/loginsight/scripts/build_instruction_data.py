from __future__ import annotations

import argparse
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_builder import build_instruction_records, split_instruction_records, summarize_label_distribution
from src.fols import run_fols_for_cases
from src.utils import load_jsonl, load_yaml, resolve_path, save_jsonl, set_seed



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 42)))

    paths = cfg["paths"]
    fols_cfg = cfg["fols"]
    instruction_cfg = cfg.get("instruction", {})

    processed_path = resolve_path(paths["processed_cases"], args.config.parent)
    fols_path = resolve_path(paths["fols_cases"], args.config.parent)
    all_instr_path = resolve_path(paths["instruction_all"], args.config.parent)
    train_instr_path = resolve_path(paths["instruction_train"], args.config.parent)
    val_instr_path = resolve_path(paths["instruction_val"], args.config.parent)

    cases = load_jsonl(processed_path)
    # 第一步先从完整日志中提取 FOLS 摘要。
    fols_cases = run_fols_for_cases(cases, fols_cfg)
    save_jsonl(fols_cases, fols_path)

    # 第二步把摘要结果改写成指令微调样本。
    records = build_instruction_records(fols_cases, instruction_cfg)
    train_records, val_records = split_instruction_records(
        records,
        val_ratio=float(instruction_cfg.get("val_ratio", 0.2)),
        seed=int(cfg.get("seed", 42)),
    )

    save_jsonl(records, all_instr_path)
    save_jsonl(train_records, train_instr_path)
    save_jsonl(val_records, val_instr_path)

    print(f"fols_cases={len(fols_cases)} saved_to={fols_path}")
    print(f"instruction_all={len(records)} saved_to={all_instr_path}")
    print(f"instruction_train={len(train_records)} dist={summarize_label_distribution(train_records)}")
    print(f"instruction_val={len(val_records)} dist={summarize_label_distribution(val_records)}")


if __name__ == "__main__":
    main()
