from __future__ import annotations

import argparse
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import build_processed_cases
from src.utils import load_yaml, resolve_path, save_jsonl



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    dataset_cfg = cfg.get("dataset", cfg)
    output_cfg = cfg.get("output", {})

    # 先把配置里的相对路径解析成绝对路径，避免从不同工作目录启动时出错。
    for key in ["os_preprocessed_dir", "openstack_jsonl_path", "hardware_csv_path", "custom_jsonl_path", "sample_jsonl_path"]:
        if key in dataset_cfg and str(dataset_cfg.get(key, "")).strip():
            dataset_cfg[key] = str(resolve_path(str(dataset_cfg[key]), args.config.parent))

    # 输出是统一 schema 的 processed_cases，供后续 FOLS/训练/推理复用。
    records = build_processed_cases(dataset_cfg)

    out_path_str = output_cfg.get("processed_path", "../data/processed/processed_cases.jsonl")
    out_path = resolve_path(str(out_path_str), args.config.parent)
    save_jsonl(records, out_path)

    print(f"processed_cases={len(records)}")
    print(f"saved_to={out_path}")


if __name__ == "__main__":
    main()
