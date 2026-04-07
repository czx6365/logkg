from __future__ import annotations

import argparse
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ablation import run_ablation
from src.utils import load_jsonl, load_yaml, resolve_path



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    paths = cfg["paths"]

    cases = load_jsonl(resolve_path(paths["processed_cases"], args.config.parent))
    fols_cfg = cfg["fols"]
    abl_cfg = cfg.get("ablation", {})

    # 这里输出的是轻量消融对比表，不会触发 LoRA 微调。
    df = run_ablation(cases, fols_cfg, abl_cfg)
    out_csv = resolve_path(str(abl_cfg.get("output_csv", "../data/processed/ablation.csv")), args.config.parent)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(df.to_string(index=False))
    print(f"ablation_csv={out_csv}")


if __name__ == "__main__":
    main()
