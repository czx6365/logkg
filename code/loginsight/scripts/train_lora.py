from __future__ import annotations

import argparse
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lora_train import train_lora
from src.utils import load_yaml



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    # 训练入口本身很薄，真正的模型加载和 Trainer 配置都在 src/lora_train.py。
    out_dir = train_lora(cfg, args.config.parent)
    print(f"final_adapter={out_dir}")


if __name__ == "__main__":
    main()
