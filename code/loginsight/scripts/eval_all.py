from __future__ import annotations

import argparse
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import evaluate_predictions, save_evaluation
from src.utils import load_jsonl, load_yaml, resolve_path



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    paths = cfg["paths"]

    pred_path = resolve_path(paths["predictions_path"], args.config.parent)
    eval_csv = resolve_path(paths["eval_csv"], args.config.parent)
    eval_json = resolve_path(paths["eval_json"], args.config.parent)

    preds = load_jsonl(pred_path)
    # 评估阶段只依赖预测结果文件，不重新跑模型。
    result = evaluate_predictions(preds)
    save_evaluation(result, eval_csv, eval_json)

    print(result["summary"])
    print(f"eval_csv={eval_csv}")
    print(f"eval_json={eval_json}")


if __name__ == "__main__":
    main()
