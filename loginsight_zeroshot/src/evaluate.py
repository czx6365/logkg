from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from .utils import ensure_dir



def evaluate_predictions(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute metrics and confusion matrix from prediction records."""
    assert len(records) > 0, "No prediction records provided."

    y_true = np.array([str(r["fault_type"]) for r in records])
    y_pred = np.array([str(r["pred_fault_type"]) for r in records])
    parse_valid = np.array([bool(r.get("parse_valid", False)) for r in records], dtype=bool)

    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))

    # validity_rate 反映模型回答是否遵守了预期输出格式，不等于分类准确率。
    metrics = {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "validity_rate": float(parse_valid.mean()),
        "n_cases": int(len(records)),
    }

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per_class = []
    for i, lb in enumerate(labels):
        per_class.append(
            {
                "fault_type": lb,
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(s[i]),
            }
        )

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "summary": metrics,
        "labels": labels,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }



def save_evaluation(result: Dict[str, Any], summary_csv: Path, details_json: Path) -> None:
    """Persist evaluation outputs to CSV and JSON."""
    ensure_dir(summary_csv)
    ensure_dir(details_json)

    pd.DataFrame([result["summary"]]).to_csv(summary_csv, index=False)

    # 单独导出 per-class 和混淆矩阵，便于论文表格或错误分析直接使用。
    per_class_path = summary_csv.with_name(summary_csv.stem + "_per_class.csv")
    pd.DataFrame(result["per_class"]).to_csv(per_class_path, index=False)

    cm_path = summary_csv.with_name(summary_csv.stem + "_confusion_matrix.csv")
    cm_df = pd.DataFrame(
        result["confusion_matrix"],
        index=[f"true::{x}" for x in result["labels"]],
        columns=[f"pred::{x}" for x in result["labels"]],
    )
    cm_df.to_csv(cm_path)

    # 详细 JSON 保持紧凑，只保留汇总和按类指标；完整混淆矩阵单独看 CSV。
    with details_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
