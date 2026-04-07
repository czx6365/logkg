from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from .fols import build_token_document_frequency, summarize_case



def _subset_cases(
    cases: Sequence[Dict[str, Any]],
    max_cases: int | None,
    random_state: int,
) -> List[Dict[str, Any]]:
    if max_cases is None or len(cases) <= max_cases:
        return list(cases)

    y = np.array([str(c["fault_type"]) for c in cases])
    idx = np.arange(len(cases))

    keep_idx, _ = train_test_split(
        idx,
        train_size=max_cases,
        random_state=random_state,
        stratify=y if len(set(y.tolist())) > 1 else None,
    )
    keep_idx = sorted(int(i) for i in keep_idx)
    return [cases[i] for i in keep_idx]



def _variant_text(
    case: Dict[str, Any],
    fols_cfg: Dict[str, Any],
    variant: str,
    doc_freq: Dict[str, int],
    total_cases: int,
) -> str:
    if variant == "without_fols":
        return "\n".join(str(x) for x in case.get("content_sequence", []))

    method_map = {
        "full_loginsight": "dbscan",
        "kmeans_replace": "kmeans",
        "agglomerative_replace": "agglomerative",
    }
    method = method_map.get(variant, "dbscan")
    out = summarize_case(case, doc_freq, total_cases, fols_cfg, method=method)
    return "\n".join(out.get("fault_summary", []))



def run_ablation(
    cases: Sequence[Dict[str, Any]],
    fols_cfg: Dict[str, Any],
    ablation_cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    Basic ablation support with lightweight evaluator.

    Approximation: uses TF-IDF + LogisticRegression instead of full LLM finetuning,
    to quickly compare representation quality under ablation variants.
    """
    variants = ablation_cfg.get(
        "variants",
        ["full_loginsight", "without_fols", "kmeans_replace", "agglomerative_replace"],
    )
    test_size = float(ablation_cfg.get("test_size", 0.2))
    random_state = int(ablation_cfg.get("random_state", 42))
    max_cases = ablation_cfg.get("max_cases", None)
    max_cases = int(max_cases) if max_cases is not None else None

    cases_used = _subset_cases(cases, max_cases=max_cases, random_state=random_state)
    y = np.array([str(c["fault_type"]) for c in cases_used])

    doc_freq = build_token_document_frequency(cases_used)
    total_cases = len(cases_used)
    rows: List[Dict[str, Any]] = []

    for variant in variants:
        # 这里比较的是不同“日志表示方式”的可分性，而不是完整微调后的最终效果。
        texts = [_variant_text(c, fols_cfg, variant, doc_freq, total_cases) for c in cases_used]

        n_classes = len(set(y.tolist()))
        test_count = max(1, int(round(len(y) * test_size)))
        min_class_count = min(np.bincount(pd.factorize(y)[0])) if len(y) > 0 else 0
        can_stratify = n_classes > 1 and test_count >= n_classes and min_class_count >= 2

        x_train, x_test, y_train, y_test = train_test_split(
            texts,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if can_stratify else None,
        )

        vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=1)
        x_train_v = vec.fit_transform(x_train)
        x_test_v = vec.transform(x_test)

        # 用轻量分类器快速做代理评估，便于低成本跑消融。
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(x_train_v, y_train)
        y_pred = clf.predict(x_test_v)

        rows.append(
            {
                "variant": variant,
                "micro_f1": float(f1_score(y_test, y_pred, average="micro", zero_division=0)),
                "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
                "n_cases_used": int(len(cases_used)),
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
            }
        )

    return pd.DataFrame(rows)
