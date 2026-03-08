from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from numpy import linalg as la
from sklearn.cluster import OPTICS
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

from model import LogKG
from process.preprocess_d1_to_logkg import build_sn_index, extract_case_rows, load_logs

CODE_DIR = Path(__file__).resolve().parent
ROOT_DIR = CODE_DIR.parent


def compute_squared_edm(X: np.ndarray) -> np.ndarray:
    n, _ = X.shape
    D = np.zeros([n, n], dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = la.norm(X[i, :] - X[j, :])
            D[j, i] = D[i, j]
    return D


def get_centroid_index(cluster_embedding: np.ndarray) -> int:
    distance_array = np.sum(compute_squared_edm(cluster_embedding), axis=1)
    return int(np.argmin(distance_array))


def build_cluster_model(min_samples: int, xi: float) -> OPTICS:
    return OPTICS(min_samples=min_samples, metric="cosine", xi=xi, algorithm="brute")


def deterministic_template_embedding(templates: List[str], embedding_size: int, seed: int) -> Dict[str, np.ndarray]:
    """
    按模板字符串生成确定性向量，避免每次运行随机噪声导致评估波动。
    """
    emb: Dict[str, np.ndarray] = {}
    for t in templates:
        h = hashlib.md5((str(seed) + "::" + t).encode("utf-8")).hexdigest()
        local_seed = int(h[:16], 16) % (2**32 - 1)
        rng = np.random.default_rng(local_seed)
        emb[t] = rng.normal(0, 1, embedding_size)
    return emb


def build_d1_cases_from_index(
    labels_df: pd.DataFrame,
    sn_index: Dict[str, object],
    history_hours: int,
    fallback_all_before: bool,
) -> Tuple[List[str], np.ndarray, Dict[str, pd.DataFrame]]:
    labels = labels_df.copy()
    labels["fault_time"] = pd.to_datetime(labels["fault_time"], errors="coerce")
    labels = labels.dropna(subset=["sn", "fault_time", "label"])

    case_name_list: List[str] = []
    y: List[int] = []
    case_log_df: Dict[str, pd.DataFrame] = {}

    window = np.timedelta64(history_hours, "h")
    for i, row in enumerate(labels.itertuples(index=False)):
        sn = str(row.sn)
        ft = pd.Timestamp(row.fault_time)
        label = int(row.label)
        case_id = f"{sn}__{ft.strftime('%Y%m%d%H%M%S')}__{i}"

        idx = sn_index.get(sn)
        case_df = extract_case_rows(
            idx=idx,
            fault_time=np.datetime64(ft, "ns"),
            window=window,
            fallback_all_before=fallback_all_before,
        )
        case_df["EventId"] = case_df["EventId"].fillna("NO_LOG").astype(str)

        case_name_list.append(case_id)
        y.append(label)
        case_log_df[case_id] = case_df

    return case_name_list, np.array(y, dtype=int), case_log_df


@dataclass
class EvalResult:
    window_hours: int
    acc_mean: float
    acc_std: float
    f1_mean: float
    f1_std: float
    confusion_matrix: np.ndarray
    labels: np.ndarray


def run_one_window_eval(
    case_name_list: List[str],
    y: np.ndarray,
    case_log_df: Dict[str, pd.DataFrame],
    embedding_size: int,
    embedding_seed: int,
    idf_threshold: float,
    min_samples: int,
    xi: float,
    n_splits: int,
    random_state: int,
    rf_n_estimators: int,
) -> EvalResult:
    all_templates = sorted(
        {
            str(eid)
            for df in case_log_df.values()
            for eid in df["EventId"].dropna().values
            if str(eid) != ""
        }
    )
    template_embedding = deterministic_template_embedding(all_templates, embedding_size, embedding_seed)

    class_counts = Counter(y.tolist())
    min_class_count = min(class_counts.values())
    effective_splits = min(n_splits, min_class_count)
    if effective_splits < 2:
        raise ValueError(f"Not enough samples per class: min_class_count={min_class_count}")

    skf = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=random_state)

    acc_list: List[float] = []
    f1_list: List[float] = []
    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []

    for fold_idx, (train_index, test_index) in enumerate(skf.split(np.arange(len(case_name_list)), y), start=1):
        train_index = train_index.tolist()
        test_index = test_index.tolist()

        train_df = {case_name_list[idx]: case_log_df[case_name_list[idx]] for idx in train_index}
        test_df = {case_name_list[idx]: case_log_df[case_name_list[idx]] for idx in test_index}

        model = LogKG(train_df, test_df, idf_threshold, template_embedding, embedding_size=embedding_size)
        model.get_train_embedding()
        model.get_test_embedding()

        train_embedding = model.train_embedding_dict
        test_embedding = model.test_embedding_dict
        train_set = np.array([train_embedding[case_name_list[idx]] for idx in train_index])
        test_set = np.array([test_embedding[case_name_list[idx]] for idx in test_index])

        cluster_model = build_cluster_model(min_samples=min_samples, xi=xi)
        cluster_result = cluster_model.fit_predict(train_set)
        class_num = np.max(cluster_result) + 1
        if class_num <= 0:
            raise ValueError("No valid clusters generated by OPTICS.")

        cluster_centroids = [
            train_index[np.where(cluster_result == i)[0][get_centroid_index(train_set[np.where(cluster_result == i)[0]])]]
            for i in range(class_num)
        ]

        classify_index = np.zeros(len(cluster_result), dtype=int) - 1
        for i in range(class_num):
            class_label = y[cluster_centroids[i]]
            classify_index[np.where(cluster_result == i)[0]] = class_label

        mask = classify_index != -1
        if not np.any(mask):
            raise ValueError("No labeled train samples after clustering.")

        classifier = RandomForestClassifier(
            n_estimators=rf_n_estimators,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        classifier.fit(train_set[mask], classify_index[mask])

        y_true = y[test_index].astype(int)
        y_pred = classifier.predict(test_set).astype(int)
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro")

        acc_list.append(float(acc))
        f1_list.append(float(macro_f1))
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
        print(f"fold {fold_idx}/{effective_splits}: acc={acc:.4f}, macro_f1={macro_f1:.4f}, test_size={len(test_index)}")

    y_true_concat = np.concatenate(y_true_all).astype(int)
    y_pred_concat = np.concatenate(y_pred_all).astype(int)
    labels = np.unique(y)
    cm = confusion_matrix(y_true_concat, y_pred_concat, labels=labels)

    return EvalResult(
        window_hours=-1,
        acc_mean=float(np.mean(acc_list)),
        acc_std=float(np.std(acc_list)),
        f1_mean=float(np.mean(f1_list)),
        f1_std=float(np.std(f1_list)),
        confusion_matrix=cm,
        labels=labels,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="D1 enhanced LogKG experiment with window search.")
    parser.add_argument("--d1_dir", type=Path, default=ROOT_DIR / "data" / "D1")
    parser.add_argument("--log_file", type=str, default="preliminary_sel_log_dataset.csv")
    parser.add_argument("--label_file", type=str, default="preliminary_train_label_dataset_s.csv")
    parser.add_argument("--window_candidates", type=str, default="6,12,24,48")
    parser.add_argument("--fallback_all_before", action="store_true", default=False)
    parser.add_argument("--use_server_model", action="store_true", default=False)
    parser.add_argument("--embedding_size", type=int, default=16)
    parser.add_argument("--embedding_seed", type=int, default=42)
    parser.add_argument("--idf_threshold", type=float, default=0.4)
    parser.add_argument("--min_samples", type=int, default=3)
    parser.add_argument("--xi", type=float, default=0.05)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--rf_n_estimators", type=int, default=300)
    parser.add_argument("--result_json", type=Path, default=CODE_DIR / "result" / "d1_enhanced_search_result.json")
    args = parser.parse_args()

    labels_df = pd.read_csv(args.d1_dir / args.label_file)
    required = {"sn", "fault_time", "label"}
    missing = required - set(labels_df.columns)
    if missing:
        raise ValueError(f"label file missing required columns: {sorted(missing)}")

    print("Loading logs and building robust templates...")
    logs, _ = load_logs(args.d1_dir / args.log_file, use_server_model=args.use_server_model)
    sn_index = build_sn_index(logs)
    print(f"log rows={len(logs)}, unique_sn={logs['sn'].nunique()}, unique_templates={logs['EventId'].nunique()}")

    window_candidates = [int(x.strip()) for x in args.window_candidates.split(",") if x.strip()]
    if not window_candidates:
        raise ValueError("window_candidates is empty.")

    all_results: List[Dict[str, object]] = []
    best: Dict[str, object] | None = None

    for wh in window_candidates:
        print("\n" + "-" * 30 + f" Window={wh}h " + "-" * 30)
        case_name_list, y, case_log_df = build_d1_cases_from_index(
            labels_df=labels_df,
            sn_index=sn_index,
            history_hours=wh,
            fallback_all_before=args.fallback_all_before,
        )
        print(f"cases={len(case_name_list)}, class_dist={Counter(y.tolist())}")

        result = run_one_window_eval(
            case_name_list=case_name_list,
            y=y,
            case_log_df=case_log_df,
            embedding_size=args.embedding_size,
            embedding_seed=args.embedding_seed,
            idf_threshold=args.idf_threshold,
            min_samples=args.min_samples,
            xi=args.xi,
            n_splits=args.n_splits,
            random_state=args.random_state,
            rf_n_estimators=args.rf_n_estimators,
        )
        result.window_hours = wh

        print("=" * 30 + " Summary " + "=" * 30)
        print(f"window={wh}h, accuracy: mean={result.acc_mean:.4f}, std={result.acc_std:.4f}")
        print(f"window={wh}h, macro_f1: mean={result.f1_mean:.4f}, std={result.f1_std:.4f}")
        print("confusion_matrix labels:", result.labels.tolist())
        print(result.confusion_matrix)

        row = {
            "window_hours": wh,
            "acc_mean": result.acc_mean,
            "acc_std": result.acc_std,
            "f1_mean": result.f1_mean,
            "f1_std": result.f1_std,
            "labels": result.labels.tolist(),
            "confusion_matrix": result.confusion_matrix.tolist(),
        }
        all_results.append(row)

        if best is None or row["f1_mean"] > best["f1_mean"]:
            best = row

    print("\n" + "#" * 30 + " Best Window " + "#" * 30)
    print(best)

    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "settings": {
            "d1_dir": str(args.d1_dir),
            "log_file": args.log_file,
            "label_file": args.label_file,
            "window_candidates": window_candidates,
            "fallback_all_before": args.fallback_all_before,
            "use_server_model": args.use_server_model,
            "embedding_size": args.embedding_size,
            "embedding_seed": args.embedding_seed,
            "idf_threshold": args.idf_threshold,
            "min_samples": args.min_samples,
            "xi": args.xi,
            "n_splits": args.n_splits,
            "random_state": args.random_state,
            "rf_n_estimators": args.rf_n_estimators,
        },
        "results": all_results,
        "best_by_macro_f1": best,
    }
    with open(args.result_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"saved search result -> {args.result_json}")


if __name__ == "__main__":
    main()
