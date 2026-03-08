from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold

from model import LogKG

CODE_DIR = Path(__file__).resolve().parent
ROOT_DIR = CODE_DIR.parent


def build_deterministic_random_embedding(event_ids: List[str], embedding_size: int, seed: int) -> Dict[str, np.ndarray]:
    embedding: Dict[str, np.ndarray] = {}
    for eid in event_ids:
        h = hashlib.md5((str(seed) + "::" + eid).encode("utf-8")).hexdigest()
        local_seed = int(h[:16], 16) % (2**32 - 1)
        rng = np.random.default_rng(local_seed)
        embedding[eid] = rng.normal(0, 1, embedding_size)
    return embedding


def load_cases(case_dir: Path) -> Tuple[List[str], Dict[str, pd.DataFrame]]:
    case_files = sorted(case_dir.glob("*.csv"))
    case_names = [p.stem for p in case_files]
    case_log_df: Dict[str, pd.DataFrame] = {}
    for p in case_files:
        df = pd.read_csv(p, low_memory=False)
        if "EventId" not in df.columns:
            raise ValueError(f"EventId missing in case file: {p}")
        df["EventId"] = df["EventId"].fillna("NO_LOG").astype(str)
        case_log_df[p.stem] = df
    return case_names, case_log_df


def build_labels(case_labels_csv: Path, case_names: List[str]) -> Tuple[np.ndarray, np.ndarray, Dict[int, str], Dict[int, str]]:
    labels_df = pd.read_csv(case_labels_csv, encoding="utf-8")
    labels_df["case_id"] = labels_df["case_id"].astype(str)
    labels_df = labels_df[labels_df["case_id"].isin(case_names)].copy()
    if labels_df.empty:
        raise ValueError("No overlapping cases between case_labels.csv and case files.")

    labels_df = labels_df.set_index("case_id")
    missing = [c for c in case_names if c not in labels_df.index]
    if missing:
        raise ValueError(f"Missing labels for {len(missing)} cases, sample={missing[:5]}")

    major_names = sorted(labels_df["major_problem_type"].astype(str).unique().tolist())
    minor_names = sorted(labels_df["minor_problem_type"].astype(str).unique().tolist())
    major_to_id = {name: i for i, name in enumerate(major_names)}
    minor_to_id = {name: i for i, name in enumerate(minor_names)}

    y_major = np.array([major_to_id[str(labels_df.loc[c, "major_problem_type"])] for c in case_names], dtype=int)
    y_minor = np.array([minor_to_id[str(labels_df.loc[c, "minor_problem_type"])] for c in case_names], dtype=int)

    id_to_major = {v: k for k, v in major_to_id.items()}
    id_to_minor = {v: k for k, v in minor_to_id.items()}
    return y_major, y_minor, id_to_major, id_to_minor


def build_case_embeddings(
    case_names: List[str],
    case_log_df: Dict[str, pd.DataFrame],
    train_index: List[int],
    test_index: List[int],
    template_embedding: Dict[str, np.ndarray],
    idf_threshold: float,
    embedding_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    train_df = {case_names[idx]: case_log_df[case_names[idx]] for idx in train_index}
    test_df = {case_names[idx]: case_log_df[case_names[idx]] for idx in test_index}

    model = LogKG(train_df, test_df, idf_threshold, template_embedding, embedding_size=embedding_size)
    model.get_train_embedding()
    model.get_test_embedding()

    x_train = np.array([model.train_embedding_dict[case_names[idx]] for idx in train_index])
    x_test = np.array([model.test_embedding_dict[case_names[idx]] for idx in test_index])
    return x_train, x_test


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def metric_mean_std(metric_list: List[Dict[str, float]], key: str) -> Tuple[float, float]:
    arr = np.array([m[key] for m in metric_list], dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


def evaluate_hierarchical(
    case_names: List[str],
    case_log_df: Dict[str, pd.DataFrame],
    y_major: np.ndarray,
    y_minor: np.ndarray,
    id_to_major: Dict[int, str],
    id_to_minor: Dict[int, str],
    embedding_size: int,
    embedding_seed: int,
    idf_threshold: float,
    n_splits: int,
    min_samples_per_class: int,
    rf_estimators: int,
    random_state: int,
) -> Dict[str, object]:
    major_counts = Counter(y_major.tolist())
    rare_major = [label for label, count in major_counts.items() if count < min_samples_per_class]
    if rare_major:
        keep = np.array([i for i, y in enumerate(y_major.tolist()) if y not in rare_major], dtype=int)
        dropped = len(y_major) - len(keep)
        case_names = [case_names[i] for i in keep.tolist()]
        y_major = y_major[keep]
        y_minor = y_minor[keep]
        case_log_df = {name: case_log_df[name] for name in case_names}
        major_counts = Counter(y_major.tolist())
        print(f"[Warn] Dropped {dropped} samples from rare major classes: {rare_major}")

    min_major_count = min(major_counts.values())
    effective_splits = min(n_splits, min_major_count)
    if effective_splits < 2:
        raise ValueError(f"Not enough samples for CV after filtering. min_major_count={min_major_count}")

    all_event_ids = sorted(
        {
            str(eid)
            for df in case_log_df.values()
            for eid in df["EventId"].dropna().values
            if str(eid) != ""
        }
    )
    template_embedding = build_deterministic_random_embedding(
        event_ids=all_event_ids,
        embedding_size=embedding_size,
        seed=embedding_seed,
    )

    skf = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=random_state)

    major_metrics_list: List[Dict[str, float]] = []
    minor_metrics_list: List[Dict[str, float]] = []
    minor_oracle_metrics_list: List[Dict[str, float]] = []

    y_major_true_all: List[np.ndarray] = []
    y_major_pred_all: List[np.ndarray] = []
    y_minor_true_all: List[np.ndarray] = []
    y_minor_pred_all: List[np.ndarray] = []
    y_minor_oracle_pred_all: List[np.ndarray] = []

    per_major_minor_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for fold_idx, (tr, te) in enumerate(skf.split(np.arange(len(case_names)), y_major), start=1):
        train_index = tr.tolist()
        test_index = te.tolist()
        x_train, x_test = build_case_embeddings(
            case_names=case_names,
            case_log_df=case_log_df,
            train_index=train_index,
            test_index=test_index,
            template_embedding=template_embedding,
            idf_threshold=idf_threshold,
            embedding_size=embedding_size,
        )

        y_major_train = y_major[train_index]
        y_major_test = y_major[test_index]
        y_minor_train = y_minor[train_index]
        y_minor_test = y_minor[test_index]

        major_clf = RandomForestClassifier(
            n_estimators=rf_estimators,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=1,
        )
        major_clf.fit(x_train, y_major_train)
        y_major_pred = major_clf.predict(x_test).astype(int)
        major_metrics = calc_metrics(y_major_test, y_major_pred)
        major_metrics_list.append(major_metrics)

        major_minor_models: Dict[int, object] = {}
        major_default_minor: Dict[int, int] = {}
        for major_id in np.unique(y_major_train):
            mask = y_major_train == major_id
            x_m = x_train[mask]
            y_m = y_minor_train[mask]
            c = Counter(y_m.tolist())
            major_default_minor[int(major_id)] = int(max(c.items(), key=lambda kv: kv[1])[0])

            if len(c) < 2:
                major_minor_models[int(major_id)] = None
                continue

            minor_clf = RandomForestClassifier(
                n_estimators=rf_estimators,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=1,
            )
            minor_clf.fit(x_m, y_m)
            major_minor_models[int(major_id)] = minor_clf

        y_minor_pred = np.zeros(len(test_index), dtype=int)
        y_minor_oracle_pred = np.zeros(len(test_index), dtype=int)

        for i in range(len(test_index)):
            predicted_major = int(y_major_pred[i])
            true_major = int(y_major_test[i])

            model_pred_major = major_minor_models.get(predicted_major)
            if model_pred_major is None:
                y_minor_pred[i] = major_default_minor.get(predicted_major, int(y_minor_train[0]))
            else:
                y_minor_pred[i] = int(model_pred_major.predict(x_test[i : i + 1])[0])

            model_true_major = major_minor_models.get(true_major)
            if model_true_major is None:
                y_minor_oracle_pred[i] = major_default_minor.get(true_major, int(y_minor_train[0]))
            else:
                y_minor_oracle_pred[i] = int(model_true_major.predict(x_test[i : i + 1])[0])

            per_major_minor_stats[true_major]["total"] += 1
            per_major_minor_stats[true_major]["correct"] += int(y_minor_pred[i] == y_minor_test[i])

        minor_metrics = calc_metrics(y_minor_test, y_minor_pred)
        minor_oracle_metrics = calc_metrics(y_minor_test, y_minor_oracle_pred)
        minor_metrics_list.append(minor_metrics)
        minor_oracle_metrics_list.append(minor_oracle_metrics)

        y_major_true_all.append(y_major_test)
        y_major_pred_all.append(y_major_pred)
        y_minor_true_all.append(y_minor_test)
        y_minor_pred_all.append(y_minor_pred)
        y_minor_oracle_pred_all.append(y_minor_oracle_pred)

        print(
            f"fold {fold_idx}/{effective_splits}: "
            f"major_acc={major_metrics['acc']:.4f}, major_macro_f1={major_metrics['macro_f1']:.4f}, "
            f"major_weighted_f1={major_metrics['weighted_f1']:.4f}, major_macro_recall={major_metrics['macro_recall']:.4f}, "
            f"minor_acc={minor_metrics['acc']:.4f}, minor_macro_f1={minor_metrics['macro_f1']:.4f}, "
            f"minor_weighted_f1={minor_metrics['weighted_f1']:.4f}, minor_macro_recall={minor_metrics['macro_recall']:.4f}, "
            f"minor_oracle_macro_f1={minor_oracle_metrics['macro_f1']:.4f}"
        )

    major_true = np.concatenate(y_major_true_all).astype(int)
    major_pred = np.concatenate(y_major_pred_all).astype(int)
    minor_true = np.concatenate(y_minor_true_all).astype(int)
    minor_pred = np.concatenate(y_minor_pred_all).astype(int)
    minor_oracle_pred = np.concatenate(y_minor_oracle_pred_all).astype(int)

    major_labels = sorted(np.unique(y_major).tolist())
    minor_labels = sorted(np.unique(y_minor).tolist())

    major_acc_mean, major_acc_std = metric_mean_std(major_metrics_list, "acc")
    major_macro_f1_mean, major_macro_f1_std = metric_mean_std(major_metrics_list, "macro_f1")
    major_weighted_f1_mean, major_weighted_f1_std = metric_mean_std(major_metrics_list, "weighted_f1")
    major_macro_recall_mean, major_macro_recall_std = metric_mean_std(major_metrics_list, "macro_recall")
    major_weighted_recall_mean, major_weighted_recall_std = metric_mean_std(major_metrics_list, "weighted_recall")

    minor_acc_mean, minor_acc_std = metric_mean_std(minor_metrics_list, "acc")
    minor_macro_f1_mean, minor_macro_f1_std = metric_mean_std(minor_metrics_list, "macro_f1")
    minor_weighted_f1_mean, minor_weighted_f1_std = metric_mean_std(minor_metrics_list, "weighted_f1")
    minor_macro_recall_mean, minor_macro_recall_std = metric_mean_std(minor_metrics_list, "macro_recall")
    minor_weighted_recall_mean, minor_weighted_recall_std = metric_mean_std(minor_metrics_list, "weighted_recall")

    minor_oracle_acc_mean, minor_oracle_acc_std = metric_mean_std(minor_oracle_metrics_list, "acc")
    minor_oracle_macro_f1_mean, minor_oracle_macro_f1_std = metric_mean_std(minor_oracle_metrics_list, "macro_f1")
    minor_oracle_weighted_f1_mean, minor_oracle_weighted_f1_std = metric_mean_std(minor_oracle_metrics_list, "weighted_f1")
    minor_oracle_macro_recall_mean, minor_oracle_macro_recall_std = metric_mean_std(minor_oracle_metrics_list, "macro_recall")
    minor_oracle_weighted_recall_mean, minor_oracle_weighted_recall_std = metric_mean_std(
        minor_oracle_metrics_list, "weighted_recall"
    )

    summary = {
        "case_count": int(len(case_names)),
        "major_class_count": int(len(major_labels)),
        "minor_class_count": int(len(minor_labels)),
        "effective_splits": int(effective_splits),
        "major_acc_mean": major_acc_mean,
        "major_acc_std": major_acc_std,
        "major_macro_f1_mean": major_macro_f1_mean,
        "major_macro_f1_std": major_macro_f1_std,
        "major_weighted_f1_mean": major_weighted_f1_mean,
        "major_weighted_f1_std": major_weighted_f1_std,
        "major_macro_recall_mean": major_macro_recall_mean,
        "major_macro_recall_std": major_macro_recall_std,
        "major_weighted_recall_mean": major_weighted_recall_mean,
        "major_weighted_recall_std": major_weighted_recall_std,
        "minor_acc_mean": minor_acc_mean,
        "minor_acc_std": minor_acc_std,
        "minor_macro_f1_mean": minor_macro_f1_mean,
        "minor_macro_f1_std": minor_macro_f1_std,
        "minor_weighted_f1_mean": minor_weighted_f1_mean,
        "minor_weighted_f1_std": minor_weighted_f1_std,
        "minor_macro_recall_mean": minor_macro_recall_mean,
        "minor_macro_recall_std": minor_macro_recall_std,
        "minor_weighted_recall_mean": minor_weighted_recall_mean,
        "minor_weighted_recall_std": minor_weighted_recall_std,
        "minor_oracle_acc_mean": minor_oracle_acc_mean,
        "minor_oracle_acc_std": minor_oracle_acc_std,
        "minor_oracle_macro_f1_mean": minor_oracle_macro_f1_mean,
        "minor_oracle_macro_f1_std": minor_oracle_macro_f1_std,
        "minor_oracle_weighted_f1_mean": minor_oracle_weighted_f1_mean,
        "minor_oracle_weighted_f1_std": minor_oracle_weighted_f1_std,
        "minor_oracle_macro_recall_mean": minor_oracle_macro_recall_mean,
        "minor_oracle_macro_recall_std": minor_oracle_macro_recall_std,
        "minor_oracle_weighted_recall_mean": minor_oracle_weighted_recall_mean,
        "minor_oracle_weighted_recall_std": minor_oracle_weighted_recall_std,
        "major_confusion_matrix": confusion_matrix(major_true, major_pred, labels=major_labels).tolist(),
        "major_confusion_labels": [id_to_major[i] for i in major_labels],
        "minor_confusion_matrix": confusion_matrix(minor_true, minor_pred, labels=minor_labels).tolist(),
        "minor_confusion_labels": [id_to_minor[i] for i in minor_labels],
        "minor_oracle_confusion_matrix": confusion_matrix(minor_true, minor_oracle_pred, labels=minor_labels).tolist(),
        "fold_metrics": [
            {
                "major_acc": float(major_m["acc"]),
                "major_macro_f1": float(major_m["macro_f1"]),
                "major_weighted_f1": float(major_m["weighted_f1"]),
                "major_macro_recall": float(major_m["macro_recall"]),
                "major_weighted_recall": float(major_m["weighted_recall"]),
                "minor_acc": float(minor_m["acc"]),
                "minor_macro_f1": float(minor_m["macro_f1"]),
                "minor_weighted_f1": float(minor_m["weighted_f1"]),
                "minor_macro_recall": float(minor_m["macro_recall"]),
                "minor_weighted_recall": float(minor_m["weighted_recall"]),
                "minor_oracle_acc": float(oracle_m["acc"]),
                "minor_oracle_macro_f1": float(oracle_m["macro_f1"]),
                "minor_oracle_weighted_f1": float(oracle_m["weighted_f1"]),
                "minor_oracle_macro_recall": float(oracle_m["macro_recall"]),
                "minor_oracle_weighted_recall": float(oracle_m["weighted_recall"]),
            }
            for major_m, minor_m, oracle_m in zip(major_metrics_list, minor_metrics_list, minor_oracle_metrics_list)
        ],
        "per_major_minor_acc": {
            id_to_major[mid]: float(v["correct"] / max(v["total"], 1)) for mid, v in per_major_minor_stats.items()
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Hierarchical (major -> minor) classifier on OS_preprocessed.")
    parser.add_argument("--os_preprocessed_dir", type=Path, default=ROOT_DIR / "data" / "OS_preprocessed")
    parser.add_argument("--embedding_size", type=int, default=16)
    parser.add_argument("--embedding_seed", type=int, default=42)
    parser.add_argument("--idf_threshold", type=float, default=0.4)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--min_samples_per_class", type=int, default=2)
    parser.add_argument("--rf_estimators", type=int, default=600)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--result_json", type=Path, default=CODE_DIR / "result" / "os_hierarchical_result.json")
    args = parser.parse_args()

    case_dir = args.os_preprocessed_dir / "cases"
    case_labels_csv = args.os_preprocessed_dir / "case_labels.csv"
    if not case_dir.exists():
        raise FileNotFoundError(f"case directory not found: {case_dir}")
    if not case_labels_csv.exists():
        raise FileNotFoundError(f"case_labels.csv not found: {case_labels_csv}")

    case_names, case_log_df = load_cases(case_dir)
    y_major, y_minor, id_to_major, id_to_minor = build_labels(case_labels_csv, case_names)

    print(f"cases={len(case_names)}, major_classes={len(set(y_major.tolist()))}, minor_classes={len(set(y_minor.tolist()))}")
    print(f"major_dist={Counter(y_major.tolist())}")
    print(f"minor_dist_top10={Counter(y_minor.tolist()).most_common(10)}")

    summary = evaluate_hierarchical(
        case_names=case_names,
        case_log_df=case_log_df,
        y_major=y_major,
        y_minor=y_minor,
        id_to_major=id_to_major,
        id_to_minor=id_to_minor,
        embedding_size=args.embedding_size,
        embedding_seed=args.embedding_seed,
        idf_threshold=args.idf_threshold,
        n_splits=args.n_splits,
        min_samples_per_class=args.min_samples_per_class,
        rf_estimators=args.rf_estimators,
        random_state=args.random_state,
    )

    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.result_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 24 + " Hierarchical Summary " + "=" * 24)
    print(f"major_acc={summary['major_acc_mean']:.4f}+-{summary['major_acc_std']:.4f}")
    print(
        f"major_macro_f1={summary['major_macro_f1_mean']:.4f}+-{summary['major_macro_f1_std']:.4f}, "
        f"major_weighted_f1={summary['major_weighted_f1_mean']:.4f}+-{summary['major_weighted_f1_std']:.4f}"
    )
    print(
        f"major_macro_recall={summary['major_macro_recall_mean']:.4f}+-{summary['major_macro_recall_std']:.4f}, "
        f"major_weighted_recall={summary['major_weighted_recall_mean']:.4f}+-{summary['major_weighted_recall_std']:.4f}"
    )
    print(f"minor_acc={summary['minor_acc_mean']:.4f}+-{summary['minor_acc_std']:.4f}")
    print(
        f"minor_macro_f1={summary['minor_macro_f1_mean']:.4f}+-{summary['minor_macro_f1_std']:.4f}, "
        f"minor_weighted_f1={summary['minor_weighted_f1_mean']:.4f}+-{summary['minor_weighted_f1_std']:.4f}"
    )
    print(
        f"minor_macro_recall={summary['minor_macro_recall_mean']:.4f}+-{summary['minor_macro_recall_std']:.4f}, "
        f"minor_weighted_recall={summary['minor_weighted_recall_mean']:.4f}+-{summary['minor_weighted_recall_std']:.4f}"
    )
    print(
        f"minor_oracle_macro_f1={summary['minor_oracle_macro_f1_mean']:.4f}+-{summary['minor_oracle_macro_f1_std']:.4f}, "
        f"minor_oracle_weighted_f1={summary['minor_oracle_weighted_f1_mean']:.4f}+-{summary['minor_oracle_weighted_f1_std']:.4f}"
    )
    print(
        f"minor_oracle_macro_recall={summary['minor_oracle_macro_recall_mean']:.4f}+-{summary['minor_oracle_macro_recall_std']:.4f}, "
        f"minor_oracle_weighted_recall={summary['minor_oracle_weighted_recall_mean']:.4f}+-{summary['minor_oracle_weighted_recall_std']:.4f}"
    )
    print(f"result_json={args.result_json}")


if __name__ == "__main__":
    main()
