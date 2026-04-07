from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from .utils import simple_tokenize



def _to_token_set(line: str) -> Set[str]:
    return set(simple_tokenize(line))



def pairwise_jaccard_distance(token_sets: Sequence[Set[str]]) -> np.ndarray:
    """Compute precomputed Jaccard distance matrix for token sets."""
    n = len(token_sets)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            # 用词集合的交并比刻画日志行相似度，越相似距离越小。
            inter = len(token_sets[i].intersection(token_sets[j]))
            union = len(token_sets[i].union(token_sets[j]))
            dist = 1.0 if union == 0 else 1.0 - (inter / union)
            D[i, j] = dist
            D[j, i] = dist
    return D



def _choose_cluster_count(n_samples: int, config_value: Any) -> int:
    if config_value is not None:
        k = int(config_value)
    else:
        k = max(2, int(round(math.sqrt(max(n_samples, 2)))))
    return max(2, min(k, n_samples))



def _collapse_consecutive(lines: Sequence[str]) -> Tuple[List[str], List[int]]:
    """Collapse only consecutive duplicate lines and keep first original index."""
    out_lines: List[str] = []
    out_idx: List[int] = []
    prev = None
    for idx, line in enumerate(lines):
        # 只折叠连续重复，避免把不同时间段里重复出现的重要事件错误合并。
        if line == prev:
            continue
        out_lines.append(line)
        out_idx.append(idx)
        prev = line
    return out_lines, out_idx



def _uniform_subsample(lines: Sequence[str], orig_idx: Sequence[int], max_lines: int) -> Tuple[List[str], List[int]]:
    """Uniformly subsample while preserving chronology."""
    n = len(lines)
    if n <= max_lines:
        return list(lines), list(orig_idx)

    # 超长序列先均匀抽样，防止距离矩阵和聚类过程在大 case 上失控。
    selected = np.linspace(0, n - 1, num=max_lines, dtype=int)
    selected = np.unique(selected)
    return [lines[int(i)] for i in selected], [orig_idx[int(i)] for i in selected]



def cluster_lines(
    lines: Sequence[str],
    token_sets: Sequence[Set[str]],
    fols_cfg: Dict[str, Any],
    method: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster lines and return labels + distance matrix."""
    assert len(lines) == len(token_sets), "lines/token_sets length mismatch"
    n = len(lines)
    if n == 0:
        return np.array([], dtype=int), np.zeros((0, 0), dtype=float)
    if n == 1:
        return np.array([0], dtype=int), np.zeros((1, 1), dtype=float)

    D = pairwise_jaccard_distance(token_sets)
    method = method.lower()

    if method == "dbscan":
        # DBSCAN 不要求预设簇数，更适合日志里“模式数未知”的场景。
        model = DBSCAN(
            eps=float(fols_cfg.get("dbscan_eps", 0.6)),
            min_samples=int(fols_cfg.get("dbscan_min_samples", 2)),
            metric="precomputed",
        )
        labels = model.fit_predict(D)
        return labels.astype(int), D

    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    X = vectorizer.fit_transform(lines)

    if method == "kmeans":
        # KMeans/Agglomerative 作为替代聚类器，主要用于消融实验。
        k = _choose_cluster_count(n, fols_cfg.get("kmeans_k", None))
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        return labels.astype(int), D

    if method == "agglomerative":
        k = _choose_cluster_count(n, fols_cfg.get("agglomerative_k", None))
        labels = AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage="ward").fit_predict(X.toarray())
        return labels.astype(int), D

    raise ValueError(f"Unsupported clustering method: {method}")



def _representatives_from_labels(
    labels: np.ndarray,
    distance_matrix: np.ndarray,
    method: str,
) -> List[Dict[str, Any]]:
    """Select cluster representatives by minimum average intra-cluster distance."""
    reps: List[Dict[str, Any]] = []
    unique_labels = sorted(set(labels.tolist()))

    for lb in unique_labels:
        idx = np.where(labels == lb)[0]
        if method == "dbscan" and lb == -1:
            # DBSCAN 的噪声点没有簇中心，这里直接把它们自己保留下来。
            for i in idx:
                reps.append({"line_index": int(i), "cluster": int(lb), "is_noise": True})
            continue

        if len(idx) == 1:
            reps.append({"line_index": int(idx[0]), "cluster": int(lb), "is_noise": False})
            continue

        # 对每个簇选“平均簇内距离最小”的句子当代表句。
        sub = distance_matrix[np.ix_(idx, idx)]
        avg_dist = sub.mean(axis=1)
        center_pos = int(np.argmin(avg_dist))
        reps.append({"line_index": int(idx[center_pos]), "cluster": int(lb), "is_noise": False})

    return reps



def build_token_document_frequency(cases: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Compute corpus-level case document frequency n_t for IDF in FOLS."""
    doc_freq: Dict[str, int] = defaultdict(int)
    for case in cases:
        unique_tokens: Set[str] = set()
        for line in case.get("content_sequence", []):
            unique_tokens.update(simple_tokenize(str(line)))
        for tok in unique_tokens:
            # 这里按 case 统计 df，而不是按行统计，强调“该 token 在多少故障案例中出现”。
            doc_freq[tok] += 1
    return dict(doc_freq)



def _line_tfidf_score(line: str, doc_freq: Dict[str, int], total_cases: int) -> float:
    tokens = simple_tokenize(line)
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    n = float(len(tokens))
    score = 0.0
    for tok, c in counts.items():
        tf = c / n
        idf = math.log(total_cases / (doc_freq.get(tok, 0) + 1.0))
        score += tf * idf
    return float(score)



def summarize_case(
    case: Dict[str, Any],
    doc_freq: Dict[str, int],
    total_cases: int,
    fols_cfg: Dict[str, Any],
    method: str | None = None,
) -> Dict[str, Any]:
    """Build FOLS summary for one case with intermediate outputs."""
    original_lines = [str(x) for x in case.get("content_sequence", [])]
    original_count = len(original_lines)

    working_lines = list(original_lines)
    working_orig_idx = list(range(original_count))

    if bool(fols_cfg.get("collapse_consecutive_duplicates", True)):
        working_lines, working_orig_idx = _collapse_consecutive(working_lines)

    max_lines = int(fols_cfg.get("max_lines_for_clustering", 2000))
    if len(working_lines) > max_lines:
        working_lines, working_orig_idx = _uniform_subsample(working_lines, working_orig_idx, max_lines)

    # 先聚类压缩“相似行”，再用 TF-IDF 对代表句做重要性排序。
    token_sets = [_to_token_set(x) for x in working_lines]
    clustering_method = (method or fols_cfg.get("clustering_method", "dbscan")).lower()

    labels, D = cluster_lines(working_lines, token_sets, fols_cfg, clustering_method)
    reps = _representatives_from_labels(labels, D, clustering_method)

    reps_scored: List[Dict[str, Any]] = []
    for r in reps:
        idx = r["line_index"]
        line = working_lines[idx]
        orig_idx = int(working_orig_idx[idx])
        token_count = len(simple_tokenize(line))
        score = _line_tfidf_score(line, doc_freq, total_cases)
        reps_scored.append(
            {
                **r,
                "line": line,
                "original_line_index": orig_idx,
                "token_count": token_count,
                "tfidf_score": score,
            }
        )

    reps_ranked = sorted(reps_scored, key=lambda x: x["tfidf_score"], reverse=True)
    threshold = float(fols_cfg.get("tfidf_threshold", 0.0))
    min_tokens = int(fols_cfg.get("min_tokens_per_line", 1))
    max_summary_lines = int(fols_cfg.get("max_summary_lines", 40))

    retained = [
        r for r in reps_ranked
        if r["tfidf_score"] >= threshold and r["token_count"] >= min_tokens
    ]
    if not retained and reps_ranked:
        retained = [reps_ranked[0]]

    retained = retained[:max_summary_lines]
    # 排名用于筛选，但最终输出恢复成时间顺序，方便模型理解故障演化过程。
    retained_chrono = sorted(retained, key=lambda x: x["original_line_index"])
    summary_lines = [x["line"] for x in retained_chrono]

    return {
        "case_id": case.get("case_id"),
        "fault_type": case.get("fault_type"),
        "dataset_name": case.get("dataset_name", ""),
        "original_line_count": int(original_count),
        "working_line_count": int(len(working_lines)),
        "line_subsampled": bool(len(working_lines) < original_count),
        "cluster_assignments": labels.tolist(),
        "cluster_line_original_indices": working_orig_idx,
        "representatives": reps_ranked,
        "summary_line_indices": [x["original_line_index"] for x in retained_chrono],
        "fault_summary": summary_lines,
    }



def run_fols_for_cases(
    cases: Sequence[Dict[str, Any]],
    fols_cfg: Dict[str, Any],
    method: str | None = None,
) -> List[Dict[str, Any]]:
    """Run FOLS over all cases."""
    total_cases = len(cases)
    assert total_cases > 0, "No cases provided to FOLS."

    doc_freq = build_token_document_frequency(cases)
    out: List[Dict[str, Any]] = []
    for case in tqdm(cases, desc="FOLS"):
        # 每个 case 独立摘要，但共享全局 df，用来衡量“这个词在整个数据集里稀不稀有”。
        out.append(summarize_case(case, doc_freq, total_cases, fols_cfg, method=method))
    return out
