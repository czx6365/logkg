from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
from tqdm import tqdm

from .utils import load_jsonl



def _extract_content(raw_line: str, content_regex: str) -> str:
    """Extract content from a raw line using configurable regex."""
    line = str(raw_line)
    # 允许通过配置切换日志正文抽取规则，便于适配不同数据源格式。
    m = re.search(content_regex, line)
    if not m:
        return line.strip()
    if "content" in m.groupdict():
        return str(m.group("content")).strip()
    if m.groups():
        return str(m.group(1)).strip()
    return line.strip()



def _apply_parser_variant(text: str, parser_variant: str) -> str:
    """
    Parser variant interface.

    drain/divlog/lilac are placeholders in this compact reproduction and
    currently fallback to regex-cleaned text.
    """
    variant = parser_variant.lower().strip()
    # 这里保留了解析器扩展点；当前轻量复现版本统一退化为正则清洗后的文本。
    if variant in {"regex", "drain", "divlog", "lilac"}:
        return text
    raise ValueError(f"Unsupported parser_variant: {parser_variant}")



def preprocess_line(raw_line: str, cfg: Dict[str, Any]) -> str:
    """Preprocess one log line."""
    # 先抽正文，再按配置移除时间戳、级别、host/pid 等噪声字段。
    content = _extract_content(raw_line, cfg.get("content_regex", r"(?P<content>.*)"))
    content = _apply_parser_variant(content, cfg.get("parser_variant", "regex"))

    for pattern in cfg.get("cleanup_patterns", []):
        content = re.sub(pattern, " ", content, flags=re.IGNORECASE)

    # 最后统一压缩空白，得到模型更容易消费的文本。
    content = re.sub(r"\s+", " ", content).strip()
    return content



def preprocess_sequence(raw_logs: Iterable[str], cfg: Dict[str, Any]) -> List[str]:
    """Preprocess sequence while keeping chronological order."""
    out: List[str] = []
    for line in raw_logs:
        cleaned = preprocess_line(str(line), cfg)
        if cleaned:
            # 保留原始时间顺序，后面的 FOLS 和推理都依赖 case 内的时序语义。
            out.append(cleaned)
    return out



def _case_record(
    case_id: str,
    fault_type: str,
    raw_logs: List[str],
    content_sequence: List[str],
    dataset_name: str,
    timestamp_start: str = "",
    timestamp_end: str = "",
) -> Dict[str, Any]:
    return {
        "case_id": case_id,
        "fault_type": fault_type,
        "raw_logs": raw_logs,
        "content_sequence": content_sequence,
        "timestamp_start": timestamp_start,
        "timestamp_end": timestamp_end,
        "dataset_name": dataset_name,
    }



def load_os_preprocessed_cases(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Adapter for ../../data/OS_preprocessed."""
    base_dir = Path(cfg["os_preprocessed_dir"])
    cases_dir = base_dir / "cases"
    labels_path = base_dir / "case_labels.csv"

    if not cases_dir.exists() or not labels_path.exists():
        raise FileNotFoundError(f"OS_preprocessed files missing under: {base_dir}")

    label_level = str(cfg.get("label_level", "major")).lower()
    if label_level not in {"major", "minor", "pair"}:
        raise ValueError("label_level must be one of {major, minor, pair}")

    label_col = {
        "major": "major_problem_type",
        "minor": "minor_problem_type",
        "pair": "pair_label",
    }[label_level]

    # 标注表负责提供 case_id -> 故障标签 的映射关系。
    labels_df = pd.read_csv(labels_path, encoding="utf-8")
    labels_df["case_id"] = labels_df["case_id"].astype(str)
    if label_level == "pair":
        labels_df["pair_label"] = (
            labels_df["major_problem_type"].astype(str)
            + "||"
            + labels_df["minor_problem_type"].astype(str)
        )
    labels_df = labels_df.set_index("case_id")

    preferred_fields = cfg.get("preferred_content_fields", ["Content", "EventTemplate", "EventId"])
    max_cases = cfg.get("max_cases", None)
    dataset_name = str(cfg.get("name", "OS_preprocessed"))

    records: List[Dict[str, Any]] = []
    case_files = sorted(cases_dir.glob("*.csv"))
    for case_path in tqdm(case_files, desc="load_os_preprocessed"):
        case_id = case_path.stem
        if case_id not in labels_df.index:
            continue

        df = pd.read_csv(case_path, low_memory=False)
        text_col = None
        for c in preferred_fields:
            if c in df.columns:
                text_col = c
                break
        if text_col is None:
            raise ValueError(f"No usable text column in {case_path}. Expected one of {preferred_fields}")

        # 每个 case 文件对应一条训练/评估样本，内部是一串按时间排列的日志行。
        raw_logs = [str(x) for x in df[text_col].fillna("").tolist() if str(x).strip()]
        content_sequence = preprocess_sequence(raw_logs, cfg)

        row = labels_df.loc[case_id]
        fault_type = str(row[label_col])
        ts_start = str(row.get("execution_start_time", ""))
        ts_end = str(row.get("execution_end_time", ""))

        records.append(
            _case_record(
                case_id=case_id,
                fault_type=fault_type,
                raw_logs=raw_logs,
                content_sequence=content_sequence,
                dataset_name=dataset_name,
                timestamp_start=ts_start,
                timestamp_end=ts_end,
            )
        )

        if max_cases is not None and len(records) >= int(max_cases):
            break

    assert records, "No OS_preprocessed cases loaded."
    return records



def load_openstack_cases(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Lightweight OpenStack adapter.

    Assumption:
    - cfg['openstack_jsonl_path'] exists
    - each JSONL item has case_id, fault_type, raw_logs(list)
    """
    p = Path(str(cfg.get("openstack_jsonl_path", "")))
    if not p.exists():
        raise FileNotFoundError(f"OpenStack JSONL not found: {p}")

    records = []
    for item in load_jsonl(p):
        raw_logs = [str(x) for x in item.get("raw_logs", item.get("logs", []))]
        content_sequence = preprocess_sequence(raw_logs, cfg)
        records.append(
            _case_record(
                case_id=str(item["case_id"]),
                fault_type=str(item["fault_type"]),
                raw_logs=raw_logs,
                content_sequence=content_sequence,
                dataset_name=str(cfg.get("name", "openstack")),
                timestamp_start=str(item.get("timestamp_start", "")),
                timestamp_end=str(item.get("timestamp_end", "")),
            )
        )
    assert records, "No OpenStack records loaded."
    return records



def load_hardware_public_cases(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Lightweight hardware-fault adapter.

    Approximation:
    - expects a CSV with columns: case_id, fault_type, log_line, timestamp(optional)
    - groups by case_id preserving file order.
    """
    p = Path(str(cfg.get("hardware_csv_path", "")))
    if not p.exists():
        raise FileNotFoundError(f"Hardware CSV not found: {p}")

    df = pd.read_csv(p, low_memory=False)
    required = {"case_id", "fault_type", "log_line"}
    if not required.issubset(df.columns):
        raise ValueError(f"Hardware CSV missing columns: {required - set(df.columns)}")

    records: List[Dict[str, Any]] = []
    for case_id, g in df.groupby("case_id", sort=False):
        raw_logs = [str(x) for x in g["log_line"].fillna("").tolist() if str(x).strip()]
        content_sequence = preprocess_sequence(raw_logs, cfg)
        fault_type = str(g["fault_type"].iloc[0])
        ts_start = str(g["timestamp"].iloc[0]) if "timestamp" in g.columns else ""
        ts_end = str(g["timestamp"].iloc[-1]) if "timestamp" in g.columns else ""
        records.append(
            _case_record(
                case_id=str(case_id),
                fault_type=fault_type,
                raw_logs=raw_logs,
                content_sequence=content_sequence,
                dataset_name=str(cfg.get("name", "hardware_public")),
                timestamp_start=ts_start,
                timestamp_end=ts_end,
            )
        )

    assert records, "No hardware records loaded."
    return records



def load_custom_jsonl_cases(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Custom/private dataset adapter via generic JSONL schema."""
    p = Path(str(cfg.get("custom_jsonl_path", "")))
    if not p.exists():
        raise FileNotFoundError(f"custom_jsonl_path not found: {p}")

    out: List[Dict[str, Any]] = []
    for item in load_jsonl(p):
        raw_logs = [str(x) for x in item.get("raw_logs", item.get("logs", []))]
        content_sequence = preprocess_sequence(raw_logs, cfg)
        out.append(
            _case_record(
                case_id=str(item["case_id"]),
                fault_type=str(item["fault_type"]),
                raw_logs=raw_logs,
                content_sequence=content_sequence,
                dataset_name=str(item.get("dataset_name", cfg.get("name", "custom"))),
                timestamp_start=str(item.get("timestamp_start", "")),
                timestamp_end=str(item.get("timestamp_end", "")),
            )
        )
    assert out, "No custom JSONL records loaded."
    return out



def load_sample_jsonl_cases(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Tiny synthetic sample adapter for smoke testing."""
    p = Path(str(cfg.get("sample_jsonl_path", "")))
    if not p.exists():
        raise FileNotFoundError(f"sample_jsonl_path not found: {p}")

    out: List[Dict[str, Any]] = []
    for item in load_jsonl(p):
        raw_logs = [str(x) for x in item.get("raw_logs", [])]
        content_sequence = preprocess_sequence(raw_logs, cfg)
        out.append(
            _case_record(
                case_id=str(item["case_id"]),
                fault_type=str(item["fault_type"]),
                raw_logs=raw_logs,
                content_sequence=content_sequence,
                dataset_name=str(item.get("dataset_name", cfg.get("name", "sample"))),
                timestamp_start=str(item.get("timestamp_start", "")),
                timestamp_end=str(item.get("timestamp_end", "")),
            )
        )
    assert out, "No sample records loaded."
    return out



def build_processed_cases(dataset_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Dispatch adapter and return processed case list."""
    adapter = str(dataset_cfg.get("adapter", "os_preprocessed")).lower()

    # 统一把不同来源的数据适配成同一套 case schema，后续模块就不需要关心原始来源。
    if adapter == "os_preprocessed":
        return load_os_preprocessed_cases(dataset_cfg)
    if adapter == "openstack":
        return load_openstack_cases(dataset_cfg)
    if adapter == "hardware_public":
        return load_hardware_public_cases(dataset_cfg)
    if adapter == "custom_jsonl":
        return load_custom_jsonl_cases(dataset_cfg)
    if adapter == "sample_jsonl":
        return load_sample_jsonl_cases(dataset_cfg)

    raise ValueError(f"Unsupported adapter: {adapter}")
