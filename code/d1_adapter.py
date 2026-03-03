from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SNLogIndex:
    # 某台机器(sn)的日志时间序列与对应事件序列（按时间升序）
    times: np.ndarray
    events: np.ndarray


def _build_event_id(msg: str, server_model: Optional[str], use_server_model: bool) -> str:
    # 可选把 server_model 拼到事件ID里，降低不同机型同文案冲突
    if use_server_model and server_model is not None:
        return f"{server_model}||{msg}"
    return msg


def _build_sn_index(log_df: pd.DataFrame, use_server_model: bool) -> Dict[str, SNLogIndex]:
    # 校验日志最小字段
    required = {"sn", "time", "msg"}
    missing = required - set(log_df.columns)
    if missing:
        raise ValueError(f"log_df missing required columns: {sorted(missing)}")

    # 统一时间格式并清理关键字段缺失值
    df = log_df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["sn", "time", "msg"])

    # 有些文件没有 server_model，补空列保持流程一致
    if "server_model" not in df.columns:
        df["server_model"] = None

    # 构建 sn -> (time[], event[]) 的索引，后续可高效按时间窗口切片
    sn_index: Dict[str, SNLogIndex] = {}
    for sn, group in df.groupby("sn", sort=False):
        g = group.sort_values("time")
        events = np.array(
            [
                _build_event_id(str(msg), None if pd.isna(sm) else str(sm), use_server_model)
                for msg, sm in zip(g["msg"].tolist(), g["server_model"].tolist())
            ],
            dtype=object,
        )
        sn_index[str(sn)] = SNLogIndex(
            times=g["time"].to_numpy(dtype="datetime64[ns]"),
            events=events,
        )
    return sn_index


def build_d1_cases(
    log_df: pd.DataFrame,
    case_df: pd.DataFrame,
    label_col: Optional[str] = "label",
    history_hours: int = 24,
    use_server_model: bool = True,
    fallback_all_before: bool = True,
) -> Tuple[List[str], np.ndarray, Dict[str, pd.DataFrame]]:
    """
    将 D1 表格日志转成 LogKG 可直接使用的 case 格式。

    参数说明:
    - log_df: 原始日志表（包含 sn/time/msg，server_model 可选）
    - case_df: case 清单（必须包含 sn/fault_time，label 可选）
    - label_col: 标签列名，若不存在则返回空标签数组
    - history_hours: 每个 case 使用故障时间前多少小时的日志
    - use_server_model: 是否把 server_model 拼入 EventId
    - fallback_all_before: 窗口内无日志时，是否回退到故障前全部日志

    返回:
    - case_name_list: case 唯一ID列表（case_df 一行对应一个 case）
    - case_truth_label: case 标签数组（若 label_col 存在）
    - case_log_df: dict[case_id]，值为仅含 EventId 列的 DataFrame
    """
    # 校验 case 最小字段
    required = {"sn", "fault_time"}
    missing = required - set(case_df.columns)
    if missing:
        raise ValueError(f"case_df missing required columns: {sorted(missing)}")

    # 先把原始日志按 sn 建索引，减少重复扫描
    sn_index = _build_sn_index(log_df, use_server_model=use_server_model)
    fault_time_series = pd.to_datetime(case_df["fault_time"], errors="coerce")

    case_name_list: List[str] = []
    labels: List[int] = []
    case_log_df: Dict[str, pd.DataFrame] = {}
    window = np.timedelta64(history_hours, "h")

    # 一条 (sn, fault_time) 记录转换为一个 case
    for row_idx, (sn_raw, fault_time_raw) in enumerate(zip(case_df["sn"].tolist(), fault_time_series.tolist())):
        if pd.isna(fault_time_raw):
            continue

        sn = str(sn_raw)
        case_id = f"{sn}__{pd.Timestamp(fault_time_raw).strftime('%Y%m%d%H%M%S')}__{row_idx}"
        case_name_list.append(case_id)

        idx = sn_index.get(sn)
        if idx is None:
            # 该 sn 在日志里不存在，填充占位事件避免空序列
            events = np.array(["__NO_LOG__"], dtype=object)
        else:
            ft = np.datetime64(pd.Timestamp(fault_time_raw), "ns")
            left = ft - window

            # 取 [fault_time-history_hours, fault_time] 区间日志
            end = np.searchsorted(idx.times, ft, side="right")
            start = np.searchsorted(idx.times, left, side="left")
            events = idx.events[start:end]

            # 若窗口内无日志，可选回退到故障前全部日志
            if len(events) == 0 and fallback_all_before and end > 0:
                events = idx.events[:end]

            if len(events) == 0:
                events = np.array(["__NO_LOG__"], dtype=object)

        # 输出格式与 LogKG 现有实现保持一致（只依赖 EventId）
        case_log_df[case_id] = pd.DataFrame({"EventId": events})

        if label_col is not None and label_col in case_df.columns:
            labels.append(int(case_df.iloc[row_idx][label_col]))

    if label_col is not None and label_col in case_df.columns:
        case_truth_label = np.array(labels, dtype=int)
    else:
        case_truth_label = np.array([], dtype=int)

    return case_name_list, case_truth_label, case_log_df
