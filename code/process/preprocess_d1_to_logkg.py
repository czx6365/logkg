from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass
class SNIndex:
    times: np.ndarray
    event_ids: np.ndarray
    templates: np.ndarray
    contents: np.ndarray


HEX_RE = re.compile(r"\b0x[0-9a-fA-F]+\b")
NUM_RE = re.compile(r"\b\d+\b")
SPACE_RE = re.compile(r"\s+")


def normalize_message(msg: str) -> str:
    text = str(msg).strip().lower()
    text = HEX_RE.sub("<HEX>", text)
    text = NUM_RE.sub("<NUM>", text)
    text = SPACE_RE.sub(" ", text)
    return text


def build_template(msg: str, server_model: str | None, use_server_model: bool) -> str:
    base = normalize_message(msg)
    if use_server_model and server_model:
        return f"{server_model}||{base}"
    return base


def build_event_id(template: str) -> str:
    return hashlib.md5(template.encode("utf-8")).hexdigest()[:16]


def load_logs(log_path: Path, use_server_model: bool) -> tuple[pd.DataFrame, Dict[str, str]]:
    logs = pd.read_csv(log_path)
    required = {"sn", "time", "msg"}
    missing = required - set(logs.columns)
    if missing:
        raise ValueError(f"log file missing required columns: {sorted(missing)}")

    if "server_model" not in logs.columns:
        logs["server_model"] = None

    logs["time"] = pd.to_datetime(logs["time"], errors="coerce")
    logs = logs.dropna(subset=["sn", "time", "msg"]).copy()

    logs["EventTemplate"] = [
        build_template(msg, None if pd.isna(sm) else str(sm), use_server_model)
        for msg, sm in zip(logs["msg"].tolist(), logs["server_model"].tolist())
    ]
    logs["EventId"] = logs["EventTemplate"].map(build_event_id)
    logs = logs.sort_values(["sn", "time"]).reset_index(drop=True)

    template_map = logs.drop_duplicates("EventId")[["EventId", "EventTemplate"]]
    template_dict = {row.EventId: row.EventTemplate for row in template_map.itertuples(index=False)}
    return logs, template_dict


def build_sn_index(logs: pd.DataFrame) -> Dict[str, SNIndex]:
    index: Dict[str, SNIndex] = {}
    for sn, g in logs.groupby("sn", sort=False):
        index[str(sn)] = SNIndex(
            times=g["time"].to_numpy(dtype="datetime64[ns]"),
            event_ids=g["EventId"].astype(str).to_numpy(dtype=object),
            templates=g["EventTemplate"].astype(str).to_numpy(dtype=object),
            contents=g["msg"].astype(str).to_numpy(dtype=object),
        )
    return index


def extract_case_rows(
    idx: SNIndex | None,
    fault_time: np.datetime64,
    window: np.timedelta64,
    fallback_all_before: bool,
) -> pd.DataFrame:
    if idx is None:
        return pd.DataFrame(
            {
                "EventId": ["NO_LOG"],
                "EventTemplate": ["__NO_LOG__"],
                "Content": ["__NO_LOG__"],
            }
        )

    left = fault_time - window
    end = np.searchsorted(idx.times, fault_time, side="right")
    start = np.searchsorted(idx.times, left, side="left")
    sel = np.arange(start, end, dtype=int)

    if sel.size == 0 and fallback_all_before and end > 0:
        sel = np.arange(0, end, dtype=int)

    if sel.size == 0:
        return pd.DataFrame(
            {
                "EventId": ["NO_LOG"],
                "EventTemplate": ["__NO_LOG__"],
                "Content": ["__NO_LOG__"],
            }
        )

    return pd.DataFrame(
        {
            "EventId": idx.event_ids[sel],
            "EventTemplate": idx.templates[sel],
            "Content": idx.contents[sel],
        }
    )


def preprocess(
    d1_dir: Path,
    log_file: str,
    label_file: str,
    output_dir: Path,
    history_hours: int,
    use_server_model: bool,
    fallback_all_before: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    case_dir = output_dir / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)

    logs, template_dict = load_logs(d1_dir / log_file, use_server_model=use_server_model)
    sn_index = build_sn_index(logs)

    labels = pd.read_csv(d1_dir / label_file)
    required = {"sn", "fault_time", "label"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"label file missing required columns: {sorted(missing)}")

    labels["fault_time"] = pd.to_datetime(labels["fault_time"], errors="coerce")
    labels = labels.dropna(subset=["sn", "fault_time", "label"]).copy()

    config: Dict[str, list[str]] = {}
    window = np.timedelta64(history_hours, "h")
    written = 0

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
        case_df.to_csv(case_dir / f"{case_id}.csv", index=False)

        label_name = f"label_{label}"
        config.setdefault(label_name, []).append(case_id)
        written += 1

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    tpl_df = pd.DataFrame(
        {
            "EventId": list(template_dict.keys()),
            "EventTemplate": list(template_dict.values()),
        }
    )
    tpl_df.to_csv(output_dir / "template_map.csv", index=False)

    meta = {
        "log_file": log_file,
        "label_file": label_file,
        "history_hours": history_hours,
        "use_server_model": use_server_model,
        "fallback_all_before": fallback_all_before,
        "case_count": written,
        "template_count": int(len(template_dict)),
        "output_case_dir": str(case_dir),
    }
    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Preprocess done. cases={written}, templates={len(template_dict)}")
    print(f"case_dir: {case_dir}")
    print(f"config : {output_dir / 'config.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess D1 dataset to LogKG case format.")
    parser.add_argument("--d1_dir", type=Path, default=ROOT_DIR / "data" / "D1")
    parser.add_argument("--log_file", type=str, default="preliminary_sel_log_dataset.csv")
    parser.add_argument("--label_file", type=str, default="preliminary_train_label_dataset_s.csv")
    parser.add_argument("--output_dir", type=Path, default=ROOT_DIR / "data" / "D1_preprocessed_s")
    parser.add_argument("--history_hours", type=int, default=24)
    parser.add_argument("--use_server_model", action="store_true", default=False)
    parser.add_argument("--fallback_all_before", action="store_true", default=False)
    args = parser.parse_args()

    preprocess(
        d1_dir=args.d1_dir,
        log_file=args.log_file,
        label_file=args.label_file,
        output_dir=args.output_dir,
        history_hours=args.history_hours,
        use_server_model=args.use_server_model,
        fallback_all_before=args.fallback_all_before,
    )


if __name__ == "__main__":
    main()
