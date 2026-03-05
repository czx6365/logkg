from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

HEX_RE = re.compile(r"\b0x[0-9a-fA-F]+\b")
NUM_RE = re.compile(r"\b\d+\b")
SPACE_RE = re.compile(r"\s+")
TAG_RE = re.compile(r"<[^>]+>")
PRELINE_SPAN_OPEN_RE = re.compile(
    r"<span[^>]*white-space\s*:\s*pre-line[^>]*>",
    flags=re.IGNORECASE,
)
BR_RE = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)


def normalize_message(msg: str) -> str:
    text = str(msg).strip().lower()
    text = HEX_RE.sub("<HEX>", text)
    text = NUM_RE.sub("<NUM>", text)
    text = SPACE_RE.sub(" ", text)
    return text


def build_event_id(template: str) -> str:
    return hashlib.md5(template.encode("utf-8")).hexdigest()[:16]


def clean_label(value: object, default: str) -> str:
    s = "" if value is None else str(value).strip()
    return s or default


def extract_log_lines_from_html(html_text: str) -> List[str]:
    match = PRELINE_SPAN_OPEN_RE.search(html_text)
    if match:
        start = match.end()
        close_idx = html_text.find("</span>", start)
        if close_idx == -1:
            close_idx = len(html_text)
        log_block = html_text[start:close_idx]
    else:
        # Fallback: keep all visible text when the expected span does not exist.
        log_block = html_text

    log_block = BR_RE.sub("\n", log_block)
    log_block = TAG_RE.sub("", log_block)
    log_block = html.unescape(log_block)
    lines = [line.strip() for line in log_block.splitlines() if line.strip()]
    return lines


def build_case_rows(lines: Iterable[str]) -> pd.DataFrame:
    rows: List[Tuple[str, str, str]] = []
    for line in lines:
        template = normalize_message(line)
        if not template:
            continue
        rows.append((build_event_id(template), template, line))

    if not rows:
        rows = [("NO_LOG", "__NO_LOG__", "__NO_LOG__")]

    return pd.DataFrame(rows, columns=["EventId", "EventTemplate", "Content"])


def preprocess_os_data(
    os_data_dir: Path,
    output_dir: Path,
    default_label_level: str = "minor",
) -> None:
    if default_label_level not in {"major", "minor", "pair"}:
        raise ValueError("default_label_level must be one of: major, minor, pair")

    output_dir.mkdir(parents=True, exist_ok=True)
    case_dir = output_dir / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)

    template_dict: Dict[str, str] = {}
    config_major: Dict[str, List[str]] = {}
    config_minor: Dict[str, List[str]] = {}
    config_pair: Dict[str, List[str]] = {}
    label_rows: List[Dict[str, str]] = []

    parse_errors: List[str] = []
    folder_count = 0
    case_count = 0

    for folder in sorted(os_data_dir.iterdir()):
        if not folder.is_dir():
            continue
        folder_count += 1

        json_path = folder / "data.json"
        html_files = sorted(folder.glob("*.html"))
        if not json_path.exists() or len(html_files) != 1:
            parse_errors.append(
                f"{folder.name}: expected data.json + exactly one html, got json={json_path.exists()}, html_count={len(html_files)}"
            )
            continue

        try:
            record = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            parse_errors.append(f"{folder.name}: failed to parse data.json: {exc}")
            continue

        html_path = html_files[0]
        try:
            html_text = html_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            parse_errors.append(f"{folder.name}: failed to read html: {exc}")
            continue

        lines = extract_log_lines_from_html(html_text)
        case_df = build_case_rows(lines)
        case_id = folder.name
        case_df.to_csv(case_dir / f"{case_id}.csv", index=False)
        case_count += 1

        for row in case_df[["EventId", "EventTemplate"]].itertuples(index=False):
            template_dict[str(row.EventId)] = str(row.EventTemplate)

        major_label = clean_label(record.get("major_problem_type"), "UNKNOWN_MAJOR")
        minor_label = clean_label(record.get("minor_problem_type"), "UNKNOWN_MINOR")
        pair_label = f"{major_label}||{minor_label}"

        config_major.setdefault(major_label, []).append(case_id)
        config_minor.setdefault(minor_label, []).append(case_id)
        config_pair.setdefault(pair_label, []).append(case_id)

        label_rows.append(
            {
                "case_id": case_id,
                "task_id": str(record.get("task_id", "")),
                "number": str(record.get("number", "")),
                "result": str(record.get("result", "")),
                "major_problem_type": major_label,
                "minor_problem_type": minor_label,
                "execution_start_time": str(record.get("execution_start_time", "")),
                "execution_end_time": str(record.get("execution_end_time", "")),
                "manual_analysis_description": str(record.get("manual_analysis_description", "")),
                "json_path": str(json_path),
                "html_path": str(html_path),
            }
        )

    default_config = {
        "major": config_major,
        "minor": config_minor,
        "pair": config_pair,
    }[default_label_level]

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(default_config, f, ensure_ascii=False, indent=2)
    with open(output_dir / "config_major.json", "w", encoding="utf-8") as f:
        json.dump(config_major, f, ensure_ascii=False, indent=2)
    with open(output_dir / "config_minor.json", "w", encoding="utf-8") as f:
        json.dump(config_minor, f, ensure_ascii=False, indent=2)
    with open(output_dir / "config_pair.json", "w", encoding="utf-8") as f:
        json.dump(config_pair, f, ensure_ascii=False, indent=2)

    tpl_df = pd.DataFrame(
        {
            "EventId": list(template_dict.keys()),
            "EventTemplate": list(template_dict.values()),
        }
    )
    tpl_df.to_csv(output_dir / "template_map.csv", index=False)

    labels_df = pd.DataFrame(label_rows)
    labels_df.to_csv(output_dir / "case_labels.csv", index=False, encoding="utf-8")

    major_count = Counter(labels_df["major_problem_type"].tolist()) if not labels_df.empty else Counter()
    minor_count = Counter(labels_df["minor_problem_type"].tolist()) if not labels_df.empty else Counter()
    pair_count = Counter(
        [f"{row['major_problem_type']}||{row['minor_problem_type']}" for row in label_rows]
    )

    meta = {
        "os_data_dir": str(os_data_dir),
        "folder_count": folder_count,
        "case_count": case_count,
        "default_label_level": default_label_level,
        "major_class_count": len(config_major),
        "minor_class_count": len(config_minor),
        "pair_class_count": len(config_pair),
        "template_count": int(len(template_dict)),
        "top_major_classes": major_count.most_common(10),
        "top_minor_classes": minor_count.most_common(10),
        "top_pair_classes": pair_count.most_common(10),
        "parse_error_count": len(parse_errors),
        "parse_errors": parse_errors[:50],
        "output_case_dir": str(case_dir),
    }
    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(
        "Preprocess done. "
        f"folders={folder_count}, cases={case_count}, "
        f"major={len(config_major)}, minor={len(config_minor)}, pair={len(config_pair)}, "
        f"templates={len(template_dict)}, parse_errors={len(parse_errors)}"
    )
    print(f"case_dir: {case_dir}")
    print(f"default config: {output_dir / 'config.json'}")
    print(f"major  config: {output_dir / 'config_major.json'}")
    print(f"minor  config: {output_dir / 'config_minor.json'}")
    print(f"pair   config: {output_dir / 'config_pair.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess OS_data dataset to LogKG case format.")
    parser.add_argument("--os_data_dir", type=Path, default=Path("../data/OS_data"))
    parser.add_argument("--output_dir", type=Path, default=Path("../data/OS_preprocessed"))
    parser.add_argument(
        "--default_label_level",
        type=str,
        default="minor",
        choices=["major", "minor", "pair"],
        help="Which label granularity to write into config.json.",
    )
    args = parser.parse_args()

    preprocess_os_data(
        os_data_dir=args.os_data_dir,
        output_dir=args.output_dir,
        default_label_level=args.default_label_level,
    )


if __name__ == "__main__":
    main()
