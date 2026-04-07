from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import LogInsightAgent
from src.utils import ensure_dir


def _load_log_lines(args: argparse.Namespace) -> list[str]:
    if args.logs_file is not None:
        with args.logs_file.open("r", encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f if line.strip()]
    return [str(line) for line in args.log_line if str(line).strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--logs-file", type=Path)
    group.add_argument("--log-line", action="append", default=[])
    parser.add_argument("--fault-type", action="append", default=[])
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--adapter-path", type=str)
    parser.add_argument("--question", type=str, default="Please diagnose the fault type and explain your judgment.")
    parser.add_argument("--case-id", type=str, default="adhoc_case")
    parser.add_argument("--dataset-name", type=str, default="adhoc")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    log_lines = _load_log_lines(args)
    agent = LogInsightAgent.from_config(
        args.config,
        fault_type_list=args.fault_type or None,
        model_name=args.model_name,
        adapter_path=args.adapter_path,
    )
    result = agent.diagnose(
        log_lines,
        question=args.question,
        fault_type_list=args.fault_type or None,
        case_id=args.case_id,
        dataset_name=args.dataset_name,
    )

    if args.output is not None:
        ensure_dir(args.output)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
