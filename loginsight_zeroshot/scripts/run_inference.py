from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loginsight_zeroshot.src.agent import LogInsightAgent
from loginsight_zeroshot.src.utils import load_jsonl, load_yaml, resolve_path, set_seed


def _build_prediction_record(case: dict, result: dict) -> dict:
    return {
        "mode": result["mode"],
        "case_id": result["case_id"],
        "dataset_name": result["dataset_name"],
        "fault_type": str(case.get("fault_type", "")),
        "pred_fault_type": result["pred_fault_type"],
        "pred_explanation": result["pred_explanation"],
        "parse_valid": result["parse_valid"],
        "raw_output": result["raw_output"],
        "summary_lines": result["summary_lines"],
    }


def _append_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _dedupe_case_ids(records: list[dict]) -> tuple[list[dict], set[str]]:
    """Keep the first saved prediction for each case_id."""
    seen: set[str] = set()
    deduped: list[dict] = []
    for record in records:
        case_id = str(record.get("case_id", ""))
        if case_id in seen:
            continue
        seen.add(case_id)
        deduped.append(record)
    return deduped, seen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", type=str, choices=["qwen", "generative"], default="qwen")
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--adapter-path", type=str)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--save-every", type=int, default=10)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    paths = cfg["paths"]
    infer_cfg = dict(cfg.get("inference", {}))

    processed_path = resolve_path(paths["processed_cases"], args.config.parent)
    pred_path = resolve_path(paths["predictions_path"], args.config.parent)

    instruction_val_value = str(paths.get("instruction_val", "")).strip()
    instruction_val_path = (
        resolve_path(instruction_val_value, args.config.parent)
        if instruction_val_value
        else None
    )

    cases = load_jsonl(processed_path)
    fault_types = sorted({str(x["fault_type"]) for x in cases})

    if (
        bool(infer_cfg.get("use_instruction_val_only", False))
        and instruction_val_path is not None
        and instruction_val_path.exists()
    ):
        val_case_ids = {str(x["case_id"]) for x in load_jsonl(instruction_val_path)}
        infer_cases_list = [x for x in cases if str(x.get("case_id")) in val_case_ids]
    else:
        infer_cases_list = cases

    cv_folds = int(infer_cfg.get("cv_folds", 1) or 1)
    if cv_folds > 1:
        print(
            "warning=cv_folds is ignored for Qwen zero-shot inference; "
            "no training split will be used."
        )

    agent = LogInsightAgent.from_config(
        args.config,
        fault_type_list=fault_types,
        model_name=args.model_name,
        adapter_path=args.adapter_path,
    )

    existing_preds: list[dict] = []
    completed_case_ids: set[str] = set()
    if args.resume and pred_path.exists():
        existing_preds = load_jsonl(pred_path)
        deduped_preds, completed_case_ids = _dedupe_case_ids(existing_preds)
        if len(deduped_preds) != len(existing_preds):
            pred_path.unlink()
            _append_jsonl(deduped_preds, pred_path)
            existing_preds = deduped_preds

    pending_cases = [x for x in infer_cases_list if str(x.get("case_id", "")) not in completed_case_ids]
    if not args.resume and pred_path.exists():
        pred_path.unlink()

    total_cases = len(infer_cases_list)
    completed_count = len(completed_case_ids)
    remaining_count = len(pending_cases)

    print(f"mode={args.mode}", flush=True)
    print(f"resume_enabled={args.resume}", flush=True)
    print(f"total_cases={total_cases}", flush=True)
    print(f"already_completed={completed_count}", flush=True)
    print(f"remaining_cases={remaining_count}", flush=True)
    if completed_count:
        print(
            f"resume_status=continuing from case {completed_count + 1} of {total_cases}",
            flush=True,
        )

    buffer: list[dict] = []
    progress = tqdm(
        pending_cases,
        desc="qwen_zeroshot",
        total=total_cases,
        initial=completed_count,
    )
    for case in progress:
        result = agent.diagnose(
            case.get("content_sequence", []),
            question=None,
            fault_type_list=fault_types,
            case_id=str(case.get("case_id", "")),
            dataset_name=str(case.get("dataset_name", "")),
            mode=args.mode,
        )
        buffer.append(_build_prediction_record(case, result))
        if len(buffer) >= max(1, int(args.save_every)):
            _append_jsonl(buffer, pred_path)
            buffer.clear()

    if buffer:
        _append_jsonl(buffer, pred_path)

    preds = load_jsonl(pred_path) if pred_path.exists() else []
    print(f"inference_cases={len(infer_cases_list)}", flush=True)
    print(f"completed_predictions={len(preds)}", flush=True)
    print(f"predictions_saved={pred_path}", flush=True)


if __name__ == "__main__":
    main()
