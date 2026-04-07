from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

from sklearn.model_selection import train_test_split



def _weak_explanation(summary_lines: Sequence[str], top_k: int = 3) -> str:
    if not summary_lines:
        return "[Approximate] No salient logs were retained by FOLS."
    # 这里不是人工标注解释，只是把最显著的几条摘要日志拼成弱监督解释。
    top = list(summary_lines[:max(1, top_k)])
    joined = " | ".join(top)
    return f"[Approximate] Diagnosis is based on top summary signals: {joined}"



def _build_input_text(summary_lines: Sequence[str]) -> str:
    body = "\n".join(f"- {x}" for x in summary_lines)
    return f"Log sequence: {body}" if body else "Log sequence: (empty)"



def build_instruction_records(
    fols_cases: Sequence[Dict[str, Any]],
    instruction_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build supervised instruction records from FOLS outputs."""
    assert len(fols_cases) > 0, "No FOLS cases provided."

    fault_type_list = sorted({str(x["fault_type"]) for x in fols_cases})
    template = str(
        instruction_cfg.get(
            "instruction_template",
            "Your task is to determine what type of fault a given set of log information belongs to. "
            "Here are the possible fault types in our data scenario: {fault_type_list}. "
            "Please determine its fault type based on the log sequence I input and provide your explanation.",
        )
    )

    mode = str(instruction_cfg.get("explanation_mode", "weak")).lower()
    weak_top_k = int(instruction_cfg.get("weak_top_k", 3))

    records: List[Dict[str, Any]] = []
    for i, case in enumerate(fols_cases):
        label = str(case["fault_type"])
        summary_lines = [str(x) for x in case.get("fault_summary", [])]

        if mode == "gold":
            explanation = str(case.get("gold_explanation", "[Approximate] Gold explanation is unavailable."))
        elif mode == "external_llm":
            explanation = "[Approximate] External-LLM draft explanation placeholder. Manual review required."
        else:
            explanation = _weak_explanation(summary_lines, top_k=weak_top_k)

        # 训练样本采用典型 instruction tuning 结构：指令 + 输入日志摘要 + 目标输出。
        record = {
            "id": f"{case.get('dataset_name', 'dataset')}_{case.get('case_id')}_{i}",
            "instruction": template.format(fault_type_list=fault_type_list),
            "input": _build_input_text(summary_lines),
            "output": f"Fault Type: {label}\nExplanation: {explanation}",
            "fault_type": label,
            "dataset_name": case.get("dataset_name", ""),
            "case_id": case.get("case_id", ""),
        }
        records.append(record)

    return records



def split_instruction_records(
    records: Sequence[Dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split instruction records into train/validation with safe stratification fallback."""
    assert 0.0 < val_ratio < 1.0, "val_ratio must be in (0,1)."
    labels = [str(r["fault_type"]) for r in records]

    unique_classes = len(set(labels))
    val_size = max(1, int(round(len(records) * val_ratio)))
    can_stratify = unique_classes > 1 and val_size >= unique_classes

    # 当样本太少或类别过多时，分层采样可能失败，这里做安全回退。
    train, val = train_test_split(
        list(records),
        test_size=val_ratio,
        random_state=seed,
        stratify=labels if can_stratify else None,
    )
    return train, val



def summarize_label_distribution(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Simple helper for logging class distribution."""
    return dict(Counter(str(r["fault_type"]) for r in records))
