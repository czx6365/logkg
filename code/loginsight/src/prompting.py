from __future__ import annotations

import re
from typing import Tuple



def build_prompt(instruction: str, input_text: str, output_text: str | None = None) -> str:
    """Build a simple instruction-tuning prompt format."""
    # 训练和推理都共用同一套 prompt 模板，保证输入分布尽量一致。
    prompt = (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{input_text}\n\n"
        "### Response:\n"
    )
    if output_text is not None:
        prompt += output_text
    return prompt



def parse_fault_and_explanation(raw_output: str) -> Tuple[str, str, bool]:
    """
    Parse model output robustly.

    Priority:
    1) regex with explicit 'Fault Type:' and 'Explanation:'
    2) fallback heuristic using first lines
    """
    text = str(raw_output).strip()

    # 优先走强约束解析：要求模型显式输出 Fault Type 和 Explanation 字段。
    m_fault = re.search(r"Fault\s*Type\s*:\s*(.+)", text, flags=re.IGNORECASE)
    m_expl = re.search(r"Explanation\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if m_fault and m_expl:
        fault = m_fault.group(1).strip().splitlines()[0].strip()
        expl = m_expl.group(1).strip()
        return fault, expl, True

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "unknown type", "", False

    # 如果模型没按模板答，退化成“首行猜标签，其余行当解释”的启发式解析。
    first = lines[0]
    fault = first
    if ":" in first:
        fault = first.split(":", 1)[1].strip() or "unknown type"

    explanation = "\n".join(lines[1:]).strip()
    return (fault if fault else "unknown type"), explanation, False



def normalize_predicted_label(label: str, known_fault_types: list[str]) -> str:
    """Normalize predicted label and allow explicit unknown type."""
    label_norm = str(label).strip()
    if not label_norm:
        return "unknown type"
    if label_norm.lower() == "unknown type":
        return "unknown type"
    if label_norm in known_fault_types:
        return label_norm
    # 不在候选标签集合中的输出统一记为 unknown，避免污染评估标签空间。
    return "unknown type"
