from __future__ import annotations

import re
from typing import Sequence, Tuple


MAJOR_FAULT_TYPE_GUIDANCE = {
    "版本缺陷": {
        "definition": (
            "Use this label when the failure is primarily caused by a product/version/release issue, "
            "such as a bad release artifact, missing or incorrect change records, or a version-level "
            "compatibility/regression not attributable to the test itself."
        ),
        "not_this": (
            "Do not use this for ordinary test script bugs, missing runtime dependencies in the test, "
            "wrong baseline/distribution, or board/qemu environment instability."
        ),
    },
    "用例/代码问题": {
        "definition": (
            "Use this label when the failure is primarily caused by the test case, test code, test design, "
            "missing execution dependency, or the case not being adapted to known code changes."
        ),
        "not_this": (
            "Do not use this when logs clearly indicate unsupported product/baseline distribution, explicit "
            "board/qemu hardware-resource or network faults, or obvious release-version defects."
        ),
    },
    "单板/qemu问题": {
        "definition": (
            "Use this label only when logs contain concrete evidence of board/qemu/environment faults, such as "
            "hardware resource not supported, board hardware fault, qemu instability, board/qemu network failure, "
            "or other low-level environment problems outside the test logic."
        ),
        "not_this": (
            "Do not use this merely because the logs contain panic, reboot, timeout, no such file, permission denied, "
            "read-only file system, execute failure, or generic command failure. Those symptoms alone are insufficient "
            "without explicit board/qemu/environment evidence."
        ),
    },
    "用例基线/分发问题": {
        "definition": (
            "Use this label when the failure is caused by baseline selection, case distribution, unsupported product "
            "target, missing required distributed resources, or a requirement that has not yet been delivered."
        ),
        "not_this": (
            "Do not use this for intrinsic test code bugs, explicit board/qemu hardware issues, or version-release "
            "defects in the product itself."
        ),
    },
    "自动化工程问题": {
        "definition": (
            "Use this label when the failure is caused by automation or test engineering infrastructure, such as "
            "framework issues, lab server/environment problems, reserved-memory configuration, or pipeline/setup issues."
        ),
        "not_this": (
            "Do not use this for test case logic bugs, board/qemu hardware-resource faults, or baseline/distribution mismatch."
        ),
    },
    "人工执行操作问题": {
        "definition": (
            "Use this label when the failure is caused by human operation mistakes, such as choosing the wrong board, "
            "manual setup mistakes, or other operator-side execution errors."
        ),
        "not_this": (
            "Do not use this when the logs mainly indicate autonomous software faults, board/qemu instability, or test code defects."
        ),
    },
    "UNKNOWN_MAJOR": {
        "definition": (
            "Use this label only when the available log evidence is insufficient to support any known fault type."
        ),
        "not_this": (
            "Do not use this if one known label is reasonably supported by specific log evidence."
        ),
    },
}



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


def format_fault_type_list(fault_type_list: Sequence[str]) -> str:
    """Format candidate labels into a readable comma-separated string."""
    labels = [str(x).strip() for x in fault_type_list if str(x).strip()]
    return ", ".join(labels) if labels else "unknown type"


def _build_major_fault_type_guidance(fault_type_list: Sequence[str]) -> str:
    """Append label definitions when the candidate set matches the OS major taxonomy."""
    labels = [str(x).strip() for x in fault_type_list if str(x).strip()]
    if not labels:
        return ""

    guided_labels = [label for label in labels if label in MAJOR_FAULT_TYPE_GUIDANCE]
    if len(guided_labels) < max(3, len(labels) - 1):
        return ""

    parts = ["Fault type definitions and counterexamples:"]
    for label in labels:
        guidance = MAJOR_FAULT_TYPE_GUIDANCE.get(label)
        if guidance is None:
            continue
        parts.append(
            f"- {label}: {guidance['definition']} Counterexample: {guidance['not_this']}"
        )

    parts.append(
        "Decision rules: Prefer the most direct root-cause evidence rather than the loudest symptom. "
        "If the logs mainly show the test script, test dependency, or case adaptation failing, prefer 用例/代码问题. "
        "If the logs mainly show wrong target product, wrong baseline, or missing distributed resource, prefer 用例基线/分发问题. "
        "Choose 单板/qemu问题 only when there is explicit board/qemu/environment evidence, not just crash-like symptoms."
    )
    return "\n".join(parts)


def build_inference_instruction(
    instruction_template: str,
    fault_type_list: Sequence[str],
    question: str | None = None,
) -> str:
    """Build a paper-style instruction with enhanced label-boundary guidance."""
    formatted_fault_types = format_fault_type_list(fault_type_list)
    paper_style_instruction = (
        "Your task is to determine what type of fault a given set of log information belongs to. "
        f"Here are the possible fault types in our data scenario: {formatted_fault_types}. "
        "Please determine its fault type based on the log sequence I input and provide your explanation."
    )
    configured_instruction = str(instruction_template).format(
        fault_type_list=formatted_fault_types
    ).strip()
    enhanced_guidance = _build_major_fault_type_guidance(fault_type_list)

    instruction_parts = [paper_style_instruction]
    if configured_instruction and configured_instruction != paper_style_instruction:
        instruction_parts.append(configured_instruction)
    if enhanced_guidance:
        instruction_parts.append(enhanced_guidance)

    strict_schema = (
        "Choose exactly one label from the candidate fault type list. "
        "If none fits, answer with unknown type.\n"
        "Respond in exactly this format:\n"
        "Fault Type: <one label>\n"
        "Explanation: <brief evidence-based explanation>"
    )
    if question and str(question).strip():
        strict_schema += f"\nUser request: {str(question).strip()}"
    instruction_parts.append(strict_schema)
    return "\n\n".join(instruction_parts)



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
