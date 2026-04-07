from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from .fols import build_token_document_frequency, summarize_case
from .prompting import build_prompt, normalize_predicted_label, parse_fault_and_explanation



def load_generation_model(base_model_name: str, adapter_path: str | None = None) -> Tuple[Any, Any]:
    """Load base model and optional LoRA adapter for generation."""
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if adapter_path:
        p = Path(adapter_path)
        if p.exists():
            # 推理时把训练得到的 adapter 叠加到底模上。
            model = PeftModel.from_pretrained(model, str(p))

    model.eval()
    return model, tokenizer



def generate_response(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """Generate one response string from prompt."""
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = temperature > 0
    with torch.no_grad():
        # 温度为 0 时走确定性生成，便于分类评估可复现。
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    return decoded[len(prompt):].strip() if decoded.startswith(prompt) else decoded.strip()


# Backward-compatible alias for any local imports that still use the old name.
_generate_response = generate_response



def _summary_for_variant(
    case: Dict[str, Any],
    all_cases: Sequence[Dict[str, Any]],
    fols_cfg: Dict[str, Any],
    variant: str,
) -> List[str]:
    if variant == "without_fols":
        # 消融场景下可以直接把完整清洗日志送给模型，不做摘要。
        return [str(x) for x in case.get("content_sequence", [])]

    method_map = {
        "full_loginsight": "dbscan",
        "kmeans_replace": "kmeans",
        "agglomerative_replace": "agglomerative",
    }
    method = method_map.get(variant, fols_cfg.get("clustering_method", "dbscan"))

    doc_freq = build_token_document_frequency(all_cases)
    total_cases = len(all_cases)
    result = summarize_case(case, doc_freq, total_cases, fols_cfg, method=method)
    return [str(x) for x in result.get("fault_summary", [])]



def infer_cases(
    cases: Sequence[Dict[str, Any]],
    all_cases_for_fols: Sequence[Dict[str, Any]],
    fault_type_list: Sequence[str],
    model: Any,
    tokenizer: Any,
    instruction_template: str,
    fols_cfg: Dict[str, Any],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    variant: str = "full_loginsight",
) -> List[Dict[str, Any]]:
    """Run inference and return prediction records."""
    known = sorted(set(str(x) for x in fault_type_list))
    preds: List[Dict[str, Any]] = []

    for case in tqdm(cases, desc="infer"):
        summary_lines = _summary_for_variant(case, all_cases_for_fols, fols_cfg, variant)
        input_text = "Log sequence: " + "\n".join(f"- {x}" for x in summary_lines)
        instruction = instruction_template.format(fault_type_list=known)
        prompt = build_prompt(instruction, input_text)
        raw_output = generate_response(model, tokenizer, prompt, max_new_tokens, temperature, top_p)

        fault, explanation, parse_valid = parse_fault_and_explanation(raw_output)
        pred_fault = normalize_predicted_label(fault, known)

        # 同时保留规范化结果和原始输出，方便后续分析模型为什么答错。
        preds.append(
            {
                "case_id": case.get("case_id"),
                "dataset_name": case.get("dataset_name", ""),
                "fault_type": str(case.get("fault_type", "")),
                "pred_fault_type": pred_fault,
                "pred_explanation": explanation,
                "parse_valid": bool(parse_valid),
                "raw_output": raw_output,
            }
        )

    return preds
