from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, List, Sequence

from .fols import build_token_document_frequency, summarize_case
from .prompting import (
    build_inference_instruction,
    normalize_predicted_label,
    parse_fault_and_explanation,
)
from .utils import load_jsonl, load_yaml, resolve_optional_path, resolve_path

SUPPORTED_MODES = {"qwen", "generative", "zero_shot", "zeroshot"}


class LogInsightAgent:
    def __init__(
        self,
        cfg: Dict[str, Any],
        config_path: Path,
        reference_cases: Sequence[Dict[str, Any]] | None = None,
        fault_type_list: Sequence[str] | None = None,
    ) -> None:
        self.cfg = cfg
        self.config_path = Path(config_path)
        self.reference_cases = list(reference_cases or [])

        self.fols_cfg = dict(cfg.get("fols", {}))
        self.infer_cfg = dict(cfg.get("inference", {}))
        self.instruction_cfg = dict(cfg.get("instruction", {}))
        self.model_cfg = dict(cfg.get("model", {}))

        configured_fault_types = [str(x).strip() for x in (fault_type_list or []) if str(x).strip()]
        if not configured_fault_types:
            configured_fault_types = sorted(
                {
                    str(x.get("fault_type", "")).strip()
                    for x in self.reference_cases
                    if str(x.get("fault_type", "")).strip()
                }
            )
        self.default_fault_type_list = configured_fault_types

        base_cases = self.reference_cases or []
        self._doc_freq = build_token_document_frequency(base_cases) if base_cases else {}
        self._total_cases = len(base_cases)

        self._model: Any | None = None
        self._tokenizer: Any | None = None

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        fault_type_list: Sequence[str] | None = None,
        model_name: str | None = None,
        adapter_path: str | None = None,
    ) -> "LogInsightAgent":
        config_path = Path(config_path)
        cfg = load_yaml(config_path)

        paths = dict(cfg.get("paths", {}))
        processed_path_value = str(paths.get("processed_cases", "")).strip()
        reference_cases: List[Dict[str, Any]] = []
        if processed_path_value:
            processed_path = resolve_path(processed_path_value, config_path.parent)
            if processed_path.exists():
                reference_cases = load_jsonl(processed_path)

        if model_name:
            cfg.setdefault("model", {})
            cfg["model"]["base_model_name"] = model_name

        if adapter_path:
            cfg.setdefault("paths", {})
            cfg["paths"]["adapter_path"] = adapter_path

        return cls(
            cfg=cfg,
            config_path=config_path,
            reference_cases=reference_cases,
            fault_type_list=fault_type_list,
        )

    def _normalize_mode(self, mode: str | None) -> str:
        selected = str(mode or "qwen").strip().lower()
        if selected in {"zero_shot", "zeroshot", "generative"}:
            selected = "qwen"
        if selected not in SUPPORTED_MODES:
            supported = ", ".join(sorted(SUPPORTED_MODES))
            raise ValueError(f"Unsupported mode '{mode}'. Supported modes: {supported}.")
        return selected

    def _ensure_generation_model(self) -> None:
        if self._model is not None:
            return

        try:
            from .infer import load_generation_model
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Generation dependencies are missing. Either install torch/transformers for the transformers backend, "
                "or configure a local llama-cli GGUF backend for Qwen zero-shot mode."
            ) from exc

        model_name = str(self.model_cfg.get("base_model_name", "Qwen/Qwen2.5-7B-Instruct"))
        adapter_path = resolve_optional_path(
            self.cfg.get("paths", {}).get("adapter_path"),
            self.config_path.parent,
        )

        self._model, self._tokenizer = load_generation_model(
            model_name,
            adapter_path=str(adapter_path) if adapter_path is not None else None,
            model_cfg=self.model_cfg,
        )

    def build_case(
        self,
        log_lines: Sequence[str],
        *,
        case_id: str = "adhoc_case",
        dataset_name: str = "adhoc",
    ) -> Dict[str, Any]:
        return {
            "case_id": case_id,
            "dataset_name": dataset_name,
            "fault_type": "",
            "content_sequence": [str(x) for x in log_lines if str(x).strip()],
        }

    def summarize_logs(self, case: Dict[str, Any]) -> Dict[str, Any]:
        if self.reference_cases:
            doc_freq = self._doc_freq
            total_cases = self._total_cases
        else:
            bootstrap_cases = [case]
            doc_freq = build_token_document_frequency(bootstrap_cases)
            total_cases = len(bootstrap_cases)

        return summarize_case(
            case=case,
            doc_freq=doc_freq,
            total_cases=total_cases,
            fols_cfg=self.fols_cfg,
            method=str(self.fols_cfg.get("clustering_method", "dbscan")),
        )

    def _build_instruction(
        self,
        fault_type_list: Sequence[str],
        question: str | None,
    ) -> str:
        template = str(
            self.instruction_cfg.get(
                "instruction_template",
                "You are a log fault diagnosis expert. Determine the most likely fault type from the log sequence. Candidate fault types: {fault_type_list}.",
            )
        )
        return build_inference_instruction(template, fault_type_list, question=question)

    def _active_fault_types(self, fault_type_list: Sequence[str] | None) -> List[str]:
        return [str(x).strip() for x in (fault_type_list or self.default_fault_type_list) if str(x).strip()]

    def _extract_summary_lines(self, case: Dict[str, Any], summary: Dict[str, Any]) -> List[str]:
        summary_lines = [str(x) for x in summary.get("fault_summary", []) if str(x).strip()]
        if summary_lines:
            return summary_lines
        return [str(x) for x in case.get("content_sequence", []) if str(x).strip()]

    def _clip_summary_lines(self, summary_lines: Sequence[str]) -> List[str]:
        """Conservatively clip oversized summaries so zero-shot prompts fit model context."""
        context_size = int(self.model_cfg.get("context_size", 8192) or 8192)
        max_input_chars = int(
            self.model_cfg.get(
                "max_input_chars",
                max(4000, min(16000, context_size * 4)),
            )
        )
        max_line_chars = int(self.model_cfg.get("max_line_chars", 320))
        max_summary_lines = int(self.model_cfg.get("max_summary_lines_hard", 24))

        clipped: List[str] = []
        total_chars = 0
        for raw_line in summary_lines:
            line = re.sub(r"\s+", " ", str(raw_line)).strip()
            if not line:
                continue
            if len(line) > max_line_chars:
                line = line[: max_line_chars - 3].rstrip() + "..."
            projected = total_chars + len(line) + 3
            if clipped and projected > max_input_chars:
                break
            clipped.append(line)
            total_chars = projected
            if len(clipped) >= max_summary_lines:
                break

        return clipped or [str(x)[:max_line_chars] for x in summary_lines[:1] if str(x).strip()]

    def _build_result(
        self,
        *,
        mode: str,
        case: Dict[str, Any],
        question: str | None,
        active_fault_types: Sequence[str],
        summary: Dict[str, Any],
        summary_lines: Sequence[str],
        pred_fault: str,
        explanation: str,
        parse_valid: bool,
        raw_output: str,
    ) -> Dict[str, Any]:
        return {
            "mode": mode,
            "case_id": str(case.get("case_id", "")),
            "dataset_name": str(case.get("dataset_name", "")),
            "question": question or "",
            "fault_type_candidates": list(active_fault_types),
            "summary_lines": list(summary_lines),
            "summary_metadata": {
                "original_line_count": int(summary.get("original_line_count", len(case.get("content_sequence", [])))),
                "working_line_count": int(summary.get("working_line_count", len(summary_lines))),
                "summary_line_indices": list(summary.get("summary_line_indices", [])),
            },
            "pred_fault_type": pred_fault,
            "pred_explanation": explanation,
            "parse_valid": bool(parse_valid),
            "raw_output": raw_output,
        }

    def _diagnose_qwen(
        self,
        case: Dict[str, Any],
        *,
        question: str | None,
        active_fault_types: Sequence[str],
    ) -> Dict[str, Any]:
        summary = self.summarize_logs(case)
        summary_lines = self._clip_summary_lines(self._extract_summary_lines(case, summary))

        instruction = self._build_instruction(active_fault_types, question)
        input_text = "Log sequence: " + "\n".join(f"- {x}" for x in summary_lines)

        self._ensure_generation_model()
        assert self._model is not None

        from .infer import generate_response

        raw_output = generate_response(
            self._model,
            self._tokenizer,
            input_text,
            max_new_tokens=int(self.infer_cfg.get("max_new_tokens", 256)),
            temperature=float(self.infer_cfg.get("temperature", 0.0)),
            top_p=float(self.infer_cfg.get("top_p", 1.0)),
            system_prompt=instruction,
            seed=int(self.cfg.get("seed", 42)),
        )

        fault, explanation, parse_valid = parse_fault_and_explanation(raw_output)
        if active_fault_types:
            pred_fault = normalize_predicted_label(fault, list(active_fault_types))
        else:
            pred_fault = fault.strip() or "unknown type"

        return self._build_result(
            mode="qwen",
            case=case,
            question=question,
            active_fault_types=active_fault_types,
            summary=summary,
            summary_lines=summary_lines,
            pred_fault=pred_fault,
            explanation=explanation,
            parse_valid=parse_valid,
            raw_output=raw_output,
        )

    def diagnose(
        self,
        log_lines: Sequence[str],
        *,
        question: str | None = None,
        fault_type_list: Sequence[str] | None = None,
        case_id: str = "adhoc_case",
        dataset_name: str = "adhoc",
        mode: str = "qwen",
    ) -> Dict[str, Any]:
        case = self.build_case(log_lines, case_id=case_id, dataset_name=dataset_name)
        active_fault_types = self._active_fault_types(fault_type_list)
        self._normalize_mode(mode)
        return self._diagnose_qwen(case, question=question, active_fault_types=active_fault_types)
