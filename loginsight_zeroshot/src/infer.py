from __future__ import annotations

from pathlib import Path
import json
import shutil
import subprocess
import time
from typing import Any, Dict, List, Sequence, Tuple
import urllib.error
import urllib.request

from tqdm import tqdm

from .fols import build_token_document_frequency, summarize_case
from .prompting import (
    build_prompt,
    build_inference_instruction,
    normalize_predicted_label,
    parse_fault_and_explanation,
)


def _looks_like_local_gguf(model_ref: str) -> bool:
    text = str(model_ref).strip()
    if not text:
        return False
    path = Path(text).expanduser()
    return text.lower().endswith(".gguf") or path.exists()


def _select_generation_backend(base_model_name: str, model_cfg: Dict[str, Any]) -> str:
    configured = str(model_cfg.get("backend", "")).strip().lower()
    if configured:
        if configured not in {"transformers", "llama_cpp_cli", "llama_server"}:
            raise ValueError(f"Unsupported generation backend: {configured}")
        return configured
    if _looks_like_local_gguf(base_model_name):
        return "llama_cpp_cli"
    return "transformers"


def _load_transformers_generation_model(
    base_model_name: str,
    adapter_path: str | None = None,
) -> Tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from peft import PeftModel
    except ModuleNotFoundError:
        PeftModel = None

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if adapter_path:
        p = Path(adapter_path)
        if p.exists() and (p / "adapter_config.json").exists():
            # 推理时把训练得到的 adapter 叠加到底模上。
            if PeftModel is None:
                raise RuntimeError("peft is required when adapter_path points to a LoRA adapter.")
            model = PeftModel.from_pretrained(model, str(p))

    model.eval()
    return model, tokenizer


def _load_llama_cpp_cli_model(
    base_model_name: str,
    adapter_path: str | None,
    model_cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], None]:
    model_path = Path(str(base_model_name)).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"Local GGUF model not found: {model_path}")

    cli_hint = str(model_cfg.get("llama_cli_path", "llama-cli")).strip() or "llama-cli"
    cli_path = shutil.which(cli_hint) if cli_hint != "llama-cli" else shutil.which("llama-cli")
    if cli_path is None and Path(cli_hint).expanduser().exists():
        cli_path = str(Path(cli_hint).expanduser())
    if cli_path is None:
        raise RuntimeError(
            "llama-cli was not found. Install llama.cpp or set model.llama_cli_path to the local executable."
        )

    backend_state: Dict[str, Any] = {
        "backend": "llama_cpp_cli",
        "cli_path": cli_path,
        "model_path": str(model_path),
        "adapter_path": adapter_path or "",
        "context_size": int(model_cfg.get("context_size", 8192)),
        "gpu_layers": str(model_cfg.get("gpu_layers", "auto")),
        "threads": int(model_cfg["threads"]) if model_cfg.get("threads") is not None else None,
        "reasoning": str(model_cfg.get("reasoning", "off")).strip().lower() or "off",
        "chat_template": str(model_cfg.get("chat_template", "")).strip(),
        "timeout_seconds": int(model_cfg.get("timeout_seconds", 600)),
        "extra_args": [str(x) for x in model_cfg.get("extra_args", [])],
    }
    return backend_state, None


def _load_llama_server_model(
    base_model_name: str,
    adapter_path: str | None,
    model_cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], None]:
    server_url = str(model_cfg.get("server_url", "")).strip().rstrip("/")
    if not server_url:
        host = str(model_cfg.get("host", "127.0.0.1")).strip() or "127.0.0.1"
        port = int(model_cfg.get("port", 8080))
        server_url = f"http://{host}:{port}"

    return (
        {
            "backend": "llama_server",
            "server_url": server_url,
            "model_name": str(model_cfg.get("server_model_name", model_cfg.get("alias", "qwen-local"))),
            "timeout_seconds": int(model_cfg.get("timeout_seconds", 600)),
            "api_key": str(model_cfg.get("api_key", "")).strip(),
            "reasoning": str(model_cfg.get("reasoning", "off")).strip().lower() or "off",
            "adapter_path": adapter_path or "",
            "base_model_name": base_model_name,
            "llama_cli_path": str(model_cfg.get("llama_cli_path", "llama-cli")).strip() or "llama-cli",
            "context_size": int(model_cfg.get("context_size", 8192)),
            "gpu_layers": str(model_cfg.get("gpu_layers", "auto")),
            "threads": int(model_cfg["threads"]) if model_cfg.get("threads") is not None else None,
            "max_retries": int(model_cfg.get("max_retries", 5)),
            "retry_backoff_seconds": float(model_cfg.get("retry_backoff_seconds", 2.0)),
        },
        None,
    )


def load_generation_model(
    base_model_name: str,
    adapter_path: str | None = None,
    model_cfg: Dict[str, Any] | None = None,
) -> Tuple[Any, Any]:
    """Load a generation backend for either transformers or local llama.cpp GGUF inference."""
    model_cfg = dict(model_cfg or {})
    backend = _select_generation_backend(base_model_name, model_cfg)
    if backend == "llama_server":
        return _load_llama_server_model(base_model_name, adapter_path, model_cfg)
    if backend == "llama_cpp_cli":
        return _load_llama_cpp_cli_model(base_model_name, adapter_path, model_cfg)
    return _load_transformers_generation_model(base_model_name, adapter_path=adapter_path)



def generate_response(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    *,
    system_prompt: str | None = None,
    seed: int | None = None,
) -> str:
    """Generate one response string from either a transformers or llama.cpp backend."""
    if isinstance(model, dict) and model.get("backend") == "llama_server":
        return _generate_llama_server_response(
            model,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            system_prompt=system_prompt,
            seed=seed,
        )
    if isinstance(model, dict) and model.get("backend") == "llama_cpp_cli":
        return _generate_llama_cpp_cli_response(
            model,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            system_prompt=system_prompt,
            seed=seed,
        )

    return _generate_transformers_response(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        system_prompt=system_prompt,
    )


def _generate_transformers_response(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    system_prompt: str | None,
) -> str:
    import torch

    if getattr(tokenizer, "chat_template", None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    else:
        merged_prompt = build_prompt(system_prompt or "", prompt)
        inputs = tokenizer(merged_prompt, return_tensors="pt")

    input_length = int(inputs["input_ids"].shape[-1])
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

    generated = out[0][input_length:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _generate_llama_cpp_cli_response(
    model_state: Dict[str, Any],
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    system_prompt: str | None,
    seed: int | None,
) -> str:
    command = [
        str(model_state["cli_path"]),
        "-m",
        str(model_state["model_path"]),
        "--simple-io",
        "--log-disable",
        "--no-display-prompt",
        "--no-show-timings",
        "--conversation",
        "--single-turn",
        "--reasoning",
        str(model_state.get("reasoning", "off")),
        "--temp",
        str(float(temperature)),
        "--top-p",
        str(float(top_p)),
        "--seed",
        str(int(seed if seed is not None else 42)),
        "--ctx-size",
        str(int(model_state.get("context_size", 8192))),
        "--gpu-layers",
        str(model_state.get("gpu_layers", "auto")),
        "--predict",
        str(int(max_new_tokens)),
    ]

    threads = model_state.get("threads")
    if threads is not None:
        command.extend(["--threads", str(int(threads))])

    system_text = (system_prompt or "").strip()
    if system_text:
        command.extend(["--system-prompt", system_text])

    chat_template = str(model_state.get("chat_template", "")).strip()
    if chat_template:
        command.extend(["--chat-template", chat_template])

    adapter_path = str(model_state.get("adapter_path", "")).strip()
    if adapter_path:
        command.extend(["--lora", adapter_path])

    extra_args = list(model_state.get("extra_args", []))
    if extra_args:
        command.extend(extra_args)

    command.extend(["--prompt", prompt])

    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=int(model_state.get("timeout_seconds", 600)),
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        message = stderr or stdout or f"llama-cli exited with code {result.returncode}"
        raise RuntimeError(f"llama.cpp generation failed: {message}")
    return _extract_llama_cpp_cli_response(result.stdout or "", prompt)


def _extract_llama_cpp_cli_response(raw_output: str, prompt: str) -> str:
    text = str(raw_output).replace("\r\n", "\n").replace("\r", "\n")
    marker = f"> {prompt}"
    marker_pos = text.rfind(marker)
    if marker_pos != -1:
        text = text[marker_pos + len(marker):]
    if "\nExiting..." in text:
        text = text.split("\nExiting...", 1)[0]

    cleaned = text.strip()
    if not cleaned:
        raise RuntimeError("llama.cpp returned an empty response.")
    return cleaned


def _generate_llama_server_response(
    model_state: Dict[str, Any],
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    system_prompt: str | None,
    seed: int | None,
) -> str:
    active_prompt = str(prompt)
    for _ in range(4):
        try:
            body = _post_llama_server_chat_completion(
                model_state,
                prompt=active_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                system_prompt=system_prompt,
                seed=seed,
            )
        except RuntimeError as exc:
            if _is_retryable_server_error(str(exc)):
                return _fallback_to_llama_cpp_cli(
                    model_state,
                    prompt=active_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    system_prompt=system_prompt,
                    seed=seed,
                )
            raise
        try:
            parsed = json.loads(body)
            content = parsed["choices"][0]["message"]["content"]
        except Exception as exc:
            error_text = body[:1000]
            if _is_context_overflow_error(error_text):
                shorter = _shrink_prompt_for_retry(active_prompt)
                if shorter != active_prompt:
                    active_prompt = shorter
                    continue
            raise RuntimeError(f"Unexpected llama-server response: {error_text}") from exc

        cleaned = str(content).strip()
        if cleaned:
            return cleaned

        shorter = _shrink_prompt_for_retry(active_prompt)
        if shorter == active_prompt:
            break
        active_prompt = shorter

    raise RuntimeError("llama-server returned an empty response after retries.")


def _post_llama_server_chat_completion(
    model_state: Dict[str, Any],
    *,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    system_prompt: str | None,
    seed: int | None,
) -> str:
    messages: List[Dict[str, str]] = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": str(model_state.get("model_name", "qwen-local")),
        "messages": messages,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_new_tokens),
        "seed": int(seed if seed is not None else 42),
    }

    reasoning = str(model_state.get("reasoning", "off")).strip().lower()
    if reasoning in {"off", "false", "0"}:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=f"{str(model_state['server_url']).rstrip('/')}/v1/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            **(
                {"Authorization": f"Bearer {model_state['api_key']}"}
                if str(model_state.get("api_key", "")).strip()
                else {}
            ),
        },
        method="POST",
    )
    max_retries = max(0, int(model_state.get("max_retries", 5)))
    backoff = max(0.1, float(model_state.get("retry_backoff_seconds", 2.0)))
    retryable_codes = {408, 429, 500, 502, 503, 504}

    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=int(model_state.get("timeout_seconds", 600))) as resp:
                return resp.read().decode("utf-8", "ignore")
        except urllib.error.HTTPError as exc:
            error_body = ""
            try:
                error_body = exc.read().decode("utf-8", "ignore")
            except Exception:
                error_body = ""
            detail = error_body or str(exc)
            if _is_context_overflow_error(detail):
                return detail
            if exc.code in retryable_codes and attempt < max_retries:
                time.sleep(backoff * (2 ** attempt))
                continue
            raise RuntimeError(f"llama-server generation failed: HTTP {exc.code}: {detail[:1000]}") from exc
        except Exception as exc:
            if attempt < max_retries:
                time.sleep(backoff * (2 ** attempt))
                continue
            raise RuntimeError(f"llama-server generation failed: {exc}") from exc


def _is_context_overflow_error(text: str) -> bool:
    lowered = str(text).lower()
    return "exceeds the available context size" in lowered or "context size" in lowered


def _is_retryable_server_error(text: str) -> bool:
    lowered = str(text).lower()
    return any(code in lowered for code in ["http 408", "http 429", "http 500", "http 502", "http 503", "http 504"])


def _fallback_to_llama_cpp_cli(
    model_state: Dict[str, Any],
    *,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    system_prompt: str | None,
    seed: int | None,
) -> str:
    cli_hint = str(model_state.get("llama_cli_path", "llama-cli")).strip() or "llama-cli"
    cli_path = shutil.which(cli_hint) if cli_hint != "llama-cli" else shutil.which("llama-cli")
    if cli_path is None and Path(cli_hint).expanduser().exists():
        cli_path = str(Path(cli_hint).expanduser())
    if cli_path is None:
        raise RuntimeError("llama-server failed and llama-cli fallback is unavailable.")

    return _generate_llama_cpp_cli_response(
        {
            "backend": "llama_cpp_cli",
            "cli_path": cli_path,
            "model_path": str(Path(str(model_state["base_model_name"])).expanduser()),
            "adapter_path": str(model_state.get("adapter_path", "")).strip(),
            "context_size": int(model_state.get("context_size", 8192)),
            "gpu_layers": str(model_state.get("gpu_layers", "auto")),
            "threads": model_state.get("threads"),
            "reasoning": str(model_state.get("reasoning", "off")),
            "chat_template": "",
            "timeout_seconds": int(model_state.get("timeout_seconds", 600)),
            "extra_args": [],
        },
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        system_prompt=system_prompt,
        seed=seed,
    )


def _shrink_prompt_for_retry(prompt: str) -> str:
    lines = str(prompt).splitlines()
    if len(lines) <= 2:
        return prompt[: max(512, len(prompt) // 2)]

    header = lines[0]
    bullet_lines = [ln for ln in lines[1:] if ln.strip()]
    if len(bullet_lines) <= 1:
        return prompt[: max(512, len(prompt) // 2)]

    keep = max(1, len(bullet_lines) // 2)
    shortened = "\n".join([header] + bullet_lines[:keep])
    if shortened == prompt:
        return prompt[: max(512, len(prompt) // 2)]
    return shortened


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
        instruction = build_inference_instruction(instruction_template, known)
        raw_output = generate_response(
            model,
            tokenizer,
            input_text,
            max_new_tokens,
            temperature,
            top_p,
            system_prompt=instruction,
        )

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
