from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from .infer import infer_cases
from .prompting import build_prompt



def run_prompt_baseline(
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
) -> List[Dict[str, Any]]:
    """Prompt-only baseline: same prompt, no LoRA adapter."""
    return infer_cases(
        cases=cases,
        all_cases_for_fols=all_cases_for_fols,
        fault_type_list=fault_type_list,
        model=model,
        tokenizer=tokenizer,
        instruction_template=instruction_template,
        fols_cfg=fols_cfg,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        variant="full_loginsight",
    )



def _cosine_sim_matrix(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = (np.linalg.norm(A, axis=1) * np.linalg.norm(b) + 1e-12)
    return (A @ b) / denom



def build_rag_prompt(
    instruction_template: str,
    fault_type_list: Sequence[str],
    query_summary_lines: Sequence[str],
    retrieved_examples: Sequence[Dict[str, Any]],
) -> str:
    """Build RAG prompt by prepending retrieved historical cases."""
    instruction = instruction_template.format(fault_type_list=sorted(set(fault_type_list)))
    query_text = "Log sequence: " + "\n".join(f"- {x}" for x in query_summary_lines)

    example_chunks = []
    for ex in retrieved_examples:
        example_chunks.append(
            "Retrieved Example:\n"
            f"Log sequence: {ex['summary_text']}\n"
            f"Fault Type: {ex['fault_type']}\n"
            f"Explanation: {ex.get('explanation', '[Approximate] historical label context')}"
        )

    full_input = "\n\n".join(example_chunks + [query_text])
    return build_prompt(instruction, full_input)



def build_rag_memory(
    cases_with_summary: Sequence[Dict[str, Any]],
    embedding_model_name: str,
) -> Dict[str, Any]:
    """Create retrieval index from historical cases."""
    encoder = SentenceTransformer(embedding_model_name)
    texts = [" ".join(x.get("summary_lines", [])) for x in cases_with_summary]
    emb = encoder.encode(texts, normalize_embeddings=True)
    return {"encoder": encoder, "embeddings": emb, "cases": list(cases_with_summary)}



def retrieve_top_k(memory: Dict[str, Any], query_text: str, top_k: int) -> List[Dict[str, Any]]:
    """Retrieve top-k similar historical cases by cosine similarity."""
    encoder = memory["encoder"]
    vec = encoder.encode([query_text], normalize_embeddings=True)[0]
    sims = _cosine_sim_matrix(memory["embeddings"], vec)
    idx = np.argsort(-sims)[:top_k]
    return [memory["cases"][int(i)] for i in idx]
