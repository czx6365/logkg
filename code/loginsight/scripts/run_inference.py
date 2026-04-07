from __future__ import annotations

import argparse
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.infer import infer_cases, load_generation_model
from src.utils import load_jsonl, load_yaml, resolve_path, save_jsonl, set_seed



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 42)))

    paths = cfg["paths"]
    fols_cfg = cfg["fols"]
    infer_cfg = cfg["inference"]

    processed_path = resolve_path(paths["processed_cases"], args.config.parent)
    instruction_val_path = resolve_path(paths["instruction_val"], args.config.parent)
    pred_path = resolve_path(paths["predictions_path"], args.config.parent)
    adapter_path = resolve_path(paths.get("adapter_path", ""), args.config.parent)

    cases = load_jsonl(processed_path)
    fault_types = sorted({str(x["fault_type"]) for x in cases})

    if bool(infer_cfg.get("use_instruction_val_only", True)) and instruction_val_path.exists():
        # 默认只在验证集 case 上推理，避免把训练样本也一起评估进去。
        val_case_ids = {str(x["case_id"]) for x in load_jsonl(instruction_val_path)}
        infer_cases_list = [x for x in cases if str(x.get("case_id")) in val_case_ids]
    else:
        infer_cases_list = cases

    model_name = str(cfg.get("model", {}).get("base_model_name", "mistralai/Mistral-7B-Instruct-v0.2"))
    adapter = str(adapter_path) if str(adapter_path) else None
    model, tokenizer = load_generation_model(model_name, adapter_path=adapter)

    # 推理阶段会再次按当前配置生成摘要，而不是直接读取训练时缓存的输入文本。
    preds = infer_cases(
        cases=infer_cases_list,
        all_cases_for_fols=cases,
        fault_type_list=fault_types,
        model=model,
        tokenizer=tokenizer,
        instruction_template=cfg["instruction"]["instruction_template"],
        fols_cfg=fols_cfg,
        max_new_tokens=int(infer_cfg.get("max_new_tokens", 256)),
        temperature=float(infer_cfg.get("temperature", 0.0)),
        top_p=float(infer_cfg.get("top_p", 1.0)),
        variant="full_loginsight",
    )

    save_jsonl(preds, pred_path)
    print(f"inference_cases={len(infer_cases_list)}")
    print(f"predictions_saved={pred_path}")


if __name__ == "__main__":
    main()
