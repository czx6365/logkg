# Qwen Zero-Shot Classification Flow

This module follows the LogInsight paper's zero-shot inference structure:

```text
raw logs
  -> preprocessing
  -> Fault-Oriented Log Summary (cluster aggregation + TF-IDF ranking)
  -> Qwen diagnosis prompt
  -> Fault Type + Explanation
  -> eval_all.py metrics
```

The default deployment path on this machine uses the local GGUF Qwen model through `llama-cli`.

## Pipeline

## Main Commands

Recommended environment setup:

```powershell
python3 -m venv .venv
.venv/bin/pip install -r loginsight_zeroshot/requirements.txt
```

Prepare data:

```powershell
python loginsight_zeroshot/scripts/prepare_data.py --config loginsight_zeroshot/configs/os_b507_minor_zero_shot_5fold.yaml
```

Run Qwen zero-shot inference:

```powershell
python3 loginsight_zeroshot/scripts/run_inference.py --config loginsight_zeroshot/configs/os_b507_minor_zero_shot_5fold.yaml --mode qwen
```

Evaluate predictions:

```powershell
python3 loginsight_zeroshot/scripts/eval_all.py --config loginsight_zeroshot/configs/os_b507_minor_zero_shot_5fold.yaml
```

Run one ad-hoc diagnosis:

```powershell
python3 loginsight_zeroshot/scripts/run_agent.py --config loginsight_zeroshot/configs/os_b507_minor_zero_shot_5fold.yaml --mode qwen --log-line "ERROR failed to execute test file"
```

## Current Model

Default model on this machine:

```yaml
model:
  backend: llama_cpp_cli
  base_model_name: /Users/czx/Models/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf
  llama_cli_path: /opt/homebrew/bin/llama-cli
```

You can override it at runtime with another local GGUF path or a HuggingFace model id:

```powershell
python3 loginsight_zeroshot/scripts/run_inference.py --config loginsight_zeroshot/configs/os_b507_minor_zero_shot_5fold.yaml --mode qwen --model-name /path/to/another-qwen.gguf
```

## Notes

- Cross-validation is disabled for zero-shot inference.
- The loader auto-selects `llama_cpp_cli` for local `.gguf` models and falls back to `transformers` for HuggingFace model ids.
- Existing files under `result/` may still contain historical outputs from older experiments.
- The prediction parser expects Qwen to return:

```text
Fault Type: <one label>
Explanation: <brief evidence-based explanation>
```
