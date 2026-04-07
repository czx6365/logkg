# LogInsight

`LogInsight` 是这个仓库里一套面向日志故障诊断的轻量复现实现。它把原始日志整理成统一 `case` 格式，再用 FOLS 风格的日志摘要压缩关键信息，随后构造指令微调数据、训练 LoRA 适配器，并完成推理、评估和消融实验。

这部分代码既支持离线实验流程，也支持单条日志序列的在线诊断入口。

## 目录结构

```text
code/loginsight/
├─ configs/
│  ├─ base.yaml          # 主流程配置：路径、FOLS、推理、消融等
│  ├─ dataset.yaml       # 数据适配与预处理配置
│  ├─ fols.yaml          # FOLS 摘要参数
│  └─ model.yaml         # LoRA 训练配置
├─ data/
│  ├─ instruction/       # 已生成的指令微调数据
│  └─ processed/         # 已生成的摘要/消融结果等中间产物
├─ scripts/
│  ├─ prepare_data.py
│  ├─ build_instruction_data.py
│  ├─ train_lora.py
│  ├─ run_inference.py
│  ├─ eval_all.py
│  ├─ run_ablation.py
│  └─ run_agent.py
└─ src/
   ├─ preprocess.py
   ├─ fols.py
   ├─ dataset_builder.py
   ├─ lora_train.py
   ├─ infer.py
   ├─ evaluate.py
   ├─ agent.py
   └─ ...
```

## 流程概览

完整流程如下：

1. `prepare_data.py`
把不同来源的数据集适配成统一 `processed_cases` JSONL。

2. `build_instruction_data.py`
对每个 case 运行 FOLS 摘要，并将摘要结果改写为指令微调样本。

3. `train_lora.py`
基于 `instruction_train.jsonl` / `instruction_val.jsonl` 训练 LoRA adapter。

4. `run_inference.py`
加载底模和可选 LoRA adapter，生成预测结果。

5. `eval_all.py`
计算 micro/macro/weighted F1、validity rate、每类指标和混淆矩阵。

6. `run_ablation.py`
比较 `full_loginsight`、`without_fols`、`kmeans_replace`、`agglomerative_replace` 等变体。

7. `run_agent.py`
对临时输入的一组日志做单次诊断，适合演示或手工排查。

## 核心模块

- `src/preprocess.py`
负责数据适配和日志清洗，输出统一 schema：
`case_id`、`fault_type`、`raw_logs`、`content_sequence`、`dataset_name` 等。

- `src/fols.py`
实现 FOLS 风格摘要。当前流程会先对日志行做相似度聚类，再结合 TF-IDF 选出最有代表性的摘要行。

- `src/dataset_builder.py`
把摘要后的 case 改写成监督微调用的 `instruction/input/output` 样本。

- `src/lora_train.py`
加载底模，自动猜测常见 LoRA target modules，并训练最终 adapter。

- `src/infer.py`
负责加载底模与 adapter、构造 prompt、生成输出、解析故障标签与解释。

- `src/evaluate.py`
负责汇总评估指标并导出 CSV/JSON。

- `src/agent.py`
对外提供 `LogInsightAgent`，可直接对一段日志文本做诊断。

## 运行前准备

下面的命令默认在 `code/loginsight` 目录下执行。

### 1. 安装依赖

仓库里目前没有单独的 `requirements.txt`，按代码实际 import，最小依赖可以这样安装：

```powershell
pip install torch transformers peft datasets scikit-learn pandas numpy pyyaml tqdm
```

如果你准备启用 `src/baselines.py` 里的 RAG baseline，再额外安装：

```powershell
pip install sentence-transformers
```

### 2. 准备数据

默认配置里：

- `configs/dataset.yaml` 会从 `../../../data/OS_preprocessed` 读取 `OS_preprocessed` 数据集
- `configs/base.yaml` 默认读取 `../data/processed/os_preprocessed_major.jsonl`

注意：

- `code/loginsight/data/processed/os_preprocessed_major.jsonl` 默认不会随仓库一起提供
- 你需要先运行 `prepare_data.py` 生成它，或者把 `configs/base.yaml` / `configs/dataset.yaml` 改成你自己的数据路径

## 数据适配

`src/preprocess.py` 当前支持这些 adapter：

- `os_preprocessed`
读取 `OS_preprocessed/cases/*.csv` 和 `case_labels.csv`

- `openstack`
读取 JSONL，每条记录至少包含 `case_id`、`fault_type`、`raw_logs`

- `hardware_public`
读取 CSV，至少包含 `case_id`、`fault_type`、`log_line`

- `custom_jsonl`
读取自定义 JSONL，每条记录至少包含 `case_id`、`fault_type`，以及 `raw_logs` 或 `logs`

- `sample_jsonl`
用于小规模 smoke test，但当前仓库未附带示例文件，需要自行准备

## 快速开始

### 1. 生成 `processed_cases`

```powershell
cd C:\Users\16973\Desktop\logKG\LogKG-main\code\loginsight
python scripts/prepare_data.py --config configs/dataset.yaml
```

输出默认写到：

```text
data/processed/os_preprocessed_major.jsonl
```

### 2. 构造 FOLS 摘要和指令数据

```powershell
python scripts/build_instruction_data.py --config configs/base.yaml
```

默认输出：

- `data/processed/os_preprocessed_major_fols.jsonl`
- `data/instruction/os_major_instruction_all.jsonl`
- `data/instruction/os_major_instruction_train.jsonl`
- `data/instruction/os_major_instruction_val.jsonl`

### 3. 训练 LoRA

```powershell
python scripts/train_lora.py --config configs/model.yaml
```

默认输出目录：

```text
checkpoints/loginsight_lora/final_adapter
```

### 4. 执行推理

```powershell
python scripts/run_inference.py --config configs/base.yaml
```

默认会读取：

- `data/processed/os_preprocessed_major.jsonl`
- `data/instruction/os_major_instruction_val.jsonl`

并输出到：

```text
data/processed/os_major_predictions.jsonl
```

### 5. 评估结果

```powershell
python scripts/eval_all.py --config configs/base.yaml
```

默认输出：

- `data/processed/os_major_eval_metrics.csv`
- `data/processed/os_major_eval_metrics_per_class.csv`
- `data/processed/os_major_eval_metrics_confusion_matrix.csv`
- `data/processed/os_major_eval_details.json`

### 6. 跑消融实验

```powershell
python scripts/run_ablation.py --config configs/base.yaml
```

默认输出：

```text
data/processed/os_major_ablation.csv
```

## 单条日志诊断

`run_agent.py` 适合对临时日志做诊断，不需要你手工拼 prompt。

### 方式一：从文件读取日志

```powershell
python scripts/run_agent.py `
  --config configs/base.yaml `
  --logs-file C:\path\to\logs.txt `
  --fault-type network `
  --fault-type storage `
  --fault-type cpu `
  --output C:\path\to\result.json
```

### 方式二：命令行直接传多条日志

```powershell
python scripts/run_agent.py `
  --config configs/base.yaml `
  --log-line "ERROR failed to connect database" `
  --log-line "retry count exceeded" `
  --fault-type database `
  --fault-type network
```

常用参数：

- `--model-name`
覆盖配置里的底模名称

- `--adapter-path`
覆盖配置里的 LoRA adapter 路径

- `--question`
追加一个额外诊断问题

- `--fault-type`
手工指定候选故障类别；不传时会尝试从 `processed_cases` 中自动推断

## 配置说明

### `configs/base.yaml`

主流程配置文件，主要包括：

- `paths`
中间文件、预测结果、评估结果、adapter 路径

- `fols`
摘要聚类与压缩参数

- `instruction`
指令模板、解释模式、验证集比例

- `inference`
生成长度、温度、是否只在 validation 集上推理

- `ablation`
消融实验配置

### `configs/dataset.yaml`

控制数据来源与清洗方式，主要包括：

- `adapter`
选择数据适配器

- `label_level`
`major` / `minor` / `pair`

- `parser_variant`
当前支持 `regex`，也保留了 `drain` / `divlog` / `lilac` 的扩展接口

- `cleanup_patterns`
用于移除时间戳、日志级别、host、pid 等噪声信息

### `configs/model.yaml`

控制 LoRA 训练，主要包括：

- `base_model_name`
- `max_token_length`
- `learning_rate`
- `batch_size`
- `num_train_epochs`
- `lora_r` / `lora_alpha` / `lora_dropout`

## 结果文件格式

`run_inference.py` 生成的预测记录至少包含：

- `case_id`
- `dataset_name`
- `fault_type`
- `pred_fault_type`
- `pred_explanation`
- `parse_valid`
- `raw_output`

其中 `parse_valid` 表示模型输出是否成功被解析成预期格式，不等价于分类是否正确。

## 开发提示

- 大型中间数据不要直接提交到 Git，尤其是 `data/processed/*.jsonl`
- 如果你切换了数据源，优先检查 `configs/dataset.yaml` 和 `configs/base.yaml` 的路径是否一致
- 如果推理时报缺少生成依赖，需要安装 `torch`、`transformers`、`peft`
- 如果只想验证流程是否能跑通，建议先用较小数据子集，或者在 `dataset.max_cases` 里限制 case 数量
