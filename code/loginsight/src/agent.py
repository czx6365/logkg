from __future__ import annotations
# 作用：
# 1. 让类型注解支持“延迟解析”，避免前向引用时出问题
# 2. 例如返回值写成 "LogInsightAgent" 时，不需要类先定义完

from pathlib import Path
# Path 用于更稳健地处理文件路径，替代字符串拼路径

from typing import Any, Dict, List, Sequence
# 类型注解相关：
# Any      -> 任意类型
# Dict     -> 字典
# List     -> 列表
# Sequence -> 序列类型（如 list / tuple 等）

from .fols import build_token_document_frequency, summarize_case
# 从当前包的 fols 模块中导入：
# build_token_document_frequency -> 构建文档频率（document frequency）
# summarize_case                -> 对一个日志 case 做摘要/压缩，提取更关键的日志行

from .prompting import build_prompt, normalize_predicted_label, parse_fault_and_explanation
# 从 prompting 模块中导入：
# build_prompt                 -> 构造给大模型的 prompt
# normalize_predicted_label    -> 将模型输出的标签规范化到候选标签中
# parse_fault_and_explanation  -> 从模型原始输出中解析“故障类型 + 解释”

from .utils import load_jsonl, load_yaml, resolve_path
# 工具函数：
# load_jsonl   -> 读取 jsonl 文件
# load_yaml    -> 读取 yaml 配置文件
# resolve_path -> 将相对路径解析成绝对路径


def _format_fault_types(fault_type_list: Sequence[str]) -> str:
    """
    把候选故障类型列表格式化成一个字符串，供 prompt 使用。

    参数：
        fault_type_list: 故障类型列表，例如 ["network", "disk", "cpu"]

    返回：
        "network, disk, cpu"
        如果列表为空，则返回 "unknown type"
    """
    # 去掉空字符串、首尾空格，只保留有效标签
    labels = [str(x).strip() for x in fault_type_list if str(x).strip()]

    # 如果有有效标签，就用逗号拼接；否则返回 unknown type
    return ", ".join(labels) if labels else "unknown type"


class LogInsightAgent:
    """
    一个最小化的 LogInsight 智能体封装类。

    核心功能包括两部分：
    1. 用 FOLS/FOLR 风格的方法对日志 case 做摘要（summarization）
    2. 调用生成式大模型对摘要后的日志进行故障诊断（diagnosis）

    这个类本质上是一个“上层调度器”：
    - 下层负责日志摘要
    - 下层负责模型加载与生成
    - 这个类把它们串起来，形成可直接调用的诊断流程
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        config_path: Path,
        reference_cases: Sequence[Dict[str, Any]] | None = None,
        fault_type_list: Sequence[str] | None = None,
    ) -> None:
        """
        初始化 LogInsightAgent。

        参数：
            cfg:
                完整配置字典，通常来自 yaml 配置文件
            config_path:
                配置文件路径，用于后续解析相对路径
            reference_cases:
                历史参考案例（训练/知识库里的 case）
            fault_type_list:
                手工指定的故障类型候选列表
        """

        # 保存原始配置
        self.cfg = cfg

        # 将配置路径统一转成 Path 对象
        self.config_path = Path(config_path)

        # 历史参考案例，若为空则用空列表
        self.reference_cases = list(reference_cases or [])

        # 各子模块配置
        self.fols_cfg = dict(cfg.get("fols", {}))            # 日志摘要相关配置
        self.infer_cfg = dict(cfg.get("inference", {}))      # 推理生成相关配置
        self.instruction_cfg = dict(cfg.get("instruction", {}))  # 指令模板相关配置
        self.model_cfg = dict(cfg.get("model", {}))          # 模型相关配置

        # 先尝试使用传入的 fault_type_list
        # 过滤空字符串，并统一转成字符串
        configured_fault_types = [str(x) for x in (fault_type_list or []) if str(x).strip()]

        # 如果外部没有传候选故障类型，就从历史 reference_cases 中自动提取
        if not configured_fault_types:
            configured_fault_types = sorted(
                {
                    str(x.get("fault_type", "")).strip()
                    for x in self.reference_cases
                    if str(x.get("fault_type", "")).strip()
                }
            )

        # 保存默认故障标签列表
        self.default_fault_type_list = configured_fault_types

        # 如果有历史案例，就基于历史案例构建 token 的文档频率
        # 这一步通常用于后续日志摘要中的重要性评估
        base_cases = self.reference_cases or []
        self._doc_freq = build_token_document_frequency(base_cases) if base_cases else {}

        # 历史案例总数
        self._total_cases = len(base_cases)

        # 模型与 tokenizer 先不加载，延迟到真正推理时再加载
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
        """
        从配置文件直接构造一个 LogInsightAgent 对象。

        这是一个工厂方法（classmethod），方便外部直接通过 yaml 配置初始化。

        参数：
            config_path:
                yaml 配置文件路径
            fault_type_list:
                可选，手工传入故障类型列表
            model_name:
                可选，覆盖配置中的 base model 名称
            adapter_path:
                可选，覆盖配置中的 LoRA / PEFT adapter 路径

        返回：
            一个初始化好的 LogInsightAgent 实例
        """

        # 统一转为 Path
        config_path = Path(config_path)

        # 读取 yaml 配置
        cfg = load_yaml(config_path)

        # 读取 paths 配置块
        paths = dict(cfg.get("paths", {}))

        # 取出 processed_cases 路径
        processed_path_value = str(paths.get("processed_cases", "")).strip()

        # 用于存放加载到的参考案例
        reference_cases: List[Dict[str, Any]] = []

        # 如果配置里写了 processed_cases 路径
        if processed_path_value:
            # 将路径解析为相对于配置文件所在目录的真实路径
            processed_path = resolve_path(processed_path_value, config_path.parent)

            # 如果文件存在，就读取 jsonl
            if processed_path.exists():
                reference_cases = load_jsonl(processed_path)

        # 如果外部显式传了 model_name，则覆盖配置文件中的模型名
        if model_name:
            cfg.setdefault("model", {})
            cfg["model"]["base_model_name"] = model_name

        # 如果外部显式传了 adapter_path，则覆盖配置文件中的 adapter 路径
        if adapter_path:
            cfg.setdefault("paths", {})
            cfg["paths"]["adapter_path"] = adapter_path

        # 返回实例
        return cls(
            cfg=cfg,
            config_path=config_path,
            reference_cases=reference_cases,
            fault_type_list=fault_type_list,
        )

    def _ensure_generation_model(self) -> None:
        """
        确保生成模型已经加载。

        采用“延迟加载”策略：
        - 如果模型已经加载过，就直接返回
        - 如果还没加载，则在这里加载模型和 tokenizer

        这样做的好处：
        1. 初始化对象时更轻量
        2. 只有真正做 diagnose 时才占用显存/内存
        """

        # 如果模型和 tokenizer 都已经加载好了，直接返回
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            # 动态导入，避免没有依赖时一开始就报错
            from .infer import load_generation_model
        except ModuleNotFoundError as exc:
            # 如果缺少 torch / transformers / peft 等依赖，给出更明确的报错
            raise RuntimeError(
                "Generation dependencies are missing. Install torch, transformers, and peft before running the agent."
            ) from exc

        # 读取基础模型名；如果配置里没有，就给一个默认值
        model_name = str(self.model_cfg.get("base_model_name", "mistralai/Mistral-7B-Instruct-v0.2"))

        # 读取 adapter 路径
        adapter_path_value = str(self.cfg.get("paths", {}).get("adapter_path", "")).strip()
        adapter_path = None

        # 如果 adapter 路径存在，则解析成真实路径
        if adapter_path_value:
            adapter_path = str(resolve_path(adapter_path_value, self.config_path.parent))

        # 真正加载模型和 tokenizer
        self._model, self._tokenizer = load_generation_model(model_name, adapter_path=adapter_path)

    def build_case(
        self,
        log_lines: Sequence[str],
        *,
        case_id: str = "adhoc_case",
        dataset_name: str = "adhoc",
    ) -> Dict[str, Any]:
        """
        把原始日志行列表包装成统一的 case 数据结构。

        参数：
            log_lines:
                原始日志行序列
            case_id:
                case 唯一标识
            dataset_name:
                数据集名称，默认 adhoc，表示临时输入

        返回：
            一个标准 case 字典，例如：
            {
                "case_id": "adhoc_case",
                "dataset_name": "adhoc",
                "fault_type": "",
                "content_sequence": [...]
            }
        """

        return {
            "case_id": case_id,
            "dataset_name": dataset_name,
            "fault_type": "",  # 在线推理时通常未知，先留空
            "content_sequence": [str(x) for x in log_lines if str(x).strip()],  # 过滤空行
        }

    def summarize_logs(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """
        对输入 case 的日志进行摘要。

        逻辑分两种情况：
        1. 如果有历史 reference_cases
           -> 使用历史案例构建好的文档频率 doc_freq
        2. 如果没有历史案例
           -> 用当前 case 自举（bootstrap）构造一个最小 doc_freq

        返回：
            summarize_case 的结果，通常包含：
            - fault_summary
            - original_line_count
            - working_line_count
            - summary_line_indices
            等信息
        """

        # 有参考案例：用历史统计信息
        if self.reference_cases:
            doc_freq = self._doc_freq
            total_cases = self._total_cases
        else:
            # 没有参考案例：只拿当前 case 自己构建一个最小统计
            bootstrap_cases = [case]
            doc_freq = build_token_document_frequency(bootstrap_cases)
            total_cases = len(bootstrap_cases)

        # 调用 summarize_case 做真正的摘要
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
        """
        构造给大模型的 instruction 部分。

        这个 instruction 的目标是：
        1. 告诉模型任务是什么
        2. 限定候选标签范围
        3. 强制输出格式统一，方便后处理解析

        参数：
            fault_type_list:
                候选故障类型列表
            question:
                用户额外问题，例如：
                “请判断根因并解释原因”

        返回：
            最终 instruction 字符串
        """

        # 读取 instruction 模板，如果没配则使用默认模板
        template = str(
            self.instruction_cfg.get(
                "instruction_template",
                "Determine the most likely fault type from the log sequence.",
            )
        )

        # 用候选标签列表替换模板中的占位符
        base_instruction = template.format(fault_type_list=_format_fault_types(fault_type_list))

        # 增加严格输出格式要求，确保后续 parse_fault_and_explanation 容易解析
        strict_schema = (
            "You are LogInsightAgent.\n"
            "Choose exactly one label from the candidate fault type list. "
            "If none fits, answer with unknown type.\n"
            "Respond in exactly this format:\n"
            "Fault Type: <one label>\n"
            "Explanation: <brief evidence-based explanation>"
        )

        # 如果用户还给了额外问题，也拼接进去
        if question:
            strict_schema += f"\nUser request: {question.strip()}"

        # 最终 instruction = 基础模板 + 严格格式约束
        return f"{base_instruction}\n\n{strict_schema}"

    def diagnose(
        self,
        log_lines: Sequence[str],
        *,
        question: str | None = None,
        fault_type_list: Sequence[str] | None = None,
        case_id: str = "adhoc_case",
        dataset_name: str = "adhoc",
    ) -> Dict[str, Any]:
        """
        对一组日志进行完整诊断。

        完整流程：
        1. 把日志包装成 case
        2. 对 case 做摘要，提取关键日志
        3. 构造候选标签 + 指令
        4. 构造 prompt
        5. 加载生成模型并推理
        6. 解析输出，得到“预测故障类型 + 解释”
        7. 返回统一结构化结果

        参数：
            log_lines:
                输入日志行
            question:
                用户附加问题
            fault_type_list:
                本次诊断指定的候选故障类型；如果不传，则用默认标签列表
            case_id:
                case 编号
            dataset_name:
                数据集名

        返回：
            一个字典，包含诊断结果和中间信息
        """

        # 第 1 步：构造 case
        case = self.build_case(log_lines, case_id=case_id, dataset_name=dataset_name)

        # 第 2 步：对日志做摘要
        summary = self.summarize_logs(case)

        # 取出摘要后的关键日志行
        summary_lines = [str(x) for x in summary.get("fault_summary", [])]

        # 如果摘要结果为空，则退化为使用原始日志
        if not summary_lines:
            summary_lines = [str(x) for x in case.get("content_sequence", [])]

        # 第 3 步：确定当前有效的候选故障类型列表
        # 优先使用本次传入的 fault_type_list，否则使用默认列表
        active_fault_types = [str(x) for x in (fault_type_list or self.default_fault_type_list) if str(x).strip()]

        # 构造 instruction
        instruction = self._build_instruction(active_fault_types, question)

        # 第 4 步：构造模型输入文本
        # 把摘要后的日志逐行加上 "- " 前缀，形成更清晰的提示结构
        input_text = "Log sequence: " + "\n".join(f"- {x}" for x in summary_lines)

        # 利用 prompting 模块统一拼装 prompt
        prompt = build_prompt(instruction, input_text)

        # 第 5 步：确保模型已加载
        self._ensure_generation_model()
        assert self._model is not None and self._tokenizer is not None

        # 动态导入生成函数
        from .infer import generate_response

        # 第 6 步：调用模型生成原始输出
        raw_output = generate_response(
            self._model,
            self._tokenizer,
            prompt,
            max_new_tokens=int(self.infer_cfg.get("max_new_tokens", 256)),  # 生成最大长度
            temperature=float(self.infer_cfg.get("temperature", 0.0)),      # 温度，默认 0 更稳定
            top_p=float(self.infer_cfg.get("top_p", 1.0)),                  # nucleus sampling 参数
        )

        # 第 7 步：解析模型输出
        # parse_fault_and_explanation 会尝试从 raw_output 中提取：
        # fault         -> 故障类型
        # explanation   -> 解释
        # parse_valid   -> 解析是否成功
        fault, explanation, parse_valid = parse_fault_and_explanation(raw_output)

        # 如果有候选标签列表，则把模型输出规范化到候选集合中
        # 例如模型输出 "network issue"，规范化成 "network"
        if active_fault_types:
            pred_fault = normalize_predicted_label(fault, active_fault_types)
        else:
            # 如果没有候选标签，就直接保留模型输出
            pred_fault = fault.strip() or "unknown type"

        # 返回统一的结构化诊断结果
        return {
            "case_id": case_id,
            "dataset_name": dataset_name,
            "question": question or "",
            "fault_type_candidates": active_fault_types,
            "summary_lines": summary_lines,
            "summary_metadata": {
                # 原始日志行数
                "original_line_count": int(summary.get("original_line_count", len(case.get("content_sequence", [])))),

                # 参与摘要/工作阶段的日志行数
                "working_line_count": int(summary.get("working_line_count", len(summary_lines))),

                # 被选进摘要的原始行索引
                "summary_line_indices": list(summary.get("summary_line_indices", [])),
            },
            "pred_fault_type": pred_fault,      # 预测故障类型
            "pred_explanation": explanation,    # 预测解释
            "parse_valid": bool(parse_valid),   # 输出是否成功解析
            "raw_output": raw_output,           # 模型原始输出，方便调试
        }