from collections import Counter
import math
import numpy as np
import pandas as pd


class LogKG:
    """
    LogKG 核心类：
    功能：把每个 case 的日志序列转成一个向量表示（case embedding）

    核心思想：
        类似 TF-IDF（Term Frequency - Inverse Document Frequency）
        但“词”变成了日志模板（EventId）
        再乘上模板的语义向量（template embedding）
        最终得到 case 级别的加权向量表示
    """

    def __init__(
        self,
        train_case_log_df: dict[str, pd.DataFrame],   # 训练集：{case_name: 日志DataFrame}
        test_case_log_df: dict[str, pd.DataFrame],    # 测试集：{case_name: 日志DataFrame}
        idf_threshold: float,                         # IDF 阈值，小于该值的模板被忽略
        template_embedding: dict,                     # {template_id: embedding_vector}
        embedding_size: int | None = None,            # embedding 维度（可自动推断）
    ) -> None:
        self.train_case_log_df = train_case_log_df
        self.test_case_log_df = test_case_log_df
        self.idf_threshold = idf_threshold
        self.template_embedding = template_embedding

        # 若未手动指定 embedding_size，则自动推断
        self.embedding_size = embedding_size or self._infer_embedding_size(template_embedding)

    # ============================
    # 自动推断 embedding 维度
    # ============================
    def _infer_embedding_size(self, template_embedding: dict) -> int:
        """
        从 template_embedding 中取第一个向量，推断 embedding 维度
        """
        if not template_embedding:
            raise ValueError("template_embedding is empty.")

        first_vector = np.asarray(next(iter(template_embedding.values())))

        # 必须是一维向量
        if first_vector.ndim != 1:
            raise ValueError("template embedding vector must be 1-D.")

        embedding_size = int(first_vector.shape[0])

        if embedding_size <= 0:
            raise ValueError("embedding_size must be positive.")

        return embedding_size


    # ============================
    # 计算训练集 IDF
    # ============================
    def get_train_idf(self):
        """
        计算每个日志模板的 IDF：

        IDF = log10( 总case数 / 包含该模板的case数 )

        直觉：
            出现越普遍的模板（比如系统启动、正常日志）
            IDF 越小 => 不重要

            只在少数case中出现的模板
            IDF 越大 => 更有区分能力
        """

        # 每个 case 中“去重后的模板集合”
        case_log_set_list = [
            list(set(df["EventId"].values))
            for df in self.train_case_log_df.values()
        ]

        # 收集所有 case 的模板
        case_all_template_occurrence = []
        for case_log_set in case_log_set_list:
            case_all_template_occurrence += case_log_set

        # 统计每个模板出现在多少个 case 中
        case_log_template_counter = dict(Counter(case_all_template_occurrence))

        # 保存训练阶段出现过的模板列表
        self.template_list = list(case_log_template_counter.keys())

        template_idf = {}

        for template in case_log_template_counter:
            idf = math.log10(
                len(case_log_set_list) / case_log_template_counter[template]
            )

            # 小于阈值的模板直接置为 0（相当于过滤掉）
            template_idf[template] = idf if idf > self.idf_threshold else 0.0

        self.template_idf = template_idf


    # ============================
    # 检查模板 embedding 是否合法
    # ============================
    def _validate_template_vector(self, template):
        """
        确保 template 的 embedding:
            1) 是一维向量
            2) 维度与 embedding_size 一致
        """
        vector = np.asarray(self.template_embedding[template], dtype=float)

        if vector.ndim != 1 or vector.shape[0] != self.embedding_size:
            raise ValueError(
                f"Template {template} embedding shape {vector.shape} does not match embedding_size={self.embedding_size}."
            )

        return vector


    # ============================
    # 生成训练集 case embedding
    # ============================
    def get_train_embedding(self):
        """
        对每个训练 case：

        case_embedding =
            Σ [ TF(template) * IDF(template) * template_embedding ]

        其中：
            TF = 模板在该case中的出现次数 / 重要模板总数
            IDF = 全局区分度
        """

        # 先计算训练IDF
        self.get_train_idf()

        case_embedding_dict = {}

        for key in self.train_case_log_df:
            log_df = self.train_case_log_df[key]
            template_sequence = log_df["EventId"].values

            # 统计当前case中模板出现次数
            case_template_counter = dict(Counter(template_sequence))

            important_log_count = 0

            # 统计“重要模板”的总次数
            for template in case_template_counter:
                if self.template_idf[template] != 0:
                    important_log_count += case_template_counter[template]
                else:
                    # IDF为0的模板直接忽略
                    case_template_counter[template] = 0

            # 初始化 embedding 向量
            case_embedding = np.zeros(self.embedding_size, dtype=float)

            # 如果没有重要日志，直接返回全零向量
            if important_log_count == 0:
                case_embedding_dict[key] = case_embedding
                continue

            # 加权求和
            for template in case_template_counter:
                case_embedding += (
                    (case_template_counter[template] / important_log_count)  # TF
                    * self.template_idf[template]                             # IDF
                    * self._validate_template_vector(template)                # 模板向量
                )

            case_embedding_dict[key] = case_embedding

        self.train_embedding_dict = case_embedding_dict


    # ============================
    # 生成测试集 embedding
    # ============================
    def get_test_embedding(self):
        """
        测试集 embedding 与训练类似，
        但：
            - 只能使用训练阶段出现过的模板
            - 使用训练阶段计算好的 IDF
        """

        case_embedding_dict = {}

        for key in self.test_case_log_df:
            log_df = self.test_case_log_df[key]
            template_sequence = log_df["EventId"].values

            case_template_counter = dict(Counter(template_sequence))
            important_log_count = 0

            for template in case_template_counter:

                # 若测试集中出现“训练没见过”的模板，直接跳过
                if template not in self.template_list:
                    continue

                if self.template_idf[template] != 0:
                    important_log_count += case_template_counter[template]
                else:
                    case_template_counter[template] = 0

            case_embedding = np.zeros(self.embedding_size, dtype=float)

            if important_log_count == 0:
                case_embedding_dict[key] = case_embedding
                continue

            for template in case_template_counter:

                if template not in self.template_list:
                    continue

                case_embedding += (
                    (case_template_counter[template] / important_log_count)
                    * self.template_idf[template]
                    * self._validate_template_vector(template)
                )

            case_embedding_dict[key] = case_embedding

        self.test_embedding_dict = case_embedding_dict