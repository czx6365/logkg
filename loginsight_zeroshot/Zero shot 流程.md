Zero shot 流程

"summary": {

​    "micro_f1": 0.4297149276786968,

​    "macro_f1": 0.17175105029640578,

​    "weighted_f1": 0.4081983774570477,

​    "validity_rate": 1.0,

​    "n_cases": 7121

  },

**1. 读取已经预处理好的 major 数据**
入口脚本是 run_inference.py，它从配置里的 processed_cases 读取样本，然后取出整批 case 的候选标签集合。
这次对应的数据文件是：
os_preprocessed_b507_major.jsonl

**2. 每个 case 先做 FOLS 摘要**
真正的 agent 在 agent.py 里。
它会先对 content_sequence 做 summarize_case(...)，把长日志压成 fault_summary，也就是论文里那种 “Fault-Oriented Log Summary” 思路。
如果摘要太长，还会做裁剪，避免超出模型上下文。

**3. 用 zero-shot prompt 直接问 Qwen**
这次不是微调模型，也没有 adapter。
它是把：

- 候选 major 标签列表
- FOLS 摘要
- 固定 instruction

一起拼成 prompt，然后直接送给本机 Qwen 推理。
这次 baseline 用的配置是：
os_b507_major_zero_shot_5fold.yaml

这个 baseline prompt 很简单，本质上就是：

- 你是日志故障诊断专家
- 从候选标签里选一个最可能的 fault type