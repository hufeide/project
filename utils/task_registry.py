"""
任务注册中心
统一管理所有任务类型的配置和处理器
"""
from .unified_inference import (
    TaskConfig, Processor, unified_inference
)
from .prompt_task import prompt_answer_analysis, prompt_answer_knowledge, prompt_answer_correct,prompt_answer_knowledge_gen, prompt_answer_correct_gen,prompt_answer_difficulty

# ===== 任务配置 =====

# 答题分析任务
ANSWER_ANALYSIS_CONFIG = TaskConfig(
    name="answer_analysis",
    prompt_generator=prompt_answer_analysis,
    required_keys={"试题分析", "答题分析"},
    model_selection="vllm",
    use_images=True
)

# 知识点判定任务
KNOWLEDGE_JUDGMENT_CONFIG = TaskConfig(
    name="answer_knowledge",
    prompt_generator=prompt_answer_knowledge,  # 暂时复用，后续可单独实现
    required_keys={"kp_code", "kp","reason","human_correct"},
    model_selection="vllm",
    use_images=True
)

# 答案比对任务
ANSWER_COMPARISON_CONFIG = TaskConfig(
    name="answer_correct",
    prompt_generator=prompt_answer_correct,  # 暂时复用，后续可单独实现
    required_keys={"question_answer", "reason", "human_correct"},
    model_selection="vllm",
    use_images=True
)

# 知识点判定任务
KNOWLEDGE_GEN_CONFIG = TaskConfig(
    name="answer_knowledge_gen",
    prompt_generator=prompt_answer_knowledge_gen,  # 暂时复用，后续可单独实现
    required_keys={"kp_code", "kp","question_type","reason"},
    model_selection="vllm",
    use_images=True
)

# 答案比对任务
ANSWER_GEN_CONFIG = TaskConfig(
    name="answer_correct_gen",
    prompt_generator=prompt_answer_correct_gen,  # 暂时复用，后续可单独实现
    required_keys={"answer"},
    model_selection="vllm",
    use_images=True
)

# ===== 注册任务 =====

# 注册答题分析任务
unified_inference.register_task(
    "answer_analysis",
    ANSWER_ANALYSIS_CONFIG,
    Processor(prompt_answer_analysis)
)

# 注册知识点判定任务
unified_inference.register_task(
    "answer_knowledge",
    KNOWLEDGE_JUDGMENT_CONFIG,
    Processor(prompt_answer_knowledge)
)



# 注册答案比对任务
unified_inference.register_task(
    "answer_correct",
    ANSWER_COMPARISON_CONFIG,
    Processor(prompt_answer_correct)
)

# 注册知识点生成任务
unified_inference.register_task(
    "answer_knowledge_gen",
    KNOWLEDGE_GEN_CONFIG,
    Processor(prompt_answer_knowledge_gen)
)

# 注册答案生成任务
unified_inference.register_task(
    "answer_correct_gen",
    ANSWER_GEN_CONFIG,
    Processor(prompt_answer_correct_gen)
)
# 注册难度判定任务
unified_inference.register_task(
    "answer_difficulty",
    ANSWER_ANALYSIS_CONFIG,  # 复用答题分析配置
    Processor(prompt_answer_difficulty)
)

# ===== 便捷函数 =====

def get_task_config(task_name: str):
    """获取任务配置"""
    return unified_inference.task_registry.get(task_name)

def list_available_tasks():
    """列出所有可用的任务类型"""
    return list(unified_inference.task_registry.keys())

def is_task_supported(task_name: str) -> bool:
    """检查任务是否被支持"""
    return task_name in unified_inference.task_registry