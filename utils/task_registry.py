"""
任务注册中心
统一管理所有任务类型的配置和处理器
"""
from .unified_inference import (
    TaskConfig, AnswerAnalysisProcessor, KnowledgeJudgmentProcessor, 
    AnswerComparisonProcessor, unified_inference
)
from .prompt_task import prompt_answer_analysis

# ===== 任务配置 =====

# 答题分析任务
ANSWER_ANALYSIS_CONFIG = TaskConfig(
    name="答题分析",
    prompt_generator=prompt_answer_analysis,
    required_keys={"题号", "试题分析", "答题分析"},
    model_selection="auto",
    use_images=True
)

# 知识点判定任务
KNOWLEDGE_JUDGMENT_CONFIG = TaskConfig(
    name="知识点判定",
    prompt_generator=prompt_answer_analysis,  # 暂时复用，后续可单独实现
    required_keys={"题号", "知识点", "判定结果"},
    model_selection="auto",
    use_images=True
)

# 答案比对任务
ANSWER_COMPARISON_CONFIG = TaskConfig(
    name="答案比对",
    prompt_generator=prompt_answer_analysis,  # 暂时复用，后续可单独实现
    required_keys={"题号", "标准答案", "学生答案", "比对结果"},
    model_selection="auto",
    use_images=True
)

# ===== 注册任务 =====

# 注册答题分析任务
unified_inference.register_task(
    "answer_analysis",
    ANSWER_ANALYSIS_CONFIG,
    AnswerAnalysisProcessor(prompt_answer_analysis)
)

# 注册知识点判定任务
unified_inference.register_task(
    "answer_knowledge",
    KNOWLEDGE_JUDGMENT_CONFIG,
    KnowledgeJudgmentProcessor(prompt_answer_analysis)
)

# 注册答案比对任务
unified_inference.register_task(
    "answer_correct",
    ANSWER_COMPARISON_CONFIG,
    AnswerComparisonProcessor(prompt_answer_analysis)
)

# 注册难度判定任务
unified_inference.register_task(
    "answer_difficulty",
    ANSWER_ANALYSIS_CONFIG,  # 复用答题分析配置
    AnswerAnalysisProcessor(prompt_answer_analysis)
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