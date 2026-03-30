"""
统一模型推理架构
支持多种任务类型：答题分析、知识点判定、答案比对等
"""
import asyncio
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass

from .json_validation import validate_single_json_string, is_list_of_list
from .image_utils import save_image_path

# ===== 配置导入 =====
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from volcenginesdkarkruntime import Ark

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")

load_dotenv(ENV_PATH)

VLLM_CLIENTS = {
    "client1": AsyncOpenAI(
        api_key=os.getenv("VLLM_CLIENT1_API_KEY", ""),
        base_url=os.getenv("VLLM_CLIENT1_BASE_URL", "http://192.168.1.210:19000/v1"),
    ),
    "client2": AsyncOpenAI(
        api_key=os.getenv("VLLM_CLIENT2_API_KEY", ""),
        base_url=os.getenv("VLLM_CLIENT2_BASE_URL", "http://192.168.1.159:21000/v1"),
    ),
}

ARK_CLIENT = Ark(
    base_url=os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
    api_key=os.getenv("ARK_API_KEY", ""),
)

ARK_MODELS = {
    "deepseek": os.getenv("ARK_MODEL_DEEPSEEK", "deepseek-v3-2-251201"),
    "doubao": os.getenv("ARK_MODEL_DOUBAO", "doubao-seed-1-6-250615"),
}

ARK_DEFAULT_PARAMS = {
    "temperature": float(os.getenv("ARK_TEMPERATURE", "0.1")),
    "max_tokens": int(os.getenv("ARK_MAX_TOKENS", "30000")),
    "thinking": {"type": "enabled"},
}

VLLM_DEFAULT_PARAMS = {
    "temperature": float(os.getenv("VLLM_TEMPERATURE", "0.7")),
    "max_tokens": int(os.getenv("VLLM_MAX_TOKENS", "80000")),
}

IMAGE_SAVE_DIR = os.getenv(
    "IMAGE_SAVE_DIR",
    os.path.join(PROJECT_ROOT, "难易度", "gradio_me", "png"),
)

client1 = VLLM_CLIENTS["client1"]
client2 = VLLM_CLIENTS["client2"]


@dataclass
class TaskConfig:
    """任务配置"""
    name: str
    prompt_generator: Callable
    required_keys: set
    model_selection: str  # "auto", "vllm_only", "ark_only"
    use_images: bool = True


class BaseTaskProcessor(ABC):
    """任务处理器基类"""
    
    @abstractmethod
    def generate_prompt(self, data: Dict[str, Any]) -> tuple:
        """生成系统提示词和用户提示词"""
        pass
    
    @abstractmethod
    def validate_result(self, result: str, required_keys: set) -> Dict[str, Any]:
        """验证结果格式"""
        pass


class UnifiedModelInference:
    """统一模型推理器"""
    
    def __init__(self):
        self.client = ARK_CLIENT
        self.task_registry: Dict[str, TaskConfig] = {}
        self.task_processors: Dict[str, BaseTaskProcessor] = {}
    
    def register_task(self, task_name: str, config: TaskConfig, processor: BaseTaskProcessor):
        """注册任务类型"""
        self.task_registry[task_name] = config
        self.task_processors[task_name] = processor
    
    def batch_inference(self, data_list: List[Dict[str, Any]], max_workers: int = 5):
        """批量推理"""
        return asyncio.run(self._run_batch(data_list, max_workers))
    
    async def _run_batch(self, data_list: List[Dict[str, Any]], max_workers: int):
        """异步批量推理"""
        sem = asyncio.Semaphore(max_workers)
        
        async def sem_task(data):
            async with sem:
                return await self._process_single_data(data)
        
        tasks = [sem_task(data) for data in data_list]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _process_single_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个数据"""
        task_name = data.get('task')
        if task_name not in self.task_registry:
            raise ValueError(f"未知任务类型: {task_name}")
        
        config = self.task_registry[task_name]
        processor = self.task_processors[task_name]
        
        # 生成提示词
        sys_prompt, prompt = processor.generate_prompt(data)
        
        # 处理图片
        raw_images = data.get('image_list', [])
        image_list = self._process_images(raw_images, data.get('uuid'))
        
        # 选择模型
        model_results = await self._call_models(
            sys_prompt, prompt, image_list, config, task_name
        )
        
        # 验证结果
        validated_results = {}
        for model_name, result in model_results.items():
            validated = processor.validate_result(result, config.required_keys)
            validated['model_name'] = model_name
            validated_results[model_name] = validated
        
        return {
            "results": validated_results,
            "prompt_info": f"系统提示词：{sys_prompt}\n用户提示词：{prompt}",
            "original_data": data
        }
    
    def _process_images(self, raw_images: List, uuid: str) -> List[str]:
        """处理图片列表"""
        if is_list_of_list(raw_images):
            image_list = [img for group in raw_images for img in group]
        else:
            image_list = raw_images if isinstance(raw_images, list) else [raw_images]
        
        # 保存图片到本地
        path_list = []
        if image_list:
            dir_path = os.path.join(IMAGE_SAVE_DIR, uuid)
            os.makedirs(dir_path, exist_ok=True)
            for index, img in enumerate(image_list):
                save_path = os.path.join(dir_path, f"{index}.png")
                path = save_image_path(img, save_path)
                path_list.append(path)
        
        return image_list
    
    async def _call_models(self, sys_prompt: str, prompt: str, 
                          image_list: List[str], config: TaskConfig, task_name: str) -> Dict[str, str]:
        """调用模型 - 分发到不同任务的工作流"""
        
        workflow_map = {
            "answer_analysis": self._workflow_answer_analysis,
            "answer_knowledge": self._workflow_answer_knowledge,
            "answer_correct": self._workflow_answer_correct,
            "answer_difficulty": self._workflow_answer_difficulty,
        }
        
        workflow_func = workflow_map.get(task_name)
        if workflow_func:
            return await workflow_func(sys_prompt, prompt, image_list, config)
        else:
            return await self._workflow_default(sys_prompt, prompt, image_list, config)
    
    async def safe_call(self, coro, model_name: str, max_retries: int = 3):
        """统一的重试机制安全调用方法"""
        for attempt in range(max_retries):
            try:
                result = await coro
                if result and result.strip():
                    return result, model_name
            except Exception as e:
                print(f"{model_name} 第{attempt + 1}次调用失败: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
        return "", model_name
    
    async def _workflow_answer_analysis(self, sys_prompt: str, prompt: str, 
                                        image_list: List[str], config: TaskConfig) -> Dict[str, str]:
        """答题分析工作流：双模型生成 + 比对"""
        results = {}
        print("工作流: 答题分析 (answer_analysis) - 双模型生成 + 比对")
        
        first_stage_tasks = [
            self.safe_call(self._call_vllm(client1, sys_prompt, prompt, image_list), "vllm_model1"),
            self.safe_call(self._call_ark(ARK_MODELS["deepseek"], sys_prompt, prompt, image_list), "ark_deepseek")
        ]
        first_stage_results = await asyncio.gather(*first_stage_tasks)
        
        for result, model_name in first_stage_results:
            results[f"stage1_{model_name}"] = result
        
        valid_results = [r for r, _ in first_stage_results if r and r.strip()]
        if len(valid_results) >= 2:
            comparison_prompt = self._build_comparison_prompt(valid_results[0], valid_results[1], prompt)
            comparison_result = await self.safe_call(
                self._call_ark(ARK_MODELS["doubao"], sys_prompt, comparison_prompt, image_list),
                "comparison_model"
            )
            results["stage2_comparison"] = comparison_result[0]
        
        return results
    
    async def _workflow_answer_knowledge(self, sys_prompt: str, prompt: str, 
                                         image_list: List[str], config: TaskConfig) -> Dict[str, str]:
        """知识点判定工作流：单模型快速判定"""
        results = {}
        print("工作流: 知识点判定 (answer_knowledge) - 单模型快速判定")
        
        task = self.safe_call(
            self._call_ark(ARK_MODELS["deepseek"], sys_prompt, prompt, image_list),
            "ark_deepseek"
        )
        result = await task
        results[result[1]] = result[0]
        
        return results
    
    async def _workflow_answer_correct(self, sys_prompt: str, prompt: str, 
                                       image_list: List[str], config: TaskConfig) -> Dict[str, str]:
        """答案比对工作流：双模型并行 + 比对"""
        results = {}
        print("工作流: 答案比对 (answer_correct) - 双模型并行 + 比对")
        
        first_stage_tasks = [
            self.safe_call(self._call_vllm(client1, sys_prompt, prompt, image_list), "vllm_model1"),
            self.safe_call(self._call_vllm(client2, sys_prompt, prompt, image_list), "vllm_model2")
        ]
        first_stage_results = await asyncio.gather(*first_stage_tasks)
        
        for result, model_name in first_stage_results:
            results[f"stage1_{model_name}"] = result
        
        valid_results = [r for r, _ in first_stage_results if r and r.strip()]
        if len(valid_results) >= 2:
            comparison_prompt = self._build_comparison_prompt(valid_results[0], valid_results[1], prompt)
            comparison_result = await self.safe_call(
                self._call_ark(ARK_MODELS["deepseek"], sys_prompt, comparison_prompt, image_list),
                "comparison_model"
            )
            results["stage2_comparison"] = comparison_result[0]
        
        return results
    
    async def _workflow_answer_difficulty(self, sys_prompt: str, prompt: str, 
                                          image_list: List[str], config: TaskConfig) -> Dict[str, str]:
        """难度判定工作流：单模型判定"""
        results = {}
        print("工作流: 难度判定 (answer_difficulty) - 单模型判定")
        
        task = self.safe_call(
            self._call_vllm(client1, sys_prompt, prompt, image_list),
            "vllm_model1"
        )
        result = await task
        results[result[1]] = result[0]
        
        return results
    
    async def _workflow_default(self, sys_prompt: str, prompt: str, 
                                image_list: List[str], config: TaskConfig) -> Dict[str, str]:
        """默认工作流：多模型并行"""
        results = {}
        print("工作流: 默认 - 多模型并行")
        
        if config.model_selection == "vllm_only":
            tasks = [
                self.safe_call(self._call_vllm(client1, sys_prompt, prompt, image_list), "vllm_model1"),
                self.safe_call(self._call_vllm(client2, sys_prompt, prompt, image_list), "vllm_model2")
            ]
        elif config.model_selection == "ark_only":
            tasks = [
                self.safe_call(self._call_ark(ARK_MODELS["deepseek"], sys_prompt, prompt, image_list), "ark_deepseek"),
                self.safe_call(self._call_ark(ARK_MODELS["doubao"], sys_prompt, prompt, image_list), "ark_doubao")
            ]
        else:
            tasks = [
                self.safe_call(self._call_vllm(client1, sys_prompt, prompt, image_list), "vllm_model1"),
                self.safe_call(self._call_vllm(client2, sys_prompt, prompt, image_list), "vllm_model2"),
                self.safe_call(self._call_ark(ARK_MODELS["deepseek"], sys_prompt, prompt, image_list), "ark_deepseek")
            ]
        
        model_responses = await asyncio.gather(*tasks)
        
        for result, model_name in model_responses:
            results[model_name] = result
        
        return results
    
    def _build_comparison_prompt(self, result1: str, result2: str, original_prompt: str) -> str:
        """构建比对提示词"""
        return f"""
请对以下两个分析结果进行比对和整合：

原始问题：
{original_prompt}

分析结果1：
{result1}

分析结果2：
{result2}

请综合两个分析结果，给出一个更全面、准确的分析：
"""
    
    async def _call_vllm(self, client: AsyncOpenAI, sys_prompt: str, 
                        prompt: str, image_list: List[str]) -> str:
        """调用 vLLM 模型"""
        content = [{"type": "text", "text": prompt}]
        
        for img_b64 in image_list:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"{img_b64}"}
            })
        
        try:
            response = await client.chat.completions.create(
                model="",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": content}
                ],
                **VLLM_DEFAULT_PARAMS,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"vLLM 调用失败: {e}")
            return ""
    
    async def _call_ark(self, model_name: str, sys_prompt: str, 
                       prompt: str, image_list: List[str]) -> str:
        """调用 Ark 模型"""
        def _sync_call():
            completion = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": [
                        *[{"type": "image_url", "image_url": {"url": url}} for url in image_list],
                        {"type": "text", "text": prompt}
                    ]}
                ],
                **ARK_DEFAULT_PARAMS,
            )
            return completion.choices[0].message.content
        
        return await asyncio.to_thread(_sync_call)


# ===== 具体任务处理器实现 =====

class AnswerAnalysisProcessor(BaseTaskProcessor):
    """答题分析处理器"""
    
    def __init__(self, prompt_generator: Callable):
        self.prompt_generator = prompt_generator
    
    def generate_prompt(self, data: Dict[str, Any]) -> tuple:
        """生成答题分析提示词"""
        return self.prompt_generator(data)
    
    def validate_result(self, result: str, required_keys: set) -> Dict[str, Any]:
        """验证答题分析结果"""
        return validate_single_json_string(result, required_keys)


class KnowledgeJudgmentProcessor(BaseTaskProcessor):
    """知识点判定处理器"""
    
    def __init__(self, prompt_generator: Callable):
        self.prompt_generator = prompt_generator
    
    def generate_prompt(self, data: Dict[str, Any]) -> tuple:
        """生成知识点判定提示词"""
        # 这里可以添加知识点判定的特定逻辑
        return self.prompt_generator(data)
    
    def validate_result(self, result: str, required_keys: set) -> Dict[str, Any]:
        """验证知识点判定结果"""
        # 知识点判定可能有不同的验证逻辑
        return validate_single_json_string(result, required_keys)


class AnswerComparisonProcessor(BaseTaskProcessor):
    """答案比对处理器"""
    
    def __init__(self, prompt_generator: Callable):
        self.prompt_generator = prompt_generator
    
    def generate_prompt(self, data: Dict[str, Any]) -> tuple:
        """生成答案比对提示词"""
        # 这里可以添加答案比对的特定逻辑
        return self.prompt_generator(data)
    
    def validate_result(self, result: str, required_keys: set) -> Dict[str, Any]:
        """验证答案比对结果"""
        # 答案比对可能有不同的验证逻辑
        return validate_single_json_string(result, required_keys)


# ===== 全局推理器实例 =====
unified_inference = UnifiedModelInference()

# ===== 便捷函数，保持向后兼容 =====
class KnowledgeEnhancedQA_list:
    """兼容旧版本的类"""
    
    def __init__(self):
        self.inference = unified_inference
    
    def batch_inference(self, data_list: List[Dict[str, Any]], max_workers: int = 5):
        """批量推理"""
        return self.inference.batch_inference(data_list, max_workers)


# 创建默认实例
qa_system = KnowledgeEnhancedQA_list()