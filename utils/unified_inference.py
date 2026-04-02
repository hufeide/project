"""
统一模型推理架构
支持多种任务类型：答题分析、知识点判定、答案比对等
"""
import asyncio
import os
import sys
import signal
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Callable, Optional, Tuple
from dataclasses import dataclass
import json
from .json_validation import validate_single_json_string, is_list_of_list, is_validated_equal
from .image_utils import save_image_path
from .logger import get_logger
import pandas as pd
import copy
logger = get_logger("unified_inference")

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
    


class UnifiedModelInference:
    """统一模型推理器"""
    
    def __init__(self):
        self.client = ARK_CLIENT
        self.task_registry: Dict[str, TaskConfig] = {}
        self.task_processors: Dict[str, BaseTaskProcessor] = {}
        def _load_prompt(file_name: str) -> str:
            path = os.path.join(PROJECT_ROOT, "data", "prompt_file", file_name)
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        self.answer_analysis_consist = _load_prompt("task_answer_analysis_consist.txt")
        self.answer_correct_gen_consist = _load_prompt("task_answer_correct_gen_consist.txt")
    def validate_result(self, result: str, required_keys: set) -> Dict[str, Any]:
        """验证答题分析结果"""
        return validate_single_json_string(result, required_keys)

    def register_task(self, task_name: str, config: TaskConfig, processor: BaseTaskProcessor):
        """注册任务类型"""
        self.task_registry[task_name] = config
        self.task_processors[task_name] = processor
    
    def batch_inference(self, data_list: List[Dict[str, Any]], max_workers: int = 3):
        """批量推理，支持 Ctrl+C 中断"""
        # 设置信号处理器
        def signal_handler(signum, frame):
            logger.warning(f"\n收到信号 {signum}，正在安全退出...")
            sys.exit(130)
        
        # 注册信号处理器
        original_sigint = signal.signal(signal.SIGINT, signal_handler)
        original_sigterm = signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # 创建新的事件循环来确保干净的状态
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                return loop.run_until_complete(self._run_batch(data_list, max_workers))
            except (KeyboardInterrupt, asyncio.CancelledError):
                logger.warning("\n检测到中断信号，正在安全退出...")
                return []
            finally:
                # 确保所有任务都被取消
                pending = asyncio.all_tasks(loop)
                if pending:
                    logger.info(f"正在取消 {len(pending)} 个待处理任务...")
                    for task in pending:
                        task.cancel()
                    # 等待所有任务完成取消
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                
                loop.close()
                
        except Exception as e:
            logger.error(f"运行时错误: {e}")
            return []
        finally:
            # 恢复原始信号处理器
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)

    async def _run_batch(self, data_list: List[Dict[str, Any]], max_workers: int):
        sem = asyncio.Semaphore(max_workers)
        tasks = []

        async def sem_task(data):
            async with sem:
                try:
                    return await self._process_single_data(data)
                except asyncio.CancelledError:
                    logger.debug(f"任务被取消: {data.get('uuid', 'unknown')}")
                    raise

        try:
            # 创建任务列表
            tasks = [asyncio.create_task(sem_task(data)) for data in data_list]
            
            # 使用 asyncio.wait 而不是 gather，这样可以更好地处理取消
            done, pending = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)
            
            # 收集结果
            results = []
            for task in done:
                try:
                    result = await task
                    results.append(result)
                except asyncio.CancelledError:
                    logger.debug("任务被取消")
                except Exception as e:
                    logger.error(f"任务执行失败: {e}")
                    
            return results
            
        except (asyncio.CancelledError, KeyboardInterrupt):
            logger.warning("🛑 检测到中断，正在强制取消所有子任务...")
            raise
        finally:
            # 无论成功还是失败/取消，都检查并关闭所有任务
            if tasks:
                running_tasks = [t for t in tasks if not t.done()]
                if running_tasks:
                    logger.info(f"正在取消 {len(running_tasks)} 个运行中的任务...")
                    for t in running_tasks:
                        t.cancel()
                    # 给子任务一点时间来处理 CancelledError（例如关闭网络连接）
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*running_tasks, return_exceptions=True),
                            timeout=5.0  # 5秒超时
                        )
                    except asyncio.TimeoutError:
                        logger.warning("部分任务取消超时，强制退出")
                    logger.info(f"✅ 已成功清理 {len(running_tasks)} 个运行中的任务。")

    async def _process_single_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个数据"""
        try:
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
                data, sys_prompt, prompt, image_list, config, task_name
            )
            
            return {
                "results": model_results,
                "prompt_info": f"系统提示词：{sys_prompt}\n任务提示词：{prompt}",
                "original_data": data
            }
        except asyncio.CancelledError:
            logger.debug(f"单个数据处理被取消: {data.get('uuid', 'unknown')}")
            raise
        except Exception as e:
            logger.error(f"处理数据时出错: {e}")
            raise
    
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
    
    async def _call_models(self,data: Dict[str, Any], sys_prompt: str, prompt: str, 
                          image_list: List[str], config: TaskConfig, task_name: str) -> Dict[str, str]:
        """调用模型 - 分发到不同任务的工作流"""
        
        workflow_map = {
            "answer_analysis": self._workflow_answer_analysis,
            "answer_knowledge_gen": self._workflow_answer_analysis,
            "answer_correct_gen": self._workflow_answer_analysis,
            "answer_knowledge": self._workflow_answer_knowledge,
            "answer_correct": self._workflow_answer_knowledge,
            "answer_difficulty": self._workflow_answer_difficulty,
        }
        
        workflow_func = workflow_map.get(task_name)
        if workflow_func:
            return await workflow_func(data, sys_prompt, prompt, image_list, config)
        else:
            raise ValueError(f"未注册任务类型: {task_name}")
    
    async def safe_call(self, func: Callable, *args, model_name: str, max_retries: int = 3, config: TaskConfig = None, **kwargs) -> Tuple[str, str]:
        """
        统一的重试机制安全调用方法
        func: 异步函数名
        args/kwargs: 传递给异步函数的参数
        config: 任务配置，用于验证结果
        """
        for attempt in range(max_retries):
            try:
                # 检查是否被取消
                if asyncio.current_task().cancelled():
                    raise asyncio.CancelledError()
                    
                result = await func(*args, **kwargs)
                if config and config.required_keys:
                    validated = self.validate_result(result, config.required_keys)
                    if validated:
                        return validated, model_name
            except asyncio.CancelledError:
                # 重要：如果是取消信号，必须直接抛出，不能重试
                logger.debug(f"{model_name} 调用被取消")
                raise
            except Exception as e:
                logger.warning(f"{model_name} 第{attempt + 1}次调用失败: {e}")
            
            if attempt < max_retries - 1:
                try:
                    await asyncio.sleep(1)
                except asyncio.CancelledError:
                    raise
        return "", model_name

    async def _workflow_answer_analysis(self, data: Dict[str, Any], sys_prompt: str, prompt: str, 
                                        image_list: List[str], config: TaskConfig) -> Dict[str, str]:
        """工作流：双模型生成 + 比对"""
        task_name = data.get('task')
        if task_name == "answer_analysis":
            task_consist = self.answer_analysis_consist
        else:
            task_consist = self.answer_correct_gen_consist
        
        def _build_analysis_prompt(result1: str, result2: str) -> str:
            return f"""
## 题目信息
材料：
{data['material']}

题目：
{data['question']}

## 请对以下两个结果做判断：
### 结果1
{result1}

### 结果2
{result2}
"""

        results = {}
        print("工作流: 双模型生成 + 比对")
        
        # 传递函数引用和参数，而不是直接传 awaitable
        t1 = asyncio.create_task(self.safe_call(self._call_vllm, client1, sys_prompt, prompt, image_list, model_name="vllm_model1", config=config))
        t2 = asyncio.create_task(self.safe_call(self._call_vllm, client2, sys_prompt, prompt, image_list, model_name="vllm_model2", config=config))
        
        stage_tasks = [t1, t2]
        try:
            first_stage_results = await asyncio.gather(*stage_tasks)
        except (asyncio.CancelledError, KeyboardInterrupt):
            for t in stage_tasks:
                if not t.done(): t.cancel()
            # 等待一小会儿确保取消动作传播
            await asyncio.gather(*stage_tasks, return_exceptions=True)
            raise
                
        for result, model_name in first_stage_results:
            results[f"{model_name}"] = result

        valid_results = [r for r in results.values() if r['is_valid']]
        comparison_result = {"correct": "", "reason": "", 'is_valid': ""}

        if len(valid_results) >= 2:
            if is_validated_equal(valid_results[0], valid_results[1], ["answer", "kp_code", "question_type"]):
                comparison_result['correct'] = "是"
                comparison_result['reason'] = "两个模型结果一致"
                comparison_result['is_valid'] = True
            else:

                comparison_prompt = _build_analysis_prompt(json.dumps(valid_results[0],ensure_ascii=False), json.dumps(valid_results[1],ensure_ascii=False))
                # 修复：移除行尾逗号，正确传递参数
                config_compare = copy.deepcopy(config)
                config_compare.required_keys = {"correct", "better", "reason"}
                res, _ = await self.safe_call(self._call_vllm, client1, task_consist, comparison_prompt, image_list, model_name="comparison_model", config=config_compare)
        else:
            logger.warning("两个模型结果为空，无法进行比对")
            # results["model_judge"] = ""
        results['comparison_result'] = res
        return results

    async def _workflow_answer_knowledge(self, data: Dict[str, Any], sys_prompt: str, prompt: str, 
                                         image_list: List[str], config: TaskConfig) -> Dict[str, str]:
        """知识点判定工作流：任一模型快速判定"""
        results = {}
        print("工作流: 知识点判定 - 双模型竞争")
        
        # 逻辑：两个模型同时跑，谁先出结果且有效就用谁
        tasks = [
            self.safe_call(self._call_vllm, client1, sys_prompt, prompt, image_list, model_name="vllm_model1", config=config),
            # self.safe_call(self._call_vllm, client2, sys_prompt, prompt, image_list, model_name="vllm_model2")
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        for response in responses:
            # 检查是否是异常对象（防止程序崩溃）
            if isinstance(response, Exception):
                print(f"任务执行出错: {response}")
                continue
                
            result, model_name = response
            if result:  # 只有结果不为空才放入字典
                results[model_name] = result
        return results

    async def _workflow_answer_difficulty(self, data: Dict[str, Any], sys_prompt: str, prompt: str, 
                                          image_list: List[str], config: TaskConfig) -> Dict[str, str]:
        """难度判定工作流"""
        res, model_name = await self.safe_call(self._call_vllm, client1, sys_prompt, prompt, image_list, model_name="vllm_model1", config=config)
        return {model_name: res}

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
            # 检查是否被取消
            if asyncio.current_task().cancelled():
                raise asyncio.CancelledError()
                
            response = await client.chat.completions.create(
                model="",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": content}
                ],
                **VLLM_DEFAULT_PARAMS,
            )
            return response.choices[0].message.content
        except asyncio.CancelledError:
            logger.debug("vLLM 调用被取消")
            raise
        except Exception as e:
            logger.error(f"vLLM 调用失败: {e}")
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
        
        try:
            # 检查是否被取消
            if asyncio.current_task().cancelled():
                raise asyncio.CancelledError()
                
            return await asyncio.to_thread(_sync_call)
        except asyncio.CancelledError:
            logger.debug("Ark 调用被取消")
            raise


# ===== 具体任务处理器实现 =====

class Processor(BaseTaskProcessor):
    """答题分析处理器"""
    
    def __init__(self, prompt_generator: Callable):
        self.prompt_generator = prompt_generator
    
    def generate_prompt(self, data: Dict[str, Any]) -> tuple:
        """生成答题分析提示词"""
        return self.prompt_generator(data)
    


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