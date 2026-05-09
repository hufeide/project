
import os
from typing import List, Optional, Dict, Any, Union

import httpx
from typing import Dict, Any

from typing import List, Dict, Any
import asyncio
import sys
import logging
import json
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --------------------------------------------------------------------
# 导入和配置
# --------------------------------------------------------------------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


# VLLM URL 配置
VLLM_URL_OCR = "http://192.168.1.159:10050/v1/chat/completions"
DEFAULT_MODEL_OCR = "" # 示例模型名


OCR_PROMPT = f"""
OCR this image,only return the text content.
""".strip()
# --------------------------------------------------------------------
# 并发控制配置 (新增)
# --------------------------------------------------------------------
# OCR 最大并发数：限制同时向 VLLM_OCR 发送的图片请求数
OCR_MAX_CONCURRENCY = 5

# -------------------------------------------------------------------
# 辅助函数（_single_image_ocr, call_vllm_ocr, call_vllm_grade, parse_string_to_json, build_rubric_prompt）
# --------------------------------------------------------------------

async def _single_image_ocr(image_b64: str, model: str = DEFAULT_MODEL_OCR, semaphore: asyncio.Semaphore = None) -> str:
    """ 
    异步调用 VLLM OCR 服务处理单个 Base64 图片，并使用信号量进行限流。
    """
    if semaphore:
        await semaphore.acquire() 

    try:
        contents = [
            {"type": "text", "text": OCR_PROMPT}
        ]
        contents.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"{image_b64}" 
                }
            }
        )

        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": contents}],
            "max_new_tokens": 4096,
            "temperature": 0.3,
            "top_k": 15,
            "top_p": 0.95, 
            "num_beams": 3 
        }
        OCR_TIMEOUT = 20
        timeout_config = httpx.Timeout(OCR_TIMEOUT, connect=5.0) 
        
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            resp = await client.post(VLLM_URL_OCR, json=payload)
        
        resp.raise_for_status()
        
        response_data = resp.json()
        return response_data["choices"][0]["message"]["content"]
        
    except httpx.TimeoutException as e:
        logger.error(f"VLLM OCR 单图片请求超时 ({OCR_TIMEOUT}s)。")
        raise RuntimeError(f"VLLM OCR 单图片请求超时 ({OCR_TIMEOUT}s)。")
    except httpx.HTTPStatusError as e:
        logger.error(f"VLLM OCR 失败: HTTP {e.response.status_code}")
        raise RuntimeError(f"VLLM OCR 失败: HTTP {e.response.status_code}")
    except httpx.RequestError as e:
        logger.error(f"VLLM OCR 请求失败: {e}")
        raise RuntimeError(f"VLLM OCR 请求失败")
    except (KeyError, ValueError) as e:
        logger.error(f"VLLM OCR 响应格式错误: {e}")
        raise RuntimeError(f"VLLM OCR 响应格式错误: {e}")
    finally:
        if semaphore:
            semaphore.release()

async def call_vllm_ocr(images_b64: List[str], model: str = DEFAULT_MODEL_OCR) -> str:
    """ 
    异步调用 VLLM OCR 服务，使用信号量限制并发数量，将图片列表转换为文本。
    
    Args:
        images_b64: Base64 编码的图片列表
        model: 使用的模型名称
        
    Returns:
        所有图片 OCR 识别结果的拼接字符串
    """
    logger.info(f"开始处理 {len(images_b64)} 张图片")
    
    semaphore = asyncio.Semaphore(OCR_MAX_CONCURRENCY)

    tasks = []
    for img_b64 in images_b64:
        task = _single_image_ocr(img_b64, model, semaphore)
        tasks.append(task)
        
    try:
        ordered_ocr_results = await asyncio.gather(*tasks)
        logger.info(f"成功识别 {len(ordered_ocr_results)} 张图片")
    except RuntimeError as e:
        logger.error(f"并发 OCR 识别过程中发生错误: {e}")
        raise RuntimeError(f"并发 OCR 识别过程中发生错误")
        
    return ordered_ocr_results


if __name__ == "__main__":
    asyncio.run(call_vllm_ocr(["base64_encoded_image1", "base64_encoded_image2"]))