
import os
from typing import List, Optional, Dict, Any, Union

import httpx
from typing import Dict, Any

from typing import List, Dict, Any
import asyncio
import sys
import logging
import json
import json
import os
import sys
import re
import ast
import copy
import pickle
import logging
import time
import io
import base64
import asyncio
import signal
from typing import Dict, Any

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from PIL import Image
from multiprocessing import Process, Queue

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils import (
    extract_question_content,
    is_valid_base64_image,

)

from utils.logger import get_logger

from utils.image_utils import save_image_path

logger = get_logger("task_analysis")

# ================= 路径加载 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------
# 导入和配置
# --------------------------------------------------------------------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import base64
from io import BytesIO
from PIL import Image

def get_image_size(image_b64):
    # 去掉 data:image/png;base64, 前缀（如果有）
    if "," in image_b64:
        image_b64 = image_b64.split(",")[1]

    image_data = base64.b64decode(image_b64)

    image = Image.open(BytesIO(image_data))

    return image.size   # (width, height)

# VLLM URL 配置
VLLM_URL_OCR = "http://192.168.1.210:10030/v1/chat/completions"
DEFAULT_MODEL_OCR = "dots-mocr" # 示例模型名


# OCR_PROMPT = f"""
# OCR this image,only return the text content.
# """.strip()

OCR_PROMPT = """Extract the text content from this image."""
# --------------------------------------------------------------------
# 并发控制配置 (新增)
# --------------------------------------------------------------------
# OCR 最大并发数：限制同时向 VLLM_OCR 发送的图片请求数
OCR_MAX_CONCURRENCY = 5

# -------------------------------------------------------------------
# 辅助函数（_single_image_ocr, call_vllm_ocr, call_vllm_grade, parse_string_to_json, build_rubric_prompt）
# --------------------------------------------------------------------
# 使用新的统一推理器
async def process_questions_with_ocr(df: pd.DataFrame):
    datas = df.to_dict(orient='records')
    all_question = []
    required_fields = ['subject', 'questionStem', 'questionType', 'questionNo', 'knowledgeCode', 'knowledge']

    for index, data in enumerate(datas):
        uuid = data.get('uuid')
        # if uuid != "aeea787a-1bf2-402c-bc7e-31ee4f182698":
        #     continue
        if data.get('task_name') == 'answer_analysis':
            if any(data.get(field) is None or data.get(field) == "" for field in required_fields):
                continue    
        subject = data['subject']
        level = data['abilityLevel']
        question_no = data['questionNo']
        question_type = data['questionType']
        know_md = data['knowledgemd']
        answer_type_prompt_one = data['answer_type_prompt_one']
        answer_type_example_one = data['answer_type_example_one']

        question_dict = extract_question_content(data['questionMaterial'], data['questionStem'], data['answer'])
        material_text = question_dict['material']
        question_text = question_dict['question']
        answer_text = question_dict['answer']
        images_list = question_dict['images_pool']

        images_list_valid = [x for x in images_list if is_valid_base64_image(x)]
        if len(images_list_valid) != len(images_list):
            raise ValueError(f"Question {index + 1} has {len(images_list_valid)} valid images out of {len(images_list)} total images")
        images_list = images_list_valid
        ocr_images = True
        IMAGE_SAVE_DIR = os.path.join(os.path.dirname(current_dir), "data", "png")
        TXT_SAVE_DIR = os.path.join(os.path.dirname(current_dir), "data", "txt")
        os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
        if len(images_list) > 0 and ocr_images:
            if os.path.exists(os.path.join(TXT_SAVE_DIR, f"{uuid}.txt")):
                with open(os.path.join(TXT_SAVE_DIR, f"{uuid}.txt"), "r") as f:
                    ocr_text = f.readlines()
                    ocr_text = [x.replace("\n", "") for x in ocr_text]
            
                for i, text in enumerate(ocr_text, 1):
                    text_replace = "【"+ text +"】"
                    material_text = material_text.replace(f"【图片{i}】", text_replace)
                    question_text = question_text.replace(f"【图片{i}】", text_replace)
                images_list = []
            else:
                try:
                    ocr_text = await call_vllm_ocr(images_list)
                except:
                    ocr_text = ""
                ocr_text = [x.strip().replace("•", " ").replace("*", " ").replace("◆", " ").replace(" ", "") if x else "" for x in ocr_text]
                os.makedirs(TXT_SAVE_DIR, exist_ok=True)
                # with open(os.path.join(TXT_SAVE_DIR, f"{uuid}.txt"), "w") as f:
                #     f.write("\n".join(ocr_text))
                with open(os.path.join(TXT_SAVE_DIR, f"{uuid}.txt"), "w", encoding="utf-8") as f:
                    f.write("\n".join(" ".join(item.split()) for item in ocr_text))
                for i, text in enumerate(ocr_text, 1):
                    text_replace = "【"+ text +"】"
                    material_text = material_text.replace(f"【图片{i}】", text_replace)
                    question_text = question_text.replace(f"【图片{i}】", text_replace)
                path_list = []
                if images_list:
                    dir_path = os.path.join(IMAGE_SAVE_DIR, uuid)
                    os.makedirs(dir_path, exist_ok=True)
                    for index, img in enumerate(images_list):
                        save_path = os.path.join(dir_path, f"{index}.png")
                        path = save_image_path(img, save_path)
                        path_list.append(path)
                images_list = []

        uuid_one = data.get('uuid')
        knowledgeCode = data.get("knowledgeCode")
        knowledge_name = data.get("knowledge")
        answer_system = data.get("answer_system_prompt_one")
        task = data.get("task")
        question_dict_one = {
            'uuid': uuid_one,
            'question_no': question_no,
            'org_material': data['questionMaterial'],
            'org_question': data['questionStem'],
            'org_answer': data['answer'],
            'material': material_text,
            'question': question_text,
            'answer': answer_text,
            'question_type': question_type,
            'knowledgeCode': knowledgeCode,
            'knowledge_name': knowledge_name,
            'level': level,
            'knowledge': know_md,
            'promote_head': answer_system,
            'promote_out': answer_type_prompt_one,
            'answer_example': answer_type_example_one,
            'task': task,  # 关键：指定任务类型
            'image_list': images_list,
            'data': data
        }

        all_question.append(question_dict_one)
    return all_question

# async def _single_image_ocr(image_b64: str, model: str = DEFAULT_MODEL_OCR, semaphore: asyncio.Semaphore = None) -> str:
#     """ 
#     异步调用 VLLM OCR 服务处理单个 Base64 图片，并使用信号量进行限流。
#     """
#     if semaphore:
#         await semaphore.acquire() 

#     try:
#         contents = [
#             {"type": "text", "text": OCR_PROMPT}
#         ]
#         contents.append(
#             {
#                 "type": "image_url",
#                 "image_url": {
#                     "url": f"{image_b64}" 
#                 }
#             }
#         )

#         payload: Dict[str, Any] = {
#             "model": model,
#             "messages": [{"role": "user", "content": contents}],
#             "max_new_tokens": 4096,
#             "temperature": 0.3,
#             "top_k": 15,
#             "top_p": 0.95, 
#             "num_beams": 3 
#         }
#         OCR_TIMEOUT = 20
#         timeout_config = httpx.Timeout(OCR_TIMEOUT, connect=5.0) 
        
#         async with httpx.AsyncClient(timeout=timeout_config) as client:
#             resp = await client.post(VLLM_URL_OCR, json=payload)
        
#         resp.raise_for_status()
        
#         response_data = resp.json()
#         return response_data["choices"][0]["message"]["content"]
        
#     except httpx.TimeoutException as e:
#         logger.error(f"VLLM OCR 单图片请求超时 ({OCR_TIMEOUT}s)。")
#         raise RuntimeError(f"VLLM OCR 单图片请求超时 ({OCR_TIMEOUT}s)。")
#     except httpx.HTTPStatusError as e:
#         if get_image_size(image_b64)[0] <= 1 or get_image_size(image_b64)[1] <= 1:
#             return "[]"
#         else:
#             logger.error(f"VLLM OCR 失败: HTTP {e.response.status_code}")
#             raise RuntimeError(f"VLLM OCR 失败: HTTP {e.response.status_code}")
#     except httpx.RequestError as e:
#         logger.error(f"VLLM OCR 请求失败: {e}")
#         raise RuntimeError(f"VLLM OCR 请求失败")
#     except (KeyError, ValueError) as e:
#         logger.error(f"VLLM OCR 响应格式错误: {e}")
#         raise RuntimeError(f"VLLM OCR 响应格式错误: {e}")
#     finally:
#         if semaphore:
#             semaphore.release()

import os
import base64
import asyncio
import logging
import mimetypes
from typing import List, Optional, Union

from openai import AsyncOpenAI
from openai import APIConnectionError, APITimeoutError, APIStatusError


logger = logging.getLogger(__name__)

QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "http://192.168.1.159:21000/v1")
DEFAULT_MODEL_OCR = os.getenv("DEFAULT_MODEL_OCR", "Qwen3Coder")
OCR_MAX_CONCURRENCY = int(os.getenv("OCR_MAX_CONCURRENCY", "8"))
OCR_TIMEOUT = float(os.getenv("OCR_TIMEOUT", "60"))

OCR_PROMPT = """
请识别图片中的文字，只返回 JSON 数组。

格式：
[
  {
    "text": "识别到的文字"
  }
]

如果图片中没有文字，返回 []。
不要输出 Markdown，不要解释。
"""


def image_path_to_data_url(image_path: str) -> str:
    """本地图片路径转 data:image/...;base64,..."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/jpeg"

    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{image_base64}"


def ensure_data_url(image_b64: str, mime_type: str = "image/jpeg") -> str:
    """
    兼容两种输入：
    1. data:image/jpeg;base64,xxx
    2. 纯 base64 字符串
    """
    if image_b64.startswith("data:image/"):
        return image_b64

    return f"data:{mime_type};base64,{image_b64}"


async def _single_image_ocr(
    image_b64: str,
    client: AsyncOpenAI,
    model: str = DEFAULT_MODEL_OCR,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> str:
    """
    异步调用本地 OpenAI-compatible OCR 服务处理单张 Base64 图片。
    """

    async def _call() -> str:
        image_url = ensure_data_url(image_b64)

        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            },
            {
                "type": "text",
                "text": OCR_PROMPT
            }
        ]

        try:
            completion = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": False
                    }
                },
                max_tokens=4000,
                temperature=0.2,
                timeout=OCR_TIMEOUT,
            )

            return completion.choices[0].message.content or "[]"

        except APITimeoutError:
            logger.error(f"Qwen OCR 单图片请求超时：{OCR_TIMEOUT}s")
            raise RuntimeError(f"Qwen OCR 单图片请求超时：{OCR_TIMEOUT}s")

        except APIStatusError as e:
            logger.error(f"Qwen OCR HTTP 失败：status={e.status_code}, body={e.response.text}")
            raise RuntimeError(f"Qwen OCR HTTP 失败：{e.status_code}")

        except APIConnectionError as e:
            logger.error(f"Qwen OCR 连接失败：{e}")
            raise RuntimeError("Qwen OCR 连接失败")

        except Exception as e:
            logger.exception(f"Qwen OCR 响应或请求异常：{e}")
            raise RuntimeError(f"Qwen OCR 响应或请求异常：{e}")

    if semaphore is not None:
        async with semaphore:
            return await _call()

    return await _call()


async def call_vllm_ocr(
    images_b64: List[str],
    model: str = DEFAULT_MODEL_OCR,
) -> List[str]:
    """
    批量异步调用 Qwen OCR。

    Args:
        images_b64: 图片 base64 列表，可以是纯 base64，也可以是 data:image/jpeg;base64,...
        model: 模型名，例如 Qwen3Coder

    Returns:
        每张图片的 OCR 结果列表，顺序与输入一致
    """
    if not images_b64:
        return []

    logger.info(f"开始处理 {len(images_b64)} 张图片")

    semaphore = asyncio.Semaphore(OCR_MAX_CONCURRENCY)

    client = AsyncOpenAI(
        base_url=QWEN_BASE_URL,
        api_key="EMPTY",
        timeout=OCR_TIMEOUT,
    )

    tasks = [
        _single_image_ocr(
            image_b64=img_b64,
            client=client,
            model=model,
            semaphore=semaphore,
        )
        for img_b64 in images_b64
    ]

    try:
        results = await asyncio.gather(*tasks)
        logger.info(f"成功识别 {len(results)} 张图片")
        return results

    except Exception as e:
        logger.error(f"并发 Qwen OCR 识别过程中发生错误：{e}")
        raise RuntimeError("并发 Qwen OCR 识别过程中发生错误")

# async def call_vllm_ocr(images_b64: List[str], model: str = DEFAULT_MODEL_OCR) -> str:
#     """ 
#     异步调用 VLLM OCR 服务，使用信号量限制并发数量，将图片列表转换为文本。
    
#     Args:
#         images_b64: Base64 编码的图片列表
#         model: 使用的模型名称
        
#     Returns:
#         所有图片 OCR 识别结果的拼接字符串
#     """
#     logger.info(f"开始处理 {len(images_b64)} 张图片")
    
#     semaphore = asyncio.Semaphore(OCR_MAX_CONCURRENCY)

#     tasks = []
#     for img_b64 in images_b64:
#         task = _single_image_ocr(img_b64, model, semaphore)
#         tasks.append(task)
        
#     try:
#         ordered_ocr_results = await asyncio.gather(*tasks)
#         logger.info(f"成功识别 {len(ordered_ocr_results)} 张图片")
#     except RuntimeError as e:
#         logger.error(f"并发 OCR 识别过程中发生错误: {e}")
#         raise RuntimeError(f"并发 OCR 识别过程中发生错误")
        
#     return ordered_ocr_results


if __name__ == "__main__":
    asyncio.run(call_vllm_ocr(["base64_encoded_image1", "base64_encoded_image2"]))