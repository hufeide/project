import asyncio
import os

from openai import AsyncOpenAI

from .prompt_task import prompt_answer_analysis
from .modle_utils import get_chat_response, get_chat_response_vLLM

from .image_utils import save_image_path
from .json_validation import validate_single_json_string, is_list_of_list

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
    os.path.join(PROJECT_ROOT, "data", "png"),
)

client1 = VLLM_CLIENTS["client1"]
client2 = VLLM_CLIENTS["client2"]

class KnowledgeEnhancedQA_list:
    def __init__(self):
        self.client = ARK_CLIENT

    def batch_inference(self, processed_data_list, max_workers=5):
        return asyncio.run(self._run_batch(processed_data_list, max_workers))

    async def _run_batch(self, data_list, max_workers):
        sem = asyncio.Semaphore(max_workers)

        async def sem_task(data):
            async with sem:
                return await self.Batch_is_easy_pre_numList_async(data)

        tasks = [sem_task(data) for data in data_list]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def Batch_is_easy_pre_numList_async(self, processed_data):
        if processed_data['task'] in ['answer_knowledge','answer_correct', 'answer_analysis','answer_difficulty']:
            sys_prompt, prompt = prompt_answer_analysis(processed_data)
        else:
            raise ValueError(f"Unknown task type: {processed_data['task']}")

        raw_images = processed_data.get('image_list', [])
        if is_list_of_list(raw_images):
            image_list = [img for group in raw_images for img in group]
        else:
            image_list = raw_images if isinstance(raw_images, list) else [raw_images]

        path_list = []
        if image_list:
            dir_path = os.path.join(IMAGE_SAVE_DIR, processed_data['uuid'])
            os.makedirs(dir_path, exist_ok=True)
            for index, img in enumerate(image_list):
                save_path = os.path.join(dir_path, f"{index}.png")
                path = save_image_path(img, save_path)
                path_list.append(path)

        async def safe_call(coro):
            try:
                return await coro
            except Exception as e:
                print(f"捕获到异常: {e}")
                return None

        async def call_model1():
            def _sync_call():
                imgs = [i for i in image_list]
                if len(imgs) > 0:
                    resp = get_chat_response_vLLM(
                        messages=[
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        image_base64_list=imgs
                    )
                    resp = resp.replace('抱歉，API 调用失败，请检查您的 API Key、网络或余额。', '')
                    if not resp.strip():
                        resp = get_chat_response([
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": prompt}
                        ])
                else:
                    resp = get_chat_response([
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": prompt}
                    ])
                return resp
            return await asyncio.to_thread(_sync_call)

        async def call_model_vllm(client):
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
                resp = response.choices[0].message.content
            except Exception as e:
                print(f"vLLM 调用失败: {e}")
                resp = ""

            return resp

        async def call_model2():
            def _sync_call():
                completion = self.client.chat.completions.create(
                    model=ARK_MODELS["deepseek"],
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

        async def call_model3():
            def _sync_call():
                completion = self.client.chat.completions.create(
                    model=ARK_MODELS["doubao"],
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

        if len(image_list) > 0:
            res1, res2 = await asyncio.gather(safe_call(call_model_vllm(client1)), safe_call(call_model_vllm(client2)))
            model1_name = "chatgpt"
            model2_name = "doubao"
        else:
            res1, res2 = await asyncio.gather(safe_call(call_model_vllm(client1)), safe_call(call_model_vllm(client2)))
            model1_name = "chatgpt"
            model2_name = "deepseek"

        required_keys = processed_data['required_keys']
        res1 = validate_single_json_string(res1, required_keys)
        res1['model_name'] = model1_name
        res2 = validate_single_json_string(res2, required_keys)
        res2['model_name'] = model2_name

        return {
            "res1": res1,
            "res2": res2,
            "prompt": f"系统提示词：{sys_prompt} /n 普通提示词：{prompt}"
        }
