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
    "max_tokens": int(os.getenv("VLLM_MAX_TOKENS", "180000")),
}

IMAGE_SAVE_DIR = os.getenv(
    "IMAGE_SAVE_DIR",
    os.path.join(PROJECT_ROOT, "难易度", "gradio_me", "png"),
)
