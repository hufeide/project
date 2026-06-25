import os
from dotenv import load_dotenv
from openai import OpenAI
import json
import base64
import imghdr
from openai import APIConnectionError, APIStatusError, RateLimitError

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY == "" or not OPENAI_API_KEY:
    print("❌ 错误：请在脚本中设置您的实际 OpenAI API Key。")
    exit()

try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    print(f"❌ 客户端初始化失败，请检查您的 API Key 是否正确: {e}")
    exit()

conversation_history = [
    {"role": "system", "content": "你是一个教育专家。请用简洁、友好的中文回答用户的问题。"}
]

def get_chat_response(messages):
    try:
        print("\n🤖 正在思考...\n")

        response = client.responses.create(
            model="gpt-5.1",
            input=messages,
            max_output_tokens=30000,
            reasoning={ "effort": "low"},
            text={"verbosity": "low"},
            prompt_cache_retention="24h",
        )

        assistant_response = response.output_text
        return assistant_response

    except APIConnectionError as e:
        print("APIConnectionError:", repr(e))
    except APIStatusError as e:
        print("APIStatusError:", e.status_code, e.response)
    except RateLimitError as e:
        print("RateLimitError:", repr(e))

def get_chat_response_vLLM(messages, image_base64_list=None):
    try:
        content_list = []
        if isinstance(messages, str):
            content_list.append({
                "type": "input_text",
                "text": messages
            })
        else:
            for msg in messages:
                content_list.append({
                    "type": "input_text",
                    "text": str(msg)
                })

        if image_base64_list:
            for img_b64 in image_base64_list:
                content_list.append({
                    "type": "input_image",
                    "image_url": img_b64
                })

        payload = [
            {
                "role": "user",
                "content": content_list,
            }
        ]

        response = client.responses.create(
            model="gpt-5.1",
            input=payload,
            max_output_tokens=30000,
            reasoning={ "effort": "low" },
            text={"verbosity": "low"},
            prompt_cache_retention="24h",
        )

        return response.output_text

    except APIConnectionError as e:
        print("APIConnectionError:", repr(e))
    except APIStatusError as e:
        print("APIStatusError:", e.status_code, e.response)
    except RateLimitError as e:
        print("RateLimitError:", repr(e))


def main():
    print("--- ChatGPT 命令行对话脚本 ---")
    print("输入 '退出' 或 'exit' 结束对话。")
    print("-------------------------------")

    system_message = conversation_history[0]["content"]
    print(f"✅ 系统提示: {system_message}")

    while True:
        user_input = input("👤 您: ").strip()

        if user_input.lower() in ["退出", "exit"]:
            print("\n👋 感谢使用，对话结束。")
            break

        if not user_input:
            continue

        conversation_history.append({"role": "user", "content": user_input})

        response = get_chat_response(conversation_history)

        print(f"🤖 助手: {response}")
        conversation_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
