import os
from openai import OpenAI
import json
import base64
import io
import imghdr
from openai import APIConnectionError, APIStatusError, RateLimitError
os.environ['http_proxy'] = 'http://172.16.50.213:7890'
os.environ['https_proxy'] = 'http://172.16.50.213:7890'

# ⚠️ 警告：请将 YOUR_OPENAI_API_KEY_HERE 替换为您自己的实际 API Key。
# 这种做法存在安全风险，请确保妥善保管您的密钥！
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY == "" or not OPENAI_API_KEY:
    print("❌ 错误：请在脚本中设置您的实际 OpenAI API Key。")
    exit()

# 初始化 OpenAI 客户端
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    print(f"❌ 客户端初始化失败，请检查您的 API Key 是否正确: {e}")
    exit()

# 对话历史，用于实现多轮对话
conversation_history = [
    {"role": "system", "content": "你是一个教育专家。请用简洁、友好的中文回答用户的问题。"}
]

# --- 函数定义 ---
def fix_base64(b64_str):
    # 去掉前后空白和 data:image/png;base64, 头
    if b64_str.startswith("data:"):
        b64_str = b64_str.split(",", 1)[1]

    b64_str = b64_str.strip()

    # 自动补齐 '='
    missing_padding = len(b64_str) % 4
    if missing_padding != 0:
        b64_str += "=" * (4 - missing_padding)

    return b64_str
def get_chat_response(messages):
    """
    调用 OpenAI API 获取模型响应。
    """
    try:
        print("\n🤖 正在思考...\n")
        
        # 调用 Chat Completion API
        response = client.responses.create(
            model="gpt-5.1",
            input=messages,
            max_output_tokens=30000,
            reasoning={ "effort": "low"},
            text={"verbosity": "low"},
            prompt_cache_retention="24h", 
        )

        
        # 提取助手的回复内容
        assistant_response = response.output_text
        return assistant_response

    # except Exception as e:
    #     print(f"\n❌ API 调用发生错误: {e}")
    #     # 如果需要查看详细错误，可以取消下面两行的注释
    #     # import traceback
    #     # traceback.print_exc()
    #     return "抱歉，API 调用失败，请检查您的 API Key、网络或余额。"
    except APIConnectionError as e:
        print("APIConnectionError:", repr(e))
    except APIStatusError as e:
        print("APIStatusError:", e.status_code, e.response)
    except RateLimitError as e:
        print("RateLimitError:", repr(e))

def base64_to_image_input(b64_str):
    # 去掉 data:image/...;base64, 头
    if b64_str.startswith("data:"):
        b64_str = b64_str.split(",", 1)[1]

    # 补齐 padding
    missing_padding = len(b64_str) % 4
    if missing_padding != 0:
        b64_str += "=" * (4 - missing_padding)

    # 解码
    img_bytes = base64.b64decode(b64_str)

    # 自动识别格式（png/jpeg）
    fmt = imghdr.what(None, img_bytes)
    mime = f"image/{fmt}" if fmt else "image/jpeg"

    return {
        "type": "input_image",
        "image": {
            "data": img_bytes,
            "mime_type": mime
        }
    }
def build_payload(messages, image_base64_list):
    payload = []

    # 用户文本消息
    content_list = []
    if isinstance(messages, str):
        content_list.append({"type": "input_text", "text": messages})
    else:
        for msg in messages:
            content_list.append({"type": "input_text", "text": str(msg)})

    payload.append({
        "role": "user",
        "content": content_list
    })

    # 图片消息放在顶层 input
    if image_base64_list:
        for b64 in image_base64_list:
            payload.append(base64_to_image_input(b64))

    return payload

# def get_chat_response_vLLM(messages, image_base64_list=None):
def get_chat_response_vLLM(messages, image_base64_list=None):
    """
    调用 OpenAI Responses API 获取模型响应。
    支持文本 + 图片（Base64）。
    """
    try:
        # print("\n🤖 正在思考...\n")

        # content 数组
        content_list = []
        # 处理文字 messages
        if isinstance(messages, str):
            content_list.append({
                "type": "input_text",
                "text": messages
            })
        else:
            # messages 是列表
            for msg in messages:
                content_list.append({
                    "type": "input_text",
                    "text": str(msg)  # 强制转换为字符串
                })

        # 处理图片
        if image_base64_list:
            for img_b64 in image_base64_list:
                content_list.append({
                    "type": "input_image",
                    "image_url": img_b64
                })
                
        

        # 最终 payload
        payload = [
            {
                "role": "user",
                "content": content_list,
            }
        ]

        # 调用 Responses API
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





# --- 主程序 ---

def main():
    """
    主对话循环。
    """
    print("--- ChatGPT 命令行对话脚本 ---")
    print("输入 '退出' 或 'exit' 结束对话。")
    print("-------------------------------")
    
    # 打印初始系统提示
    system_message = conversation_history[0]["content"]
    print(f"✅ 系统提示: {system_message}")

    while True:
        # 接收用户输入
        user_input = input("👤 您: ").strip()

        # 检查退出命令
        if user_input.lower() in ["退出", "exit"]:
            print("\n👋 感谢使用，对话结束。")
            break
        
        if not user_input:
            continue

        # 将用户消息添加到历史记录
        conversation_history.append({"role": "user", "content": user_input})

        # 获取模型响应
        response = get_chat_response(conversation_history)
        
        # 打印并保存助手响应到历史记录
        print(f"🤖 助手: {response}")
        conversation_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()