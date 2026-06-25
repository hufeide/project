import json
from typing import Any
import re

def is_validated_equal(v1: dict, v2: dict, keys=None) -> bool:
    """
    判断两个 validated dict 是否一致：
    - 只比较指定 keys（默认 answer / kp_code / question_type）
    - 对于“同时存在”的 key，必须值完全相等
    """

    if keys is None:
        keys = ["answer", "kp_code", "question_type"]
    has_any_key = any(k in v1 or k in v2 for k in keys)
    if not has_any_key:
        return False
    for k in keys:
        if k in v1 and k in v2:
            if v1[k] != v2[k]:
                return False

    return True



import json
import re

def fix_inner_quotes(s: str) -> str:
    """
    修复 JSON 字符串内部未转义的 "
    核心：只在字符串内部做转义
    """
    result = []
    in_string = False
    escape = False

    for i, ch in enumerate(s):
        if ch == '"' and not escape:
            # 判断是不是字符串边界
            # 前面是 : 或 , 或 { → 开始字符串
            # 后面是 , 或 } → 结束字符串
            prev = s[i-1] if i > 0 else ""
            nxt = s[i+1] if i+1 < len(s) else ""

            if not in_string:
                in_string = True
                result.append(ch)
            else:
                # 判断是否应该结束字符串
                if nxt in [",", "}", "]"]:
                    in_string = False
                    result.append(ch)
                else:
                    # ❗ 这是内部引号 → 转义
                    result.append('\\"')
                    continue
        else:
            result.append(ch)

        escape = (ch == '\\' and not escape)

    return "".join(result)


def safe_json_loads(json_str):
    if not isinstance(json_str, str):
        return None

    # 1. 提取 JSON
    match = re.search(r"\{.*\}", json_str, re.DOTALL)
    if not match:
        return None
    s = match.group()

    # 2. 清理
    s = s.strip()

    try:
        return json.loads(s)
    except:
        pass

    # ===== 修复阶段 =====
    fixed = s

    # 控制字符
    fixed = re.sub(r"[\x00-\x1F]+", " ", fixed)

    # 奇怪符号
    fixed = fixed.replace("╱", "/")

    # ⭐ 核心修复：内部引号
    fixed = fix_inner_quotes(fixed)

    # 去尾逗号
    fixed = re.sub(r",\s*}", "}", fixed)

    # 截断
    last = fixed.rfind("}")
    if last != -1:
        fixed = fixed[:last+1]

    try:
        return json.loads(fixed)
    except Exception as e:
        # debug用
        # print(fixed)
        return None

def safe_json_loads_plus(json_str):
    data = safe_json_loads(json_str)
    if data is not None:
        return data

    # 最终兜底
    try:
        from json_repair import repair_json
        return json.loads(repair_json(json_str))
    except Exception:
        return None

def validate_single_json_string(json_str, required_keys):
    required_keys = set(required_keys)

    empty_dict = {key: "" for key in required_keys}
    empty_dict["is_valid"] = False

    data = safe_json_loads_plus(json_str)
    if data is None:
        return empty_dict

    # list 处理
    if isinstance(data, list):
        data = next((x for x in data if isinstance(x, dict)), None)
        if data is None:
            return empty_dict

    if not isinstance(data, dict):
        return empty_dict

    # key 标准化
    data = {str(k).strip(): v for k, v in data.items()}

    # 校验 key
    if not required_keys.issubset(data.keys()):
        return empty_dict

    # 校验值
    for k in required_keys:
        v = data[k]
        if v is None or (isinstance(v, str) and v.strip() == ""):
            return empty_dict

    data["is_valid"] = True
    return data

def is_list_of_list(data: Any) -> bool:
    if not isinstance(data, list):
        return False
    return all(isinstance(item, dict) for item in data)
