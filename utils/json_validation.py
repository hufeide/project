import json
from typing import Any


def validate_single_json_string(json_str, required_keys):
    """校验单道题目的 JSON 字符串"""
    empty_dict = {key: "" for key in required_keys}
    empty_dict['is_valid'] = False
    try:
        data = json.loads(json_str)
        if isinstance(data, list):
            data = data[0]
        data['is_valid'] = True
        if not isinstance(data, dict):
            return empty_dict

        current_keys = set(data.keys())
        missing = required_keys - current_keys
        if missing:
            return empty_dict

        for key in required_keys:
            val = data.get(key)
            if val is None or str(val).strip() == "":
                return empty_dict

        return data

    except json.JSONDecodeError:
        return empty_dict


def is_list_of_list(data: Any) -> bool:
    if not isinstance(data, list):
        return False
    return all(isinstance(item, dict) for item in data)


def copy_list_with_zero_values(original_list):
    """
    复制列表并将所有元素中的浮点数字段设为-1
    假设每个元素是字典
    """
    if not original_list:
        return []

    def process_item_simple(item):
        return {
            key: (-1 if isinstance(value, float) else value)
            for key, value in item.items()
        }

    copied_list = []
    for item in original_list:
        new_item = item.copy()
        new_item = process_item_simple(new_item)
        copied_list.append(new_item)

    return copied_list


def deduplicate_complete_dicts(data):
    seen = set()
    result = []
    for item in data:
        dict_str = str(sorted(item.items()))
        if dict_str not in seen:
            seen.add(dict_str)
            result.append(item)
    return result


def normalize_value(value):
    """标准化数值：限制在0-1之间，保留4位小数"""
    if isinstance(value, (int, float)):
        value = float(value)
        if value > 1:
            return 1.0
        elif value < 0:
            return 0.0
        else:
            return round(value, 4)
    return value


def get_field(data):
    filtered_data = {}
    numeric_keys = [key for key, value in data.items() if isinstance(value, (int, float)) and key != "题号"]
    if len(numeric_keys) == 3:
        new_data = data.copy()
        for i, old_key in enumerate(numeric_keys, 1):
            new_data[f'参数{i}'] = new_data.pop(old_key)
        data = new_data

    temp_data = final_main(data)
    required_keys = [
        "参数1",
        "参数2",
        "参数3",
        "句法难度",
        "平均对数词频",
        "考查方式_系数",
        "考查方式",
        "考法",
        "CEFR平均等级(sigmoid)",
        "知识点_系数",
        "考法_系数",
        "说明",
    ]

    for key in required_keys:
        if key in temp_data[0]:
            filtered_data[key] = temp_data[0][key]
    return filtered_data


def review_json(knowledge_result_ori, yuwen_is_matrial=[]):
    """
    重写后的 JSON 评审函数
    返回值: (our_model1_list, our_model2_list, error_message)
    """
    our_model1_list, our_model2_list = None, None
    error = ""

    m1_raw = knowledge_result_ori[0]
    m2_raw = knowledge_result_ori[1]

    is_m1_valid = is_valid_json(m1_raw)
    is_m2_valid = is_valid_json(m2_raw)

    try:
        if is_m1_valid and is_m2_valid:
            try:
                our_model1_list = get_feild_model(knowledge_result_ori, 0)
                our_model2_list = get_feild_model(knowledge_result_ori, 1)
            except Exception:
                our_model1_list = get_feild_model(knowledge_result_ori, 0)
                our_model2_list = our_model1_list
                error = "deepseek 解析json错误"

        elif is_m1_valid:
            our_model1_list = get_feild_model(knowledge_result_ori, 0)
            our_model2_list = copy_list_with_zero_values(our_model1_list)
            error = "deepseek 解析json错误"

        elif is_m2_valid:
            our_model2_list = get_feild_model(knowledge_result_ori, 1)
            our_model1_list = copy_list_with_zero_values(our_model2_list)
            error = "chatGPT 网络链接超时" if m1_raw is None else "chatGPT 解析json错误"

        else:
            our_model1_list = get_feild_model(knowledge_result_ori, 0)
            our_model2_list = our_model1_list
            error = "双模型解析 JSON 均失败，采用保底逻辑"

    except Exception as e:
        return [], [], f"核心解析异常: {str(e)}"

    target_len = len(yuwen_is_matrial)

    if our_model1_list and our_model2_list:
        if len(our_model1_list) != len(our_model2_list):
            if len(our_model1_list) == target_len:
                our_model2_list = copy_list_with_zero_values(our_model1_list)
                error = "deepseek 模型幻觉，输出长度与输入不一致"
            elif len(our_model2_list) == target_len:
                our_model1_list = copy_list_with_zero_values(our_model2_list)
                error = "chatGPT 模型幻觉，输出长度与输入不一致"
            else:
                our_model2_list = copy_list_with_zero_values(our_model1_list)
                error = "双模型长度均异常，强制对齐"

    return our_model1_list or [], our_model2_list or [], error
