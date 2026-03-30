import json
from typing import Any


def validate_single_json_string(json_str, required_keys):
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
