import json
import os
import pickle

import chardet
import pandas as pd


def pkl_json(pkl_path, json_path):
    output_path = json_path

    def convert_set_to_list(obj):
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: convert_set_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_set_to_list(v) for v in obj]
        else:
            return obj

    with open(pkl_path, 'rb') as f:
        all_results_m = pickle.load(f)
    safe_results = convert_set_to_list(all_results_m)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(safe_results, f, ensure_ascii=False, indent=4)


def read_csv_auto(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(300000))
    encoding = result['encoding'] if result['encoding'] else 'utf-8'
    print(f"[INFO] 检测到文件编码: {encoding}")
    df = pd.read_csv(file_path, encoding=encoding)
    return df


def read_md_fun(x, current_dir):
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return ""
        else:
            file_path = os.path.join(current_dir, f"knowledge/{x}.md")
            with open(file_path, 'r', encoding='utf-8') as f:
                md = f.read().replace('_x000d_', '').replace('\u3000', ' ')
                return md[15000:]
    except:
        return ""


def split_text(text, max_chunk_size=30000):
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) > max_chunk_size:
            chunks.append(current_chunk)
            current_chunk = para
        else:
            current_chunk += "\n" + para
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def read_txt_auto(file_path):
    """自动检测编码并读取TXT文件"""
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(300000))
    encoding = result['encoding'] if result['encoding'] else 'utf-8'
    print(f"[INFO] 检测到文件编码: {encoding}")

    with open(file_path, 'r', encoding=encoding) as f:
        content = f.read()

    return content
