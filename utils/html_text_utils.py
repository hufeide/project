import re

import pandas as pd
from bs4 import BeautifulSoup

from .image_utils import extract_image_from_html
import json
import os
import pickle


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


def clean_html_text(html_content, start_num=1):
    """清理HTML内容：去除标签、保留着重号并插入图片占位符"""
    soup = BeautifulSoup(html_content, 'html.parser')

    imgs = soup.find_all('img', class_='dscimg')
    src_list = [extract_image_from_html(x) for x in imgs]

    current_num = start_num
    for i, img in enumerate(imgs):
        img_placeholder = f"【图片{current_num}】"
        img.replace_with(img_placeholder)
        current_num += 1

    dot_spans = soup.find_all('span', class_='dot')
    for span in dot_spans:
        dot_text = span.get_text()
        span.replace_with(f"〖{dot_text}〗")

    for br in soup.find_all("br"):
        br.replace_with("\n")

    for p in soup.find_all("p"):
        p.insert_after("\n")

    raw_text = soup.get_text()

    cleaned_text = re.sub(r'\r\n', '\n', raw_text)
    cleaned_text = re.sub(r'\r', '\n', cleaned_text)
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)

    lines = []
    for line in cleaned_text.split('\n'):
        stripped_line = line.strip()
        if stripped_line:
            lines.append(stripped_line)
        elif line:
            lines.append("")

    cleaned_text = '\n'.join(lines)
    cleaned_text = cleaned_text.strip()

    cleaned_text = cleaned_text.replace('\xa0', ' ').replace('\u2000', ' ').replace('\u3000', '    ').replace('_x000d_', ' ').replace('\n', ' ')

    return cleaned_text, src_list, current_num


def is_empty_text(x) -> bool:
    if pd.isna(x):
        return True
    if isinstance(x, str):
        return x.strip() == ""
    return False


def extract_question_content(material_html, question_full_html, answer_html):
    """
    分别存放：材料、题干、选项(不拆开)、答案
    严格执行全局图片编号
    """
    all_images = []

    if not is_empty_text(material_html):
        material_text, m_imgs, next_num = clean_html_text(material_html, start_num=1)
        all_images.extend(m_imgs)
    else:
        material_text = ""
        next_num = 1

    q_full_text, q_imgs, next_num = clean_html_text(question_full_html, start_num=next_num)
    all_images.extend(q_imgs)

    answer_text, a_imgs, _ = clean_html_text(answer_html, start_num=next_num)
    all_images.extend(a_imgs)

    return {
        "material": material_text,
        "question": q_full_text,
        "answer": answer_text,
        "images_pool": all_images
    }



