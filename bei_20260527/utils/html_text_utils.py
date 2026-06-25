import re

import pandas as pd
from bs4 import BeautifulSoup

from .image_utils import extract_image_from_html
import json
import os
import pickle
import unicodedata

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
    if not html_content or html_content is None or pd.isna(html_content):
        return "", [], start_num

    # 1. 预处理：先干掉 HTML 里的 \r\n，防止它们被识别为文本节点
    html_content = html_content.replace('\r\n', '').replace('\r', '').replace('\n', '')
    
    soup = BeautifulSoup(html_content, 'html.parser')

    # 2. 处理图片
    imgs = soup.find_all('img', class_='dscimg')
    src_list = [extract_image_from_html(x) for x in imgs] # 假设已定义
    current_num = start_num
    for img in imgs:
        img.replace_with(f"【图片{current_num}】")
        current_num += 1

    # 3. 处理着重号
    for span in soup.find_all('span', class_='dot'):
        span.replace_with(f"<dot>{span.get_text()}</dot>")

    # 4. 处理下划线（填空位）- 先处理下划线标签，再处理内容中的下划线
    for u_tag in soup.find_all('u'):
        u_content = u_tag.get_text()
        u_tag.replace_with(f"<u>{u_content}</u>")


    # 5. 处理波浪线
    for w_tag in soup.find_all('w'):
        w_content = w_tag.get_text()
        if w_content:
            u_tag.replace_with(f"<w>{w_content}</w>")



    # 6. 关键改进：处理块级元素的换行，但保护内联元素
    # 我们只在 <p> 和 <br> 处显式换行
    for br in soup.find_all("br"):
        br.replace_with("\n")
    
    for p in soup.find_all("p"):
        # 在段落结束处加换行，但先检查它是否已经有换行了
        p.append("\n") 

    # 7. 终极清洗
    # 使用 NFKC 标准化处理 \xa0 等
    raw_text = unicodedata.normalize('NFKC', soup.get_text())
    
    # 替换 Excel 特殊字符
    raw_text = raw_text.replace('_x000d_', '') 

    # 处理下划线内容（________）
    raw_text = re.sub(r'_+', '____', raw_text)

    # 核心：将"多个换行"压缩，但将"单字间的换行"直接删掉
    # 这里我们采用一种折中方案：先处理掉那些夹在汉字/数字中间的单个换行
    cleaned_text = re.sub(r'(?<=[\u4e00-\u9fa5\d\w])\n(?=[\u4e00-\u9fa5\d\w])', '', raw_text)
    
    # 压缩过多的空行
    cleaned_text = re.sub(r'\n{2,}', '\n', cleaned_text)
    
    return cleaned_text.strip(), src_list, current_num

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



