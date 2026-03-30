import re

import pandas as pd
from bs4 import BeautifulSoup

from .image_utils import extract_image_from_html


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


def universal_question_extractor(cleaned_text):
    """
    通用题目解析方法：按文本结构提取内容
    避开【图片n】这种占位符，只对真正的业务模块标识进行分割
    """
    extracted_content = []

    split_pattern = r'(【(?!图片).*?】)'

    if not re.search(split_pattern, cleaned_text):
        blocks = [("【题干/答案】", cleaned_text)]
    else:
        content_blocks = re.split(split_pattern, cleaned_text)

        blocks = []
        if content_blocks[0].strip():
            blocks.append(("【题干/答案】", content_blocks[0].strip()))

        for i in range(1, len(content_blocks), 2):
            if i + 1 < len(content_blocks):
                tag = content_blocks[i].strip()
                content = content_blocks[i + 1].strip()
                if tag and content:
                    blocks.append((tag, content))

    for tag, content in blocks:
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if lines:
            cleaned_content = '\n'.join(lines)
            block_result = f"{tag}\n{cleaned_content}"
            extracted_content.append(block_result)

    return extracted_content if extracted_content else ["未提取到有效题目内容"]


def write_results_to_file(extracted_content, output_file_path):
    """将提取结果写入txt文件"""
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for i, content in enumerate(extracted_content):
                if i > 0:
                    f.write('\n\n')
                f.write(content)
        print(f"结果已成功写入文件: {output_file_path}")
    except Exception as e:
        print(f"写入文件失败: {str(e)}")


def clean_text(text):
    """清理文本中的多余空格和特殊字符"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:()\-]', '', text)
    return text.strip()
