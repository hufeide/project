import json
import os
import sys
import re
import ast
import copy
import pickle
import logging
import time
import io
import base64
from typing import Dict, Any

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils import (
    KnowledgeEnhancedQA_list,
    extract_question_content,
    is_valid_base64_image,
    pkl_json,
    qa_system,  # 使用新的统一推理器
    list_available_tasks,  # 查看支持的任务
)
from utils.logger import get_logger

logger = get_logger("task_analysis")

our_subject = {"语文"}

# ================= 路径加载 =================
current_dir = os.path.dirname(os.path.abspath(__file__))

# ================= 全局资源 =================
# 使用新的统一推理器
qa_system = qa_system

def process_question(datas: Dict[str, Any], task: str) -> Dict[str, Any]:
    """
    等价于原 Flask 接口 /difficulty_jud
    使用新的统一架构
    """
    logger.info(f"Received {task} request")

    all_question = []
    all_uuid_mapping = {}

    required_fields = ['subject', 'questionMateria', 'questionStem', 'questionType', 'questionNo', 'knowledge']

    images_end_list = []
    for index, data in enumerate(datas):
        if (not data) or any(field not in data for field in required_fields):
            raise ValueError("Invalid input format in list item")

        subject = data['subject']
        if subject not in our_subject:
            raise ValueError(f"Unsupported subject: {subject}")
        level = data['abilityLevel']
        question_no = data['questionNo']
        question_type = data['questionType']
        know_md = data['knowledgemd']

        answer_type_prompt_one = answer_type_prompt.get(question_type)
        answer_type_example_one = answer_type_example.get(question_type)
        if 1 == 1:
            answer_type_example_one = ""
            know_md = "".join(pd.read_csv("/data/weidu_new/code_25/0703/dfjg_chinese_rec_v1/Template/exam_item_analysis/难易度/gradio_me/temp/merged_analysis.txt").iloc[:, 0].values)

        question_dict = extract_question_content(data['questionMateria'], data['questionStem'], data['answer'])
        material_text = question_dict['material']
        question_text = question_dict['question']
        answer_text = question_dict['answer']
        images_list = question_dict['images_pool']

        images_list_valid = [x for x in images_list if is_valid_base64_image(x)]
        if len(images_list_valid) != len(images_list):
            raise ValueError(f"Question {index + 1} has {len(images_list_valid)} valid images out of {len(images_list)} total images")
        images_list = images_list_valid
        uuid_one = data.get('uuid')
        all_uuid_mapping[index + 1] = uuid_one

        logger.info(f"Processing question {index + 1}")
        knowledge_point = data.get("knowledge")
        required_keys = {"题号", "试题分析", "答题分析"}
        question_dict_one = {
            'uuid': uuid_one,
            'question_no': question_no,
            'material': material_text,
            'question': question_text,
            'answer': answer_text,
            'question_type': question_type,
            'knowledge_point': knowledge_point,
            'level': level,
            'knowledge': know_md,
            'promote_head': answer_system,
            'promote_out': answer_type_prompt_one,
            'answer_example': answer_type_example_one,
            'task': task,  # 关键：指定任务类型
            'image_list': images_list,
            'required_keys': required_keys,
        }
        images_end_list.append(images_list)
        all_question.append(question_dict_one)

    pre_processed_data = all_question

    logger.info("LLM processing...")

    for index, item in enumerate(pre_processed_data):
        if os.path.exists(f'/data/weidu_new/code_25/0703/dfjg_chinese_rec_v1/Template/exam_item_analysis/data/{item["uuid"]}.pkl'):
            pkl_json(f'/data/weidu_new/code_25/0703/dfjg_chinese_rec_v1/Template/exam_item_analysis/data/{item["uuid"]}.pkl', f'/data/weidu_new/code_25/0703/dfjg_chinese_rec_v1/Template/exam_item_analysis/data/{item["uuid"]}.json')

    if 1 == 1:
        for index, item in enumerate(pre_processed_data):
            pkl_path = f'/data/weidu_new/code_25/0703/dfjg_chinese_rec_v1/Template/exam_item_analysis/data/{item["uuid"]}.pkl'
            if os.path.exists(pkl_path):
                all_results_m = pickle.load(open(pkl_path, 'rb'))
                if all_results_m['res1']['试题分析'] != "" and all_results_m['res2']['答题分析'] != "":
                    continue

            try:
                # 使用新的统一推理器
                all_results = qa_system.batch_inference([item], max_workers=10)
                all_results_m = all_results[0] | item

                with open(f'/data/weidu_new/code_25/0703/dfjg_chinese_rec_v1/Template/exam_item_analysis/data/{item["uuid"]}.pkl', 'wb') as f:
                    pickle.dump(all_results_m, f)
                pkl_json(f'/data/weidu_new/code_25/0703/dfjg_chinese_rec_v1/Template/exam_item_analysis/data/{item["uuid"]}.pkl', f'/data/weidu_new/code_25/0703/dfjg_chinese_rec_v1/Template/exam_item_analysis/data/{item["uuid"]}.json')
                print(index)
            except:
                continue


class knowledge_md:
    def __init__(self):
        self.knowledge_point = pd.read_excel("/data/weidu_new/code_25/0703/dfjg_chinese_rec_v1/Template/exam_item_analysis/难易度/gradio_me/参考知识.xlsx")

    def get_knowledge_point(self, knowledge_str):
        matched = self.knowledge_point[self.knowledge_point['知识代码'] == knowledge_str]
        if len(matched) == 0:
            return ""
        knowledge_point = matched['文件'].values[0]
        try:
            with open(f'/data/weidu_new/code_25/0703/dfjg_chinese_rec_v1/Template/exam_item_analysis/难易度/gradio_me/knowledge/{knowledge_point}.md', 'r', encoding='utf-8') as f:
                knowledge_md = f.read()
            return knowledge_md
        except FileNotFoundError:
            return ""

if __name__ == "__main__":
    # 测试新的统一架构
    print("=== 支持的任务类型 ===")
    print(list_available_tasks())
    
    file_path = '/data/weidu_new/code_25/0703/dfjg_chinese_rec_v1/Template/exam_item_analysis/难易度/global/ai_model_apply_py/data/260323_chinese_json.json'

    def load_and_process_dfjg(file_path):
        tasks = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                for item in row.get('list', []):
                    if item.get('is_matrial') == '是':
                        material = item.get('questionStem', '')
                        for sub in item.get('list', []):
                            tasks.append({
                                "uuid": sub.get('uuid'),
                                "context": material,
                                "question": sub.get('questionStem', ''),
                                "answer": sub.get('answer', ''),
                                "type": sub.get('type'),
                                "knowledge": sub.get('knowledge')
                            })
                    else:
                        tasks.append({
                            "uuid": item.get('uuid'),
                            "context": None,
                            "question": item.get('questionStem', ''),
                            "answer": item.get('answer', ''),
                            "type": item.get('type'),
                            "knowledge": item.get('knowledge')
                        })
        return tasks

    tasks = load_and_process_dfjg(file_path)
    df = pd.DataFrame(tasks)
    df.rename(columns={
        'context': 'questionMateria',
        'question': 'questionStem',
        'type': 'questionType',
    }, inplace=True)
    df['questionNo'] = df.index + 1
    df['abilityLevel'] = df['knowledge'].str.split('-').str[0]
    df['subject'] = '语文'
    knowledge_md_obj = knowledge_md()
    df['knowledgemd'] = df.apply(lambda x: knowledge_md_obj.get_knowledge_point(x['knowledge']), axis=1)
    df.to_excel('/data/weidu_new/code_25/0703/dfjg_chinese_rec_v1/Template/exam_item_analysis/难易度/gradio_me/data/processed_data_0323.xlsx', index=False)
    data = df.to_dict(orient='records')

    PROMPT_DIR = os.path.join(BASE_DIR, "data/prompt_file/")

    def load_prompt(file_name):
        file_path = os.path.join(PROMPT_DIR, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_name.endswith('.json'):
                content = f.read()
                lines = content.split('\n')
                json_lines = []
                for line in lines:
                    stripped = line.strip()
                    if not stripped.startswith('//'):
                        json_lines.append(line)
                    else:
                        break
                return json.loads('\n'.join(json_lines))
            else:
                return f.read()

    answer_system = load_prompt('answer_prompt_system.txt')
    answer_type_example = load_prompt('answer_example.json')
    answer_type_prompt = load_prompt('answer_type.json')
    task = "answer_analysis"
    res = process_question(data, task=task)
