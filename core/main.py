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
from multiprocessing import Process, Queue
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
    result_dir = os.path.join(os.path.dirname(current_dir), "data","result",task)
    os.makedirs(result_dir, exist_ok=True)
    logger.info(f"Received {task} request")
    if 1 == 1:
        for index, item in enumerate(datas):
            pkl_path = f'{result_dir}/{item["uuid"]}.pkl'
            if os.path.exists(pkl_path):
                # continue
                all_results_m = pickle.load(open(pkl_path, 'rb'))
                if task in ["answer_analysis"]:
                    if all_results_m['results']['vllm_model1']['试题分析'] != "" and all_results_m['results']['vllm_model2']['答题分析'] != "":
                        continue
                if task in ["answer_correct_gen" , "answer_knowledge_gen"]:
                    if all_results_m['results']['comparison_result']['is_valid'] != "":
                        continue
                if task in ["answer_correct" , "answer_knowledge_gen"]:
                    if all_results_m['results']['vllm_model1']['is_valid'] != "":
                        continue

            try:
                # 使用新的统一推理器
                all_results = qa_system.batch_inference([item], max_workers=10)
                all_results_m = all_results[0] | item
                
                with open(f'{result_dir}/{item["uuid"]}.pkl', 'wb') as f:
                    pickle.dump(all_results_m, f)
                pkl_json(f'{result_dir}/{item["uuid"]}.pkl', f'{result_dir}/{item["uuid"]}.json')
                print(index)
            except:
                continue


class knowledge_md:
    def __init__(self):
        self.knowledge_point = pd.read_excel(os.path.join(os.path.dirname(current_dir),"data", "参考知识.xlsx"))

    def get_knowledge_point(self, knowledge_str):
        matched = self.knowledge_point[self.knowledge_point['知识代码'] == knowledge_str]
        if len(matched) == 0:
            return ""
        knowledge_point = matched['文件'].values[0]
        try:
            with open(f'{BASE_DIR}/data/knowledge/{knowledge_point}.md', 'r', encoding='utf-8') as f:
                knowledge_md = f.read()
            return knowledge_md
        except FileNotFoundError:
            return ""
def match_example(df, question_type, knowledge_code):
    def split_codes(x):
        if pd.isna(x):
            return []
        return [i.strip() for i in str(x).strip("、").split("、") if i.strip()]
    df_filtered = df[df["题型"] == question_type].copy()
    df_filtered["code_list"] = df_filtered["对应广东字典库条目"].apply(split_codes)

    matched = df_filtered[df_filtered["code_list"].apply(lambda x: knowledge_code in x)]

    if matched.empty:
        return None

    return "".join(matched["示例"])
def fill_example(row, df):
    result = match_example(df, row["questionType"], row["knowledge"])

    # 👉 fallback：没有匹配就保留原值
    return result if result is not None else row["answer_type_example_one"]
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='运行任务分析')
    parser.add_argument('--task', type=str, default='answer_analysis', 
                        choices=['answer_analysis', 'answer_correct', 'answer_knowledge', 'answer_correct_gen', 'answer_knowledge_gen'],
                        help='任务类型')
    args = parser.parse_args()
    task = args.task
    
    print(f"=== 支持的任务类型 ===")
    print(list_available_tasks())
    print(f"=== 当前任务: {task} ===")
    
    # file_path = '/data/weidu_new/code_25/0703/dfjg_chinese_rec_v1/Template/exam_item_analysis/难易度/global/ai_model_apply_py/data/260323_chinese_json.json'
    file_path = "/data/weidu_new/code_25/0703/dfjg_chinese_rec_v1/Template/exam_item_analysis/project/data/260323_chinese_json.json"
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
    knowledge_dict = pd.read_excel("/data/weidu_new/code_25/0703/dfjg_chinese_rec_v1/Template/exam_item_analysis/project/data/广东语文应试知识点.xlsx")
    knowledge_dict.columns = ['id','knowledge', 'knowledge_name']
    df = pd.DataFrame(tasks)
    df = df.merge(knowledge_dict[['knowledge', 'knowledge_name']], on='knowledge', how='left')
    
    df.rename(columns={
        'context': 'questionMateria',
        'question': 'questionStem',
        'type': 'questionType',
    }, inplace=True)
    df['questionNo'] = df.index + 1
    df['abilityLevel'] = df['knowledge'].str.split('-').str[0]
    df['subject'] = '语文'
  # df.to_excel('/data/weidu_new/code_25/0703/dfjg_chinese_rec_v1/Template/exam_item_analysis/难易度/gradio_me/data/processed_data_0323.xlsx', index=False)
    

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
    # task ="answer_analysis"# "answer_correct" # "answer_analysis"#"answer_knowledge"#"answer_correct_gen" # "answer_knowledge_gen"#
    df['task'] = task
    if task == "answer_analysis":
        answer_system = load_prompt('task_answer_analysis_sys.txt')
        answer_type_example = load_prompt('example_answer_analysis.json')
        answer_type_prompt = load_prompt('task_answer_analysis.json')
        df["answer_system_prompt_one"] = answer_system
        df["answer_type_prompt_one"] = [answer_type_prompt.get(x) for x in df["questionType"]]
        df["answer_type_example_one"] = [answer_type_example.get(x) for x in df["questionType"]]
        answer_consist = pd.read_excel(os.path.join(PROMPT_DIR, "example_answer_analysis.xlsx"))
        answer_consist["对应广东字典库条目"] = answer_consist["对应广东字典库条目"].ffill()
        df["answer_type_example_one"] = df.apply(lambda row: fill_example(row, answer_consist), axis=1)
        df.to_excel('/data/weidu_new/code_25/0703/dfjg_chinese_rec_v1/Template/exam_item_analysis/project/data/processed_data_0323.xlsx', index=False)

        ###知识
        knowledge_md_obj = knowledge_md()
        df['knowledgemd'] = df.apply(lambda x: knowledge_md_obj.get_knowledge_point(x['knowledge']), axis=1)

    if task == "answer_knowledge":
        answer_system = load_prompt('task_answer_knowledge_sys.txt')
        df["answer_system_prompt_one"] = answer_system
        df["answer_type_prompt_one"] = ""
        df["answer_type_example_one"] = ""
        
        know_md = knowledge_dict[['knowledge', 'knowledge_name']]
        know_md.columns = ['kp_code', 'kd']
        know_md.sort_values(by='kp_code', inplace=True)
        df['knowledgemd'] = json.dumps(know_md.to_dict(orient='records'), ensure_ascii=False)
    
    if task == "answer_correct":
        answer_system = load_prompt('task_answer_correct_sys.txt')
        df["answer_system_prompt_one"] = answer_system
        df["answer_type_prompt_one"] = ""
        df["answer_type_example_one"] = ""
        df['knowledgemd'] = ""

    if task == "answer_correct_gen":
        answer_system = load_prompt('task_answer_correct_gen_sys.txt')
        df["answer_system_prompt_one"] = answer_system
        df["answer_type_prompt_one"] = ""
        df["answer_type_example_one"] = ""
        df['knowledgemd'] = ""
    if task == "answer_knowledge_gen":
        answer_system = load_prompt('task_answer_knowledge_gen_sys.txt')
        df["answer_system_prompt_one"] = answer_system
        df["answer_type_prompt_one"] = ""
        df["answer_type_example_one"] = ""

        know_md = knowledge_dict[['knowledge', 'knowledge_name']]
        know_md.columns = ['kp_code', 'kd']
        know_md.sort_values(by='kp_code', inplace=True)
        df['knowledgemd'] = json.dumps(know_md.to_dict(orient='records'), ensure_ascii=False)
    
    
    # if 1==1:
    #     df = df.sample(frac=0.2)
    #     df['knowledge'] = df['knowledge'].sample(frac=1).values
    #     df['knowledge_name'] = df['knowledge_name'].sample(frac=1).values
    #     df['answer'] = df['answer'].sample(frac=1).values

    datas = df.to_dict(orient='records')
    all_question = []
    required_fields = ['subject', 'questionMateria', 'questionStem', 'questionType', 'questionNo', 'knowledge']

    images_end_list = []
    for index, data in enumerate(datas):
        subject = data['subject']
        level = data['abilityLevel']
        question_no = data['questionNo']
        question_type = data['questionType']
        know_md = data['knowledgemd']
        answer_type_prompt_one = data['answer_type_prompt_one']
        answer_type_example_one = data['answer_type_example_one']

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

        knowledge_code = data.get("knowledge")
        knowledge_name = data.get("knowledge_name")

        question_dict_one = {
            'uuid': uuid_one,
            'question_no': question_no,
            'material': material_text,
            'question': question_text,
            'answer': answer_text,
            'question_type': question_type,
            'knowledge_code': knowledge_code,
            'knowledge_name': knowledge_name,
            'level': level,
            'knowledge': know_md,
            'promote_head': answer_system,
            'promote_out': answer_type_prompt_one,
            'answer_example': answer_type_example_one,
            'task': task,  # 关键：指定任务类型
            'image_list': images_list,
        }
        images_end_list.append(images_list)
        all_question.append(question_dict_one)
    res = process_question(all_question, task=task)
# python core/main.py --task answer_analysis
# python core/main.py --task answer_correct
# python core/main.py --task answer_correct_gen
# python core/main.py --task answer_knowledge_gen
