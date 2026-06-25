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
import asyncio
import signal
from typing import Dict, Any
from collections import defaultdict
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from PIL import Image
from multiprocessing import Process, Queue

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils import (
    extract_question_content,
    is_valid_base64_image,
    pkl_json,
    qa_system,  # 使用新的统一推理器
    list_available_tasks,  # 查看支持的任务

)
from utils.ocr_vllm import process_questions_with_ocr

from utils.logger import get_logger
from api.api import get_taskgroup_list,get_taskgroup
from utils.data_clean import get_task_from_json,knowledge_md,fill_example
from utils.image_utils import save_image_path
from utils.upload_fun import upload_analysis,upload_answer_gen,upload_knowledge_gen,upload_answer_correct,upload_knowledge
logger = get_logger("task_analysis")
from utils.http_request import ModelService,TaskDetailService,ModelCompareResultAdd,CleanDataUpdate,TaskDetailUpdate,ModelRecordAdd
# ================= 路径加载 =================
current_dir = os.path.dirname(os.path.abspath(__file__))

# ================= 全局资源 =================
def task_preprocess(tasks,mode="local",task_name=None):
    """
    预处理任务数据
    """

    knowledge_dict = pd.read_excel("/data/weidu_new/code_25/0703/dfjg_chinese_rec_v1/Template/exam_item_analysis/project/data/广东语文应试知识点.xlsx")
    knowledge_dict.columns = ['id','section','knowledgeCode', 'knowledge','knowledge_detail']
    df = pd.DataFrame(tasks)

    df.rename(columns={
        'context': 'questionMateria',
        'question': 'questionStem',
        'type': 'questionType',
    }, inplace=True)
    if mode == "local":
        df = df.merge(knowledge_dict[['knowledgeCode', 'knowledge']], on='knowledgeCode', how='left')
        df['questionNo'] = df.index + 1
        df['abilityLevel'] = df['knowledgeCode'].str.split('-').str[0]
        df['subject'] = '语文'
  # df.to_excel('/data/weidu_new/code_25/0703/dfjg_chinese_rec_v1/Template/exam_item_analysis/project/data/processed_data_0323.xlsx', index=False)
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
    df = df[df['taskStatus'] == 0]
    df = df[df['task'] == task]
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
        # df.to_excel('/data/weidu_new/code_25/0703/dfjg_chinese_rec_v1/Template/exam_item_analysis/project/data/processed_data_0323.xlsx', index=False)
        ###知识
        knowledge_md_obj = knowledge_md()
        df['knowledgemd'] = df.apply(lambda x: knowledge_md_obj.get_knowledge_point(x['knowledgeCode']), axis=1)

    if task == "answer_knowledge":
        answer_system = load_prompt('task_answer_knowledge_sys.txt')
        df["answer_system_prompt_one"] = answer_system
        df["answer_type_prompt_one"] = ""
        df["answer_type_example_one"] = ""
        
        know_md = knowledge_dict[['section','knowledgeCode', 'knowledge','knowledge_detail']]
        know_md.columns = ['板块','知识点代码', '知识点','知识点详情']
        know_md.sort_values(by='板块', inplace=True)
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

        know_md = knowledge_dict[['section','knowledgeCode', 'knowledge','knowledge_detail']]
        know_md.columns = ['板块','知识点代码', '知识点','知识点详情']
        know_md.sort_values(by='板块', inplace=True)
        df['knowledgemd'] = json.dumps(know_md.to_dict(orient='records'), ensure_ascii=False)
    return df 
def process_question(datas: Dict[str, Any], task: str) -> Dict[str, Any]:
    """
    等价于原 Flask 接口 /difficulty_jud
    使用新的统一架构
    """

    logger.info(f"Received {task} request")
    if 1 == 1:
        for index, item in enumerate(datas):
            result_dir = os.path.join(os.path.dirname(current_dir), "data","result",task,str(item.get('data', {}).get('taskGroupId')))
            os.makedirs(result_dir, exist_ok=True)
            pkl_path = f'{result_dir}/{item["uuid"]}.pkl'
            if os.path.exists(pkl_path):
                # continue
                all_results_m = pickle.load(open(pkl_path, 'rb'))
                if task in ["answer_analysis"]:
                    if all_results_m['results']['vllm_model1']['试题分析'] != "" and all_results_m['results']['vllm_model2']['答题分析'] != "":
                        upload_analysis(item, all_results_m)
                        continue
                
                if task in ["answer_correct_gen" , "answer_knowledge_gen"]:
                    if all_results_m['results']['comparison_result']['is_valid'] != "":
                        continue
                    if task=="answer_correct_gen":
                        upload_answer_gen(item, all_results_m)
                        continue
                    if task=="answer_knowledge_gen":
                        upload_knowledge_gen(item, all_results_m)
                        continue
                
                if task in ["answer_correct" , "answer_knowledge"]:
                    if all_results_m['results']['vllm_model1']['is_valid'] != True:
                        continue
                    if task=="answer_correct":
                        upload_answer_correct(item, all_results_m)
                        continue
                    if task=="answer_knowledge":
                        upload_knowledge(item, all_results_m)
                        continue
            if 1==1:
                try:
                    # 使用新的统一推理器
                    all_results = qa_system.batch_inference([item], max_workers=10)
                    all_results_m = all_results[0] | item
                    
                    with open(f'{result_dir}/{item["uuid"]}.pkl', 'wb') as f:
                        pickle.dump(all_results_m, f)
                    pkl_json(f'{result_dir}/{item["uuid"]}.pkl', f'{result_dir}/{item["uuid"]}.json')
                    
                    print(index)
                except KeyboardInterrupt:
                    logger.info("收到中断信号，正在退出...")
                    sys.exit(130)
                except Exception as e:
                    logger.error(f"处理项目 {item.get('uuid', 'unknown')} 时出错: {e}")
                    continue

                for index, item in enumerate(datas):
                    result_dir = os.path.join(os.path.dirname(current_dir), "data","result",task,str(item.get('data', {}).get('taskGroupId')))
                    os.makedirs(result_dir, exist_ok=True)
                    pkl_path = f'{result_dir}/{item["uuid"]}.pkl'
                    if os.path.exists(pkl_path):
                        # continue
                        all_results_m = pickle.load(open(pkl_path, 'rb'))
                        if task in ["answer_analysis"]:
                            if all_results_m['results']['vllm_model1']['试题分析'] != "" and all_results_m['results']['vllm_model2']['答题分析'] != "":
                                upload_analysis(item, all_results_m)
                                continue
                        
                        if task in ["answer_correct_gen" , "answer_knowledge_gen"]:
                            if all_results_m['results']['comparison_result']['is_valid'] != "":
                                continue
                            if task=="answer_correct_gen":
                                upload_answer_gen(item, all_results_m)
                                continue
                            if task=="answer_knowledge_gen":
                                upload_knowledge_gen(item, all_results_m)
                                continue
                        
                        if task in ["answer_correct" , "answer_knowledge"]:
                            if all_results_m['results']['vllm_model1']['is_valid'] != True:
                                continue
                            if task=="answer_correct":
                                upload_answer_correct(item, all_results_m)
                                continue
                            if task=="answer_knowledge":
                                upload_knowledge(item, all_results_m)
                                continue


if __name__ == "__main__":
    # 设置信号处理器
    def signal_handler(signum, frame):
        logger.info(f"收到信号 {signum}，正在退出...")
        sys.exit(130)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    import argparse
    parser = argparse.ArgumentParser(description='运行任务分析')
    parser.add_argument('--task', type=str, default='answer_correct', 
                        choices=['answer_analysis', 'answer_correct', 'answer_knowledge', 'answer_correct_gen', 'answer_knowledge_gen'],
                        help='任务类型')
    args = parser.parse_args()
    task = args.task
    
    print(f"=== 支持的任务类型 ===")
    print(list_available_tasks())
    print(f"=== 当前任务: {task} ===")
    
    # file_path = '/data/weidu_new/code_25/0703/dfjg_chinese_rec_v1/Template/exam_item_analysis/难易度/global/ai_model_apply_py/data/260323_chinese_json.json'
    if task =="answer_analysis":
        task_type_now = 5
    elif task == "answer_correct":
        task_type_now = 2
    elif task == "answer_knowledge":
        task_type_now = 3
    tasks = []
    mode = "request"
    if mode == "local":
        tasks = get_task_from_json()
    else:
        taskgroup = get_taskgroup(subject="语文", task_type=task_type_now)['rows']
        for taskgroup_one in taskgroup:
            taskgroup_id = taskgroup_one['id']
            tasktype = taskgroup_one['taskType']
            subjectid = taskgroup_one['subjectId']
            # if taskgroup_id not in [8780]:
            #     continue
            tasks.extend(get_taskgroup_list(taskgroup_id, tasktype,subjectid)['rows'])
    
    # tasks = tasks[:100]
    task_name_list = []
    for task_one in tasks:
        if task_type_now == 3:
            if task == 'answer_knowledge':
                if task_one['knowledgeCode'] == "" or task_one['questionType'] == "":
                    task_name_list.append('answer_knowledge_gen')
                else:
                    task_name_list.append(task)
        elif task_type_now == 2:
            if task == 'answer_correct':
                if task_one['answer'] == "" :
                    task_name_list.append('answer_correct_gen')
                else:
                    task_name_list.append(task)
        else:
            task_name_list.append(task)
    
    for i,item in enumerate(tasks):
        item['task'] = task_name_list[i]

    
    grouped_tasks = defaultdict(list)

    for i, item in enumerate(tasks):
        task_name = task_name_list[i]

        item['task'] = task_name
        grouped_tasks[task_name].append(item)

    # 转成普通 dict
    grouped_tasks = dict(grouped_tasks)

    print(grouped_tasks)
   
    df_all = pd.DataFrame()
    for k,v in grouped_tasks.items():
        df = task_preprocess(v,task_name=k)
        df_all = pd.concat([df_all, df], axis=0)

    all_question = asyncio.run(process_questions_with_ocr(df_all))
    # pd.DataFrame(all_question).to_excel('/data/weidu_new/code_25/0703/dfjg_chinese_rec_v1/Template/exam_item_analysis/project/data/processed_data_0323.xlsx', index=False)
    res = process_question(all_question, task=task)
# python core/upload_api.py --task answer_analysis
# python core/upload_api.py --task answer_correct
# python core/upload_api.py --task answer_knowledge
# python core/upload_api.py --task answer_correct_gen
# python core/upload_api.py --task answer_knowledge_gen
