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
from utils.ocr_vllm import call_vllm_ocr

from utils.logger import get_logger
from api.api import get_taskgroup_list,get_taskgroup
from utils.data_clean import get_task_from_json,knowledge_md,fill_example
from utils.image_utils import save_image_path

logger = get_logger("task_analysis")
from utils.http_request import ModelService,TaskDetailService,ModelCompareResultAdd,CleanDataUpdate,TaskDetailUpdate,ModelRecordAdd
# ================= 路径加载 =================
current_dir = os.path.dirname(os.path.abspath(__file__))

# ================= 全局资源 =================
# 使用新的统一推理器

def upload_knowledge(item: Dict[str, Any],all_results_m: Dict[str, Any]):
    """
    等价于原 Flask 接口 /knowledge_gen
    """
    data = item['data']
    model_service = ModelService()
    TaskDetail_Service = TaskDetailService()
    task_id = item.get('data', {}).get('taskId')
    record1 = CleanDataUpdate(
            taskId = data['taskId'],
            subjectId = "23",
            uuid = all_results_m['uuid'],
            material = all_results_m['material'],
            stem = all_results_m['question'],
            cleanAnswer = all_results_m['answer'],
    )
    record2 = ModelCompareResultAdd(
        taskId = task_id,
        modelName1 = '题库',
        modelName2 = 'vllm_model2',
        compareModelName = 'vllm_model3',
        compareResult = str(1 if all_results_m.get('results', {}).get('comparison_result', {}).get('human_correct') == "是" else 0),
        reason = all_results_m.get('results', {}).get('comparison_result', {}).get('reason'),
        model1Result1 = all_results_m['knowledgeCode'],
        model1Result2 = all_results_m['knowledge_name'],
        model1Result3 = all_results_m['question_type'],
        model2Result1 = all_results_m['results']['vllm_model1']['kp_code'],#暂时不使用
        model2Result2 = all_results_m['results']['vllm_model1']['kp'],#暂时不使用
        model2Result3 = all_results_m['question_type'],#暂时不使用
        effectModel = int(1),
    )
    record4_1 = ModelRecordAdd(
        taskId=task_id,
        modelName="vllm_model1", # 传入数字也可以自动转为枚举
        flow="最终结果",
        modelInput=all_results_m['prompt_info'],
        modelOutput=str(all_results_m['results']['vllm_model1']),
        outputParsedResult1 = str(all_results_m['results']['vllm_model1']['kp_code']),
        outputParsedResult2 = str(all_results_m['results']['vllm_model1']['kp']),#暂时不使用
        outputParsedResult3 = str(all_results_m['question_type'])#暂时不使用
    )
    record4_2 = ModelRecordAdd(
        taskId=task_id,
        modelName="vllm_model2", # 传入数字也可以自动转为枚举
        flow="最终结果",
        modelInput=all_results_m['prompt_info'],
        modelOutput=str(all_results_m['results']['vllm_model1']),
        outputParsedResult1 = str(all_results_m['results']['vllm_model1']['kp_code']),
        outputParsedResult2 = str(all_results_m['results']['vllm_model1']['kp']),
        outputParsedResult3 = str(all_results_m['question_type'])
    )
    record4_3 = ModelRecordAdd(
        taskId=task_id,
        modelName="vllm_model3", # 传入数字也可以自动转为枚举
        flow="最终结果",
        modelInput="",
        modelOutput=str(1 if all_results_m.get('results', {}).get('comparison_result', {}).get('correct') == "是" else 0),
        outputParsedResult1 = "",
        outputParsedResult2 = "",
        outputParsedResult3 = ""
    )
    if all_results_m.get('results', {}).get('comparison_result', {}).get('correct') == "是":
        taskStatus = 2
    else:
        taskStatus = 3
        
    record3 = TaskDetailUpdate(
        taskId = task_id,
        taskStatus = taskStatus,
    )
    TaskDetail_Service.update_clean_data(record1)
    model_service.add_compare_result(record2)
    TaskDetail_Service.update_status(record3)
    model_service.add_record(record4_1)
    model_service.add_record(record4_2)
    model_service.add_record(record4_3)


def upload_answer_correct(item: Dict[str, Any],all_results_m: Dict[str, Any]):
    """
    等价于原 Flask 接口 /knowledge_gen
    """
    data = item['data']
    model_service = ModelService()
    TaskDetail_Service = TaskDetailService()
    task_id = item.get('data', {}).get('taskId')
    record1 = CleanDataUpdate(
            taskId = data['taskId'],
            subjectId = "23",
            uuid = all_results_m['uuid'],
            material = all_results_m['material'],
            stem = all_results_m['question'],
            cleanAnswer = all_results_m['answer'],
    )
    record2 = ModelCompareResultAdd(
        taskId = task_id,
        modelName1 = '题库',
        modelName2 = 'vllm_model2',
        compareModelName = 'vllm_model3',
        compareResult = str(1 if all_results_m.get('results', {}).get('comparison_result', {}).get('human_correct') == "是" else 0),
        reason = all_results_m.get('results', {}).get('comparison_result', {}).get('reason'),
        model1Result1 = all_results_m['answer'],
        model2Result1 = all_results_m['results']['vllm_model1']['question_answer'],
        effectModel = int(1),
    )
    record4_1 = ModelRecordAdd(
        taskId=task_id,
        modelName="vllm_model1", # 传入数字也可以自动转为枚举
        flow="最终结果",
        modelInput=all_results_m['prompt_info'],
        modelOutput=str(all_results_m['results']['vllm_model1']),
        outputParsedResult1 = str(all_results_m['results']['vllm_model1']['question_answer']),
    )
    record4_2 = ModelRecordAdd(
        taskId=task_id,
        modelName="vllm_model2", # 传入数字也可以自动转为枚举
        flow="最终结果",
        modelInput=all_results_m['prompt_info'],
        modelOutput=str(all_results_m['results']['vllm_model1']),
        outputParsedResult1 = str(all_results_m['results']['vllm_model1']['question_answer']),
    )
    record4_3 = ModelRecordAdd(
        taskId=task_id,
        modelName="vllm_model3", # 传入数字也可以自动转为枚举
        flow="最终结果",
        modelInput="",
        modelOutput=str(1 if all_results_m.get('results', {}).get('comparison_result', {}).get('correct') == "是" else 0),
        outputParsedResult1 = "",
        outputParsedResult2 = "",
        outputParsedResult3 = ""
    )
    if all_results_m.get('results', {}).get('comparison_result', {}).get('correct') == "是":
        taskStatus = 2
    else:
        taskStatus = 3
        
    record3 = TaskDetailUpdate(
        taskId = task_id,
        taskStatus = taskStatus,
    )
    TaskDetail_Service.update_clean_data(record1)
    model_service.add_compare_result(record2)
    TaskDetail_Service.update_status(record3)
    model_service.add_record(record4_1)
    model_service.add_record(record4_2)
    model_service.add_record(record4_3)

def upload_knowledge_gen(item: Dict[str, Any],all_results_m: Dict[str, Any]):
    """
    等价于原 Flask 接口 /knowledge_gen
    """
    data = item['data']
    model_service = ModelService()
    TaskDetail_Service = TaskDetailService()
    task_id = item.get('data', {}).get('taskId')
    record1 = CleanDataUpdate(
            taskId = data['taskId'],
            subjectId = "23",
            uuid = all_results_m['uuid'],
            material = all_results_m['material'],
            stem = all_results_m['question'],
            cleanAnswer = all_results_m['answer'],
    )
    record2 = ModelCompareResultAdd(
        taskId = task_id,
        modelName1 = 'vllm_model1',
        modelName2 = 'vllm_model2',
        compareModelName = 'vllm_model3',
        compareResult = all_results_m['results']['comparison_result']['correct'],
        reason = all_results_m['results']['comparison_result']['reason'],
        model1Result1 = all_results_m['results']['vllm_model1']['kp_code'],
        model1Result2 = all_results_m['results']['vllm_model1']['kp'],
        model1Result3 = all_results_m['results']['vllm_model1']['question_type'],
        model2Result1 = all_results_m['results']['vllm_model2']['kp_code'],
        model2Result2 = all_results_m['results']['vllm_model2']['kp'],
        model2Result3 = all_results_m['results']['vllm_model2']['question_type'],
        effectModel = int(1),
    )
    record4_1 = ModelRecordAdd(
        taskId=task_id,
        modelName="vllm_model1", # 传入数字也可以自动转为枚举
        flow="最终结果",
        modelInput=all_results_m['prompt_info'],
        modelOutput=str(all_results_m['results']['vllm_model1']),
        outputParsedResult1 = str(all_results_m['results']['vllm_model1']['kp_code']),
        outputParsedResult2 = str(all_results_m['results']['vllm_model1']['kp']),
        outputParsedResult3 = str(all_results_m['results']['vllm_model1']['question_type'])
    )
    record4_2 = ModelRecordAdd(
        taskId=task_id,
        modelName="vllm_model2", # 传入数字也可以自动转为枚举
        flow="最终结果",
        modelInput=all_results_m['prompt_info'],
        modelOutput=str(all_results_m['results']['vllm_model2']),
        outputParsedResult1 = str(all_results_m['results']['vllm_model2']['kp_code']),
        outputParsedResult2 = str(all_results_m['results']['vllm_model2']['kp']),
        outputParsedResult3 = str(all_results_m['results']['vllm_model2']['question_type'])
    )
    record4_3 = ModelRecordAdd(
        taskId=task_id,
        modelName="vllm_model3", # 传入数字也可以自动转为枚举
        flow="最终结果",
        modelInput="",
        modelOutput=str(1 if all_results_m.get('results', {}).get('comparison_result', {}).get('correct') == "是" else 0),
        outputParsedResult1 = "",
        outputParsedResult2 = "",
        outputParsedResult3 = ""
    )
    if all_results_m.get('results', {}).get('comparison_result', {}).get('correct') == "是":
        taskStatus = 2
    else:
        taskStatus = 3
        
    record3 = TaskDetailUpdate(
        taskId = task_id,
        taskStatus = taskStatus,
    )
    TaskDetail_Service.update_clean_data(record1)
    model_service.add_compare_result(record2)
    TaskDetail_Service.update_status(record3)
    model_service.add_record(record4_1)
    model_service.add_record(record4_2)
    model_service.add_record(record4_3)


def upload_answer_gen(item: Dict[str, Any],all_results_m: Dict[str, Any]):
    """
    等价于原 Flask 接口 /knowledge_gen
    """
    data = item['data']
    model_service = ModelService()
    TaskDetail_Service = TaskDetailService()
    task_id = item.get('data', {}).get('taskId')
    record1 = CleanDataUpdate(
            taskId = data['taskId'],
            subjectId = "23",
            uuid = all_results_m['uuid'],
            material = all_results_m['material'],
            stem = all_results_m['question'],
            cleanAnswer = all_results_m['answer'],
    )
    record2 = ModelCompareResultAdd(
        taskId = task_id,
        modelName1 = 'vllm_model1',
        modelName2 = 'vllm_model2',
        compareModelName = 'vllm_model3',
        compareResult = all_results_m['results']['comparison_result']['correct'],
        reason = all_results_m['results']['comparison_result']['reason'],
        model1Result1 = all_results_m['results']['vllm_model1']['answer'],
        model2Result1 = all_results_m['results']['vllm_model2']['answer'],
        effectModel = int(1),
    )
    record4_1 = ModelRecordAdd(
        taskId=task_id,
        modelName="vllm_model1", # 传入数字也可以自动转为枚举
        flow="最终结果",
        modelInput=all_results_m['prompt_info'],
        modelOutput=str(all_results_m['results']['vllm_model1']),
        outputParsedResult1 = str(all_results_m['results']['vllm_model1']['answer']),
    )
    record4_2 = ModelRecordAdd(
        taskId=task_id,
        modelName="vllm_model2", # 传入数字也可以自动转为枚举
        flow="最终结果",
        modelInput=all_results_m['prompt_info'],
        modelOutput=str(all_results_m['results']['vllm_model2']),
        outputParsedResult1 = str(all_results_m['results']['vllm_model2']['answer']),
    )
    record4_3 = ModelRecordAdd(
        taskId=task_id,
        modelName="vllm_model3", # 传入数字也可以自动转为枚举
        flow="最终结果",
        modelInput="",
        modelOutput=str(1 if all_results_m.get('results', {}).get('comparison_result', {}).get('correct') == "是" else 0),
        outputParsedResult1 = "",
        outputParsedResult2 = "",
        outputParsedResult3 = ""
    )
    if all_results_m.get('results', {}).get('comparison_result', {}).get('correct') == "是":
        taskStatus = 2
    else:
        taskStatus = 3
        
    record3 = TaskDetailUpdate(
        taskId = task_id,
        taskStatus = taskStatus,
    )
    TaskDetail_Service.update_clean_data(record1)
    model_service.add_compare_result(record2)
    TaskDetail_Service.update_status(record3)
    model_service.add_record(record4_1)
    model_service.add_record(record4_2)
    model_service.add_record(record4_3)

def upload_analysis(item: Dict[str, Any],all_results_m: Dict[str, Any]):
    data = item['data']
    model_service = ModelService()
    TaskDetail_Service = TaskDetailService()
    task_id = item.get('data', {}).get('taskId')
    record1 = CleanDataUpdate(
            taskId = data['taskId'],
            subjectId = "23",
            uuid = all_results_m['uuid'],
            material = all_results_m['material'],
            stem = all_results_m['question'],
            cleanAnswer = all_results_m['answer'],
    )
    record2 = ModelCompareResultAdd(
        taskId = task_id,
        modelName1 = 'vllm_model1',
        modelName2 = 'vllm_model2',
        compareModelName = 'vllm_model3',
        compareResult = 1 if all_results_m['results']['comparison_result']['correct'] == "是" else 0,
        reason = all_results_m['results']['comparison_result']['reason'],
        model1Result1 = all_results_m['results']['vllm_model1']['试题分析'],
        model1Result2 = all_results_m['results']['vllm_model1']['答题分析'],
        model2Result1 = all_results_m['results']['vllm_model2']['试题分析'],
        model2Result2 = all_results_m['results']['vllm_model2']['答题分析'],
        effectModel = int(all_results_m['results']['comparison_result']['better']),
    )
    record4_1 = ModelRecordAdd(
        taskId=task_id,
        modelName="vllm_model1", # 传入数字也可以自动转为枚举
        flow="最终结果",
        modelInput=all_results_m['prompt_info'],
        modelOutput=str(all_results_m['results']['vllm_model1']),
        outputParsedResult1 = str(all_results_m['results']['vllm_model1']['试题分析']),
        outputParsedResult2 = str(all_results_m['results']['vllm_model1']['答题分析'])
    )
    record4_2 = ModelRecordAdd(
        taskId=task_id,
        modelName="vllm_model2", # 传入数字也可以自动转为枚举
        flow="最终结果",
        modelInput=all_results_m['prompt_info'],
        modelOutput=str(all_results_m['results']['vllm_model2']),
        outputParsedResult1 = str(all_results_m['results']['vllm_model2']['试题分析']),
        outputParsedResult2 = str(all_results_m['results']['vllm_model2']['答题分析'])
    )
    record4_3 = ModelRecordAdd(
        taskId=task_id,
        modelName="vllm_model3", # 传入数字也可以自动转为枚举
        flow="最终结果",
        modelInput="",
        modelOutput=str(all_results_m['results']['comparison_result']),
        outputParsedResult1 = "",
        outputParsedResult2 = ""
    )
    if all_results_m['results']['comparison_result']['correct'] == "是":
        taskStatus = 2
    else:
        taskStatus = 3
        
    record3 = TaskDetailUpdate(
        taskId = task_id,
        taskStatus = taskStatus,
    )
    TaskDetail_Service.update_clean_data(record1)
    model_service.add_compare_result(record2)
    TaskDetail_Service.update_status(record3)
    model_service.add_record(record4_1)
    model_service.add_record(record4_2)
    model_service.add_record(record4_3)

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
            if 1==2:
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


if __name__ == "__main__":
    # 设置信号处理器
    def signal_handler(signum, frame):
        logger.info(f"收到信号 {signum}，正在退出...")
        sys.exit(130)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    import argparse
    parser = argparse.ArgumentParser(description='运行任务分析')
    parser.add_argument('--task', type=str, default='answer_knowledge_gen', 
                        choices=['answer_analysis', 'answer_correct', 'answer_knowledge', 'answer_correct_gen', 'answer_knowledge_gen'],
                        help='任务类型')
    args = parser.parse_args()
    task = args.task
    
    print(f"=== 支持的任务类型 ===")
    print(list_available_tasks())
    print(f"=== 当前任务: {task} ===")
    
    # file_path = '/data/weidu_new/code_25/0703/dfjg_chinese_rec_v1/Template/exam_item_analysis/难易度/global/ai_model_apply_py/data/260323_chinese_json.json'
    tasks = []
    mode = "request"
    if mode == "local":
        tasks = get_task_from_json()
    else:
        taskgroup = get_taskgroup(subject="语文", task_type=5)['rows']
        for taskgroup_one in taskgroup:
            taskgroup_id = taskgroup_one['id']
            tasktype = taskgroup_one['taskType']
            subjectid = taskgroup_one['subjectId']
            # if taskgroup_id != 571:
            #     continue
            tasks.extend(get_taskgroup_list(taskgroup_id, tasktype,subjectid)['rows'])
    
    tasks = tasks[:50]
    
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
    
    
    # if 1==1:
    #     df = df.sample(frac=0.2)
    #     df['knowledge'] = df['knowledge'].sample(frac=1).values
    #     df['knowledge_name'] = df['knowledge_name'].sample(frac=1).values
    #     df['answer'] = df['answer'].sample(frac=1).values

    async def process_questions_with_ocr():
        datas = df.to_dict(orient='records')
        all_question = []
        required_fields = ['subject', 'questionStem', 'questionType', 'questionNo', 'knowledgeCode', 'knowledge']

        for index, data in enumerate(datas):
            uuid = data.get('uuid')
            # if uuid != "aeea787a-1bf2-402c-bc7e-31ee4f182698":
            #     continue
            if any(data.get(field) is None or data.get(field) == "" for field in required_fields):
                continue    
            subject = data['subject']
            level = data['abilityLevel']
            question_no = data['questionNo']
            question_type = data['questionType']
            know_md = data['knowledgemd']
            answer_type_prompt_one = data['answer_type_prompt_one']
            answer_type_example_one = data['answer_type_example_one']

            question_dict = extract_question_content(data['questionMaterial'], data['questionStem'], data['answer'])
            material_text = question_dict['material']
            question_text = question_dict['question']
            answer_text = question_dict['answer']
            images_list = question_dict['images_pool']

            images_list_valid = [x for x in images_list if is_valid_base64_image(x)]
            if len(images_list_valid) != len(images_list):
                raise ValueError(f"Question {index + 1} has {len(images_list_valid)} valid images out of {len(images_list)} total images")
            images_list = images_list_valid
            ocr_images = True
            IMAGE_SAVE_DIR = os.path.join(os.path.dirname(current_dir), "data", "png")
            TXT_SAVE_DIR = os.path.join(os.path.dirname(current_dir), "data", "txt")
            os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
            if len(images_list) > 0 and ocr_images:
                ocr_text = await call_vllm_ocr(images_list)
                ocr_text = [x.strip().replace("•", "").replace("*", "").replace("◆", "") if x else "" for x in ocr_text]
                os.makedirs(TXT_SAVE_DIR, exist_ok=True)
                with open(os.path.join(TXT_SAVE_DIR, f"{uuid}.txt"), "w") as f:
                    f.write("\n".join(ocr_text))
                for i, text in enumerate(ocr_text, 1):
                    text_replace = "【"+ text +"】"
                    material_text = material_text.replace(f"【图片{i}】", text_replace)
                    question_text = question_text.replace(f"【图片{i}】", text_replace)
                path_list = []
                if images_list:
                    dir_path = os.path.join(IMAGE_SAVE_DIR, uuid)
                    os.makedirs(dir_path, exist_ok=True)
                    for index, img in enumerate(images_list):
                        save_path = os.path.join(dir_path, f"{index}.png")
                        path = save_image_path(img, save_path)
                        path_list.append(path)
                images_list = []

            uuid_one = data.get('uuid')
            knowledgeCode = data.get("knowledgeCode")
            knowledge_name = data.get("knowledge")

            question_dict_one = {
                'uuid': uuid_one,
                'question_no': question_no,
                'org_material': data['questionMaterial'],
                'org_question': data['questionStem'],
                'org_answer': data['answer'],
                'material': material_text,
                'question': question_text,
                'answer': answer_text,
                'question_type': question_type,
                'knowledgeCode': knowledgeCode,
                'knowledge_name': knowledge_name,
                'level': level,
                'knowledge': know_md,
                'promote_head': answer_system,
                'promote_out': answer_type_prompt_one,
                'answer_example': answer_type_example_one,
                'task': task,  # 关键：指定任务类型
                'image_list': images_list,
                'data': data
            }

            all_question.append(question_dict_one)
        return all_question
    
    all_question = asyncio.run(process_questions_with_ocr())
    # pd.DataFrame(all_question).to_excel('/data/weidu_new/code_25/0703/dfjg_chinese_rec_v1/Template/exam_item_analysis/project/data/processed_data_0323.xlsx', index=False)
    res = process_question(all_question, task=task)
# python core/upload_api.py --task answer_analysis
# python core/upload_api.py --task answer_correct
# python core/upload_api.py --task answer_knowledge
# python core/upload_api.py --task answer_correct_gen
# python core/upload_api.py --task answer_knowledge_gen
