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
from typing import List
from utils.logger import get_logger
from api.qiansheng_api import get_taskgroup_list,get_taskgroup,PromptService,TaskGroupUpdate,TaskGroupService
from utils.data_clean import get_task_from_json,knowledge_md,fill_example
from utils.image_utils import save_image_path
from utils.upload_fun import upload_analysis,upload_answer_gen,upload_knowledge_gen,upload_answer_correct,upload_knowledge
logger = get_logger("task_analysis")
# from utils.http_request import PromptService,TaskGroupUpdate,TaskGroupService
# ================= 路径加载 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
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
# ================= 全局资源 =================
knowledge_dict = pd.read_excel("/data/weidu_new/code_25/0703/dfjg_chinese_rec_v1/Template/exam_item_analysis/project/data/广东语文应试知识点.xlsx")
knowledge_dict.columns = ['id','section','knowledgeCode', 'knowledge','knowledge_detail']
def task_preprocess(tasks,mode="local",task_name=None):
    """
    预处理任务数据
    """
    for task in tasks:
        if task["task_name"] =='answer_analysis':
            task['knowledgeCode'] =  task.get('verifyKnowledgeCode') if task.get('verifyKnowledgeCode') else task.get('knowledgeCode', '').replace(" ", "")
            task['knowledge'] = task.get('verifyKnowledge') if task.get('verifyKnowledge') else task.get('knowledge', '').replace(" ", "")
            task['questionType'] = task.get('verifyQuestionType') if task.get('verifyQuestionType') else task.get('questionType', '').replace(" ", "")
            task['answer'] = task.get('verifyAnswer') if task.get('verifyAnswer') else task.get('answer', '').replace(" ", "")
    df = pd.DataFrame(tasks)
    
    df.rename(columns={
        'context': 'questionMateria',
        'question': 'questionStem',
        'type': 'questionType',
    }, inplace=True)

    group_mask = (
    df.groupby('taskGroupId')['taskStatus']
        .transform(lambda x: x.isin([2, 3]).all())
    )
    task_group_obj = TaskGroupService()
    success_group_ids = df.loc[group_mask,'taskGroupId'].unique()
    for group_id in success_group_ids:
        
        record_group = TaskGroupUpdate(
            taskGroupId = int(group_id),
            groupStatus = 2,
        )
        response = task_group_obj.update_status(record_group)
        if response.code!=200:
            logger.error(f"更新任务组状态失败，任务组ID：{group_id}")

    inprocess_group_ids = df.loc[~group_mask,'taskGroupId'].unique()
    for group_id in inprocess_group_ids:
        record_group_inprocess = TaskGroupUpdate(
            taskGroupId = int(group_id),
            groupStatus = 1,
        )
        response = task_group_obj.update_status(record_group_inprocess)
        if response.code!=200:
            logger.error(f"更新任务组状态失败，任务组ID：{group_id}")



    df = df[df['taskStatus'] == 0] #0初始化、1进行中、2校验通过待抽检、3校验不通过待审核、4已完成5已同步6失败
    df['task'] = task_name
    prompt_obj = PromptService()
    query = {
            "subject": tasks[0]['subject'],
            "area": tasks[0]['area'],
            "series": tasks[0]['series'],
            "studySection": tasks[0]['studySection'],
            "taskType": tasks[0]['taskType'],
            # "knowledgeCode": tasks[0]['knowledgeCode'], # 知识点
            # "questionType": tasks[0]['questionType'], # 题型
            "modelName": "Qwen",
            "pageSize": 2000,
    }
    response = prompt_obj.get_list(query)
    if response.code==200:
        response_data = response.rows
    response_data_df = pd.DataFrame(response_data)
    response_data_df = response_data_df.drop_duplicates(subset=['questionType'])
    
    # df = df[df['task'] == task]
    if task_name == "answer_analysis":
        answer_system = load_prompt('task_answer_analysis_sys.txt')
        #answer_system =  response_data_df['systemPromptContent'][0]#
        answer_type_example = load_prompt('example_answer_analysis.json')
        answer_type_prompt = response_data_df.set_index('questionType')['taskPromptContent'].to_dict()#load_prompt('task_answer_analysis.json')
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

    if task_name == "answer_knowledge":
        answer_system = load_prompt('task_answer_knowledge_sys.txt')
        #answer_system =  response_data_df['systemPromptContent'][0]
        df["answer_system_prompt_one"] = answer_system
        df["answer_type_prompt_one"] = ""
        df["answer_type_example_one"] = ""
        
        know_md = knowledge_dict[['section','knowledgeCode', 'knowledge','knowledge_detail']].copy()
        know_md.columns = ['板块','知识点代码', '知识点','知识点详情']
        know_md.sort_values(by='板块', inplace=True)
        df['knowledgemd'] = json.dumps(know_md.to_dict(orient='records'), ensure_ascii=False)
    
    if task_name == "answer_correct":
        answer_system = load_prompt('task_answer_correct_sys.txt')
        #answer_system =  response_data_df['systemPromptContent'][0]
        df["answer_system_prompt_one"] = answer_system
        df["answer_type_prompt_one"] = ""
        df["answer_type_example_one"] = ""
        df['knowledgemd'] = ""

    if task_name == "answer_correct_gen":
        answer_system = load_prompt('task_answer_correct_gen_sys.txt')
        # answer_system =  response_data_df['systemPromptContent'][0]
        df["answer_system_prompt_one"] = answer_system
        df["answer_type_prompt_one"] = ""
        df["answer_type_example_one"] = ""
        df['knowledgemd'] = ""
    if task_name == "answer_knowledge_gen":
        answer_system = load_prompt('task_answer_knowledge_gen_sys.txt')
        # answer_system =  response_data_df['systemPromptContent'][0]
        df["answer_system_prompt_one"] = answer_system
        df["answer_type_prompt_one"] = ""
        df["answer_type_example_one"] = ""

        know_md = knowledge_dict[['section','knowledgeCode', 'knowledge','knowledge_detail']]
        know_md.columns = ['板块','知识点代码', '知识点','知识点详情']
        know_md.sort_values(by='板块', inplace=True)
        df['knowledgemd'] = json.dumps(know_md.to_dict(orient='records'), ensure_ascii=False)
    return df 
    
def process_question(datas: Dict[str, Any]) -> Dict[str, Any]:
    """
    等价于原 Flask 接口 /difficulty_jud
    使用新的统一架构
    """

    if 1 == 1:
        for index, item in enumerate(datas):
            task = item.get('task', "")
            result_dir = os.path.join(os.path.dirname(current_dir), "data","result",task,str(item.get('data', {}).get('taskGroupId')))
            os.makedirs(result_dir, exist_ok=True)
            pkl_path = f'{result_dir}/{item["uuid"]}.pkl'
            if os.path.exists(pkl_path):
                continue
            # if os.path.exists(pkl_path):
            #     # continue
            #     all_results_m = pickle.load(open(pkl_path, 'rb'))
            #     if task in ["answer_analysis"]:
            #         if all_results_m['results']['vllm_model1']['试题分析'] != "" and all_results_m['results']['vllm_model2']['答题分析'] != "":
            #             upload_analysis(item, all_results_m)
            #             continue
                
            #     if task in ["answer_correct_gen" , "answer_knowledge_gen"]:
            #         if all_results_m['results']['comparison_result']['is_valid'] != "":
            #             continue
            #         if task=="answer_correct_gen":
            #             upload_answer_gen(item, all_results_m)
            #             continue
            #         if task=="answer_knowledge_gen":
            #             upload_knowledge_gen(item, all_results_m)
            #             continue
                
            #     if task in ["answer_correct" , "answer_knowledge"]:
            #         if all_results_m['results']['vllm_model1']['is_valid'] != True:
            #             continue
            #         if task=="answer_correct":
            #             upload_answer_correct(item, all_results_m)
            #             continue
            #         if task=="answer_knowledge":
            #             upload_knowledge(item, all_results_m)
            #             continue
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

        # for index, item in enumerate(datas):
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


def fetch_tasks_periodically(
    subject: str = "语文",
    poll_interval: int = 60,
    batch_size: int = 1,
    max_pending_time: int = 300,
    task_types: List[str] = None,
    task_name: str = None,
):
    """
    定时拉取任务并批量处理
    
    Args:
        subject: 学科类型
        poll_interval: 轮询间隔（秒）
        batch_size: 达到此数量时立即处理
        max_pending_time: 最大等待时间（秒），超过此时间即使未达到 batch_size 也处理
        task_types: 要拉取的任务类型列表
    
    Returns:
        None
    """
    if task_types is None:
        task_types = ['answer_analysis', 'answer_correct', 'answer_knowledge']#[task_name]#['answer_analysis', 'answer_correct', 'answer_knowledge']
    
    TASK_TYPE_MAP = {
        'answer_analysis': 5,
        'answer_correct': 2,
        'answer_knowledge': 3,
    }
    
    from collections import defaultdict
    
    pending_tasks = []
    pending_task_ids = set()
    last_process_time = time.time()
    knowledge_dict = None
    
    try:
        knowledge_path = "/data/weidu_new/code_25/0703/dfjg_chinese_rec_v1/Template/exam_item_analysis/project/data/广东语文应试知识点.xlsx"
        if os.path.exists(knowledge_path):
            knowledge_dict = pd.read_excel(knowledge_path)
            knowledge_dict.columns = ['id', 'section', 'knowledgeCode', 'knowledge', 'knowledge_detail']
            logger.info("知识点字典加载成功")
    except Exception as e:
        logger.error(f"加载知识点字典失败: {e}")
    
    def fetch_new_tasks():
        """拉取新任务"""
        new_tasks = []
        for task_name, task_type in TASK_TYPE_MAP.items():
            if task_name not in task_types:
                continue
            try:
                if subject == "语文":
                    subjectId = "23"
                taskgroup_0 = get_taskgroup(subjectId=subjectId, task_type=task_type,status = 0,page_size=100).rows
                taskgroup_1 = get_taskgroup(subjectId=subjectId, task_type=task_type,status = 1,page_size=100).rows
                taskgroup = taskgroup_0 + taskgroup_1
                taskgroup = [x.to_dict() for x in taskgroup]
                for taskgroup_one in taskgroup:
                    taskgroup_id = taskgroup_one['id']
                    tasktype = taskgroup_one['taskType']
                    subjectid = taskgroup_one['subjectId']
                    tasks = get_taskgroup_list(taskgroup_id, tasktype, subjectid)
                    tasks = [x.to_dict() for x in tasks]
                    for task in tasks:
                        task['task_name'] = task_name
                        task_id = task.get('id') or task.get('taskId')
                        if task_id and task_id not in pending_task_ids:
                            new_tasks.append(task)
                            pending_task_ids.add(task_id)
            except Exception as e:
                logger.error(f"拉取任务失败: {e}")
        return new_tasks
    
    def process_batch():
        """处理当前批次的所有任务"""
        nonlocal pending_tasks, last_process_time
        
        if not pending_tasks:
            return
        
        logger.info(f"开始处理批次任务，共 {len(pending_tasks)} 条")
        
        grouped_tasks = defaultdict(list)
        task_name_map = {}
        
        for item in pending_tasks:
            task_name = item.get('task_name', 'answer_analysis')
            task_name_map[id(item)] = task_name
            grouped_tasks[task_name].append(item)
        
        df_all = pd.DataFrame()
        
        for task_name, tasks in grouped_tasks.items():
            try:
                df = task_preprocess(tasks, mode="request", task_name=task_name)
                if df is not None and len(df) > 0:
                    df_all = pd.concat([df_all, df], axis=0)
            except Exception as e:
                logger.error(f"预处理任务失败: {e}")
        
        if len(df_all) > 0:
            try:
                logger.info(f"开始处理 {len(df_all)} 条数据")
                all_question = asyncio.run(process_questions_with_ocr(df_all))
                if all_question:
                    for task_name in grouped_tasks.keys():
                        res = process_question(all_question)
                        logger.info(f"任务类型 {task_name} 处理完成，结果: {res}")
            except Exception as e:
                logger.error(f"处理任务失败: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        pending_tasks = []
        pending_task_ids.clear()
        last_process_time = time.time()
    
    logger.info(f"启动定时拉取任务，轮询间隔: {poll_interval}秒，批次大小: {batch_size}，最大等待时间: {max_pending_time}秒")
    
    while True:
        try:
            new_tasks = fetch_new_tasks()
            if new_tasks:
                pending_tasks.extend(new_tasks)
                logger.info(f"拉取到 {len(new_tasks)} 个新任务，当前待处理: {len(pending_tasks)}")
            
            current_time = time.time()
            time_elapsed = current_time - last_process_time
            
            should_process = (
                len(pending_tasks) >= batch_size or 
                time_elapsed >= max_pending_time
            )
            
            if should_process and pending_tasks:
                process_batch()
            
            for _ in range(min(poll_interval, 10)):
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("收到中断信号，正在处理剩余任务...")
            process_batch()
            break
        except Exception as e:
            logger.error(f"定时拉取出错: {e}")
            time.sleep(5)


if __name__ == "__main__":
    # 设置信号处理器
    def signal_handler(signum, frame):
        logger.info(f"收到信号 {signum}，正在退出...")
        sys.exit(130)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    import argparse
    parser = argparse.ArgumentParser(description='运行任务分析')
    parser.add_argument('--mode', type=str, default='periodic',
                        choices=['single', 'periodic'],
                        help='运行模式: single(单次执行) 或 periodic(定时拉取任务)')
    parser.add_argument('--task', type=str, default='answer_analysis', 
                        choices=['answer_analysis', 'answer_correct', 'answer_knowledge', 'answer_correct_gen', 'answer_knowledge_gen'],
                        help='任务类型 (单次模式用)')
    parser.add_argument('--subject', type=str, default='语文',
                        help='学科类型')
    parser.add_argument('--poll-interval', type=int, default=60,
                        help='轮询间隔（秒），默认60秒')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='批次大小，达到此数量立即处理，默认10')
    parser.add_argument('--max-pending-time', type=int, default=300,
                        help='最大等待时间（秒），超过此时间即使未达到批次大小也处理，默认300秒')
    args = parser.parse_args()
    
    if args.mode == 'periodic':
        print("=" * 50)
        print("启动定时拉取任务模式")
        print(f"学科: {args.subject}")
        print(f"轮询间隔: {args.poll_interval} 秒")
        print(f"批次大小: {args.batch_size}")
        print(f"最大等待时间: {args.max_pending_time} 秒")
        print("=" * 50)
        
        fetch_tasks_periodically(
            subject=args.subject,
            poll_interval=args.poll_interval,
            batch_size=args.batch_size,
            max_pending_time=args.max_pending_time,
            task_name=args.task
        )
    else:
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
            # taskgroup = get_taskgroup(subject="语文", task_type=task_type_now)#['rows']
            # for taskgroup_one in taskgroup:
            #     taskgroup_id = taskgroup_one['id']
            #     tasktype = taskgroup_one['taskType']
            #     subjectid = taskgroup_one['subjectId']
            #     # if taskgroup_id not in [8780]:
            #     #     continue
            # tasks.extend(get_taskgroup_list(taskgroup_id, tasktype,subjectid))#['rows'])
            subjectId = "23"
            taskgroup_0 = get_taskgroup(subjectId=subjectId, task_type=task_type_now,status = 0,page_size=100).rows
            taskgroup_1 = get_taskgroup(subjectId=subjectId, task_type=task_type_now,status = 1,page_size=100).rows
            taskgroup = taskgroup_0 + taskgroup_1
            taskgroup = [x.to_dict() for x in taskgroup]
            tasks = []
            for taskgroup_one in taskgroup:
                taskgroup_id = taskgroup_one['id']
                tasktype = taskgroup_one['taskType']
                subjectid = taskgroup_one['subjectId']
                tasks_one = get_taskgroup_list(taskgroup_id, tasktype, subjectid)
                tasks_one = [x.to_dict() for x in tasks_one]
                tasks.extend(tasks_one)
    
    # tasks = tasks[:2]
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
        item['task_name'] = task_name_list[i]

    
    grouped_tasks = defaultdict(list)

    for i, item in enumerate(tasks):
        task_name = task_name_list[i]

        item['task_name'] = task_name
        grouped_tasks[task_name].append(item)

    # 转成普通 dict
    grouped_tasks = dict(grouped_tasks)

    print(grouped_tasks)
   
    df_all = pd.DataFrame()
    for k,v in grouped_tasks.items():
        df = task_preprocess(v,task_name=k)
        df_all = pd.concat([df_all, df], axis=0)

    all_question = asyncio.run(process_questions_with_ocr(df_all))
    res = process_question(all_question)
# ps -ef | grep task_api_prompt.py
# python core/upload_api.py --task answer_analysis
# python core/upload_api.py --task answer_correct
# python core/upload_api.py --task answer_knowledge
# python core/upload_api.py --task answer_correct_gen
# python core/upload_api.py --task answer_knowledge_gen
