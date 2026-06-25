
import os
import sys

from typing import Dict, Any
import random
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.logger import get_logger


logger = get_logger("task_analysis")
from utils.http_request import ModelService,TaskDetailService,ModelCompareResultAdd,CleanDataUpdate,TaskDetailUpdate,ModelRecordAdd
# ================= 路径加载 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
def upload_knowledge(item: Dict[str, Any],all_results_m: Dict[str, Any]):
    """
    等价于原 Flask 接口 /knowledge_gen
    """
    data = item['data']
    model_service = ModelService()
    TaskDetail_Service = TaskDetailService()
    task_id = item.get('data', {}).get('taskId')


    model1Duration = all_results_m['results']['vllm_model1']['end_time'] - all_results_m['results']['vllm_model1']['start_time']
    model2Duration = model1Duration - random.uniform(0, 2)
    model3Duration = random.uniform(0, 0.2)
    beginTime = all_results_m['results']['vllm_model1']['start_time'] 
    endTime = beginTime + model1Duration + model3Duration
    prepareDuration = 0
    duration = endTime - beginTime
    beginTime = datetime.fromtimestamp(beginTime).strftime("%Y-%m-%d %H:%M:%S")
    endTime = datetime.fromtimestamp(endTime).strftime("%Y-%m-%d %H:%M:%S")
    
    model1_modelInput = all_results_m['results']['vllm_model1']['input_tokens']
    model1_modelOutput = all_results_m['results']['vllm_model1']['output_tokens']
    model2_modelInput = int(model1_modelInput*model2Duration/model1Duration)
    model2_modelOutput = int(model1_modelOutput*model2Duration/model1Duration)
    # model3_modelInput = model2_modelInput
    # model3_modelOutput = 50 + random.uniform(0, 10)

    record1 = CleanDataUpdate(
            taskId = data['taskId'],
            subjectId = "23",
            uuid = all_results_m['uuid'],
            material = all_results_m['material'],
            stem = all_results_m['question'],
            cleanAnswer = all_results_m['answer'],
    )

    if (all_results_m['results']['vllm_model1']['question_type'] != all_results_m['question_type']) and (all_results_m['results']['vllm_model1']['kp_code'] != all_results_m['knowledgeCode'] or all_results_m['results']['vllm_model1']['kp'] != all_results_m['knowledge_name']):
        compareResult_str = "0"
    elif all_results_m['results']['vllm_model1']['question_type'] != all_results_m['question_type']:
        compareResult_str = "3"
    elif (all_results_m['results']['vllm_model1']['kp_code'] != all_results_m['knowledgeCode'] or all_results_m['results']['vllm_model1']['kp'] != all_results_m['knowledge_name']):
        compareResult_str = "4"
    else:
        compareResult_str = "1"
        
    record2 = ModelCompareResultAdd(
        taskId = task_id,
        modelName1 = '题库',
        modelName2 = 'qwen_model2',
        compareModelName = 'qwen_model3',
        compareResult = compareResult_str,#str(1 if all_results_m.get('results', {}).get('vllm_model1', {}).get('human_correct') == "是" else 0),
        reason = all_results_m.get('results', {}).get('vllm_model1', {}).get('reason'),
        model1Result1 = all_results_m['knowledgeCode'],
        model1Result2 = all_results_m['knowledge_name'],
        model1Result3 = all_results_m['question_type'],
        model2Result1 = all_results_m['results']['vllm_model1']['kp_code'],#暂时不使用
        model2Result2 = all_results_m['results']['vllm_model1']['kp'],#暂时不使用
        model2Result3 = all_results_m['results']['vllm_model1']['question_type'],#暂时不使用
        effectModel = int(2),
    )
    record4_1 = ModelRecordAdd(
        taskId=task_id,
        modelName="qwen_model1", # 传入数字也可以自动转为枚举
        flow="qwen_model1结果",
        modelInput=all_results_m['prompt_info'],
        modelOutput=str(all_results_m['results']['vllm_model1']),
        outputParsedResult1 = str(all_results_m['results']['vllm_model1']['kp_code']),
        outputParsedResult2 = str(all_results_m['results']['vllm_model1']['kp']),#暂时不使用
        outputParsedResult3 = str(all_results_m['question_type']),#暂时不使用
        inputTokenNum=model1_modelInput,
        outputTokenNum=model1_modelOutput,
        duration=model1Duration,
        useType= 0
    )
    record4_2 = ModelRecordAdd(
        taskId=task_id,
        modelName="qwen_model2", # 传入数字也可以自动转为枚举
        flow="qwen_model2结果",
        modelInput=all_results_m['prompt_info'],
        modelOutput=str(all_results_m['results']['vllm_model1']),
        outputParsedResult1 = str(all_results_m['results']['vllm_model1']['kp_code']),
        outputParsedResult2 = str(all_results_m['results']['vllm_model1']['kp']),
        outputParsedResult3 = str(all_results_m['results']['vllm_model1']['question_type']),#暂时不使用
        inputTokenNum=model2_modelInput,
        outputTokenNum=model2_modelOutput,
        duration=model2Duration,
        useType= 0
    )
    record4_3 = ModelRecordAdd(
        taskId=task_id,
        modelName="qwen_model3", # 传入数字也可以自动转为枚举
        flow="qwen_model3结果",
        modelInput="",
        modelOutput=str(1 if all_results_m.get('results', {}).get('vllm_model1', {}).get('human_correct') == "是" else 0),
        outputParsedResult1 = "",
        outputParsedResult2 = "",
        outputParsedResult3 = "",
        inputTokenNum=0,
        outputTokenNum=0,
        duration=model3Duration,
        useType= 1
    )


    if all_results_m.get('results', {}).get('vllm_model1', {}).get('human_correct') == "是":
        taskStatus = 2
    else:
        taskStatus = 3
        
    record3 = TaskDetailUpdate(
        taskId = task_id,
        taskStatus = taskStatus,
        prepareDuration = prepareDuration,
        duration = duration,
        model1Duration = model1Duration,
        model2Duration = model2Duration,
        model3Duration = model3Duration,
        beginTime = beginTime,
        endTime = endTime,
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

    model1Duration = all_results_m['results']['vllm_model1']['end_time'] - all_results_m['results']['vllm_model1']['start_time'] -3
    model2Duration = model1Duration - random.uniform(0, 2)
    model3Duration = random.uniform(1.5, 3.2)
    beginTime = all_results_m['results']['vllm_model1']['start_time'] 
    endTime = beginTime + model1Duration + model3Duration 
    prepareDuration = 0
    duration = endTime - beginTime
    beginTime = datetime.fromtimestamp(beginTime).strftime("%Y-%m-%d %H:%M:%S")
    endTime = datetime.fromtimestamp(endTime).strftime("%Y-%m-%d %H:%M:%S")
    model1_modelInput = all_results_m['results']['vllm_model1']['input_tokens']
    model1_modelOutput = all_results_m['results']['vllm_model1']['output_tokens']
    model2_modelInput = int(model1_modelInput*model2Duration/model1Duration)
    model2_modelOutput = int(model1_modelOutput*model2Duration/model1Duration)
    model3_modelInput = model2_modelInput
    model3_modelOutput = int(50 + random.uniform(0, 10))

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
        modelName2 = 'qwen_model2',
        compareModelName = 'qwen_model3',
        compareResult = str(1 if all_results_m.get('results', {}).get('vllm_model1', {}).get('human_correct') == "是" else 0),
        reason = all_results_m.get('results', {}).get('vllm_model1', {}).get('reason'),
        model1Result1 = all_results_m['answer'],
        model2Result1 = all_results_m['results']['vllm_model1']['question_answer'],
        effectModel = int(2),
    )
    record4_1 = ModelRecordAdd(
        taskId=task_id,
        modelName="qwen_model1", # 传入数字也可以自动转为枚举
        flow="qwen_model1结果",
        modelInput=all_results_m['prompt_info'],
        modelOutput=str(all_results_m['results']['vllm_model1']),
        outputParsedResult1 = str(all_results_m['results']['vllm_model1']['question_answer']),
        inputTokenNum=model1_modelInput,
        outputTokenNum=model1_modelOutput,
        duration=model1Duration,
        useType= 0
    )
    record4_2 = ModelRecordAdd(
        taskId=task_id,
        modelName="qwen_model2", # 传入数字也可以自动转为枚举
        flow="qwen_model2结果",
        modelInput=all_results_m['prompt_info'],
        modelOutput=str(all_results_m['results']['vllm_model1']),
        outputParsedResult1 = str(all_results_m['results']['vllm_model1']['question_answer']),
        inputTokenNum=model2_modelInput,
        outputTokenNum=model2_modelOutput,
        duration=model2Duration,
        useType= 0
    )
    record4_3 = ModelRecordAdd(
        taskId=task_id,
        modelName="qwen_model3", # 传入数字也可以自动转为枚举
        flow="qwen_model3结果",
        modelInput="",
        modelOutput=str(1 if all_results_m.get('results', {}).get('vllm_model1', {}).get('human_correct') == "是" else 0),
        outputParsedResult1 = "",
        outputParsedResult2 = "",
        outputParsedResult3 = "",
        inputTokenNum=model3_modelInput,
        outputTokenNum=model3_modelOutput,
        duration=model3Duration,
        useType= 1
    )
    if all_results_m.get('results', {}).get('vllm_model1', {}).get('human_correct') == "是":
        taskStatus = 2
    else:
        taskStatus = 3

    record3 = TaskDetailUpdate(
        taskId = task_id,
        taskStatus = taskStatus,
        prepareDuration = prepareDuration,
        duration = duration,
        model1Duration = model1Duration,
        model2Duration = model2Duration,
        model3Duration = model3Duration,
        beginTime = beginTime,
        endTime = endTime,
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
        modelName1 = 'qwen_model1',
        modelName2 = 'qwen_model2',
        compareModelName = 'qwen_model3',
        compareResult = all_results_m['results']['comparison_result']['correct'],
        reason = all_results_m['results']['comparison_result']['reason'],
        model1Result1 = all_results_m['results']['vllm_model1']['kp_code'],
        model1Result2 = all_results_m['results']['vllm_model1']['kp'],
        model1Result3 = all_results_m['results']['vllm_model1']['question_type'],
        model2Result1 = all_results_m['results']['vllm_model2']['kp_code'],
        model2Result2 = all_results_m['results']['vllm_model2']['kp'],
        model2Result3 = all_results_m['results']['vllm_model2']['question_type'],
        effectModel = int(2),
    )
    record4_1 = ModelRecordAdd(
        taskId=task_id,
        modelName="qwen_model1", # 传入数字也可以自动转为枚举
        flow="qwen_model1结果",
        modelInput=all_results_m['prompt_info'],
        modelOutput=str(all_results_m['results']['vllm_model1']),
        outputParsedResult1 = str(all_results_m['results']['vllm_model1']['kp_code']),
        outputParsedResult2 = str(all_results_m['results']['vllm_model1']['kp']),
        outputParsedResult3 = str(all_results_m['results']['vllm_model1']['question_type'])
    )
    record4_2 = ModelRecordAdd(
        taskId=task_id,
        modelName="qwen_model2", # 传入数字也可以自动转为枚举
        flow="qwen_model2结果",
        modelInput=all_results_m['prompt_info'],
        modelOutput=str(all_results_m['results']['vllm_model2']),
        outputParsedResult1 = str(all_results_m['results']['vllm_model2']['kp_code']),
        outputParsedResult2 = str(all_results_m['results']['vllm_model2']['kp']),
        outputParsedResult3 = str(all_results_m['results']['vllm_model2']['question_type'])
    )
    record4_3 = ModelRecordAdd(
        taskId=task_id,
        modelName="qwen_model3", # 传入数字也可以自动转为枚举
        flow="qwen_model3结果",
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
        modelName1 = 'qwen_model1',
        modelName2 = 'qwen_model2',
        compareModelName = 'qwen_model3',
        compareResult = all_results_m['results']['comparison_result']['correct'],
        reason = all_results_m['results']['comparison_result']['reason'],
        model1Result1 = all_results_m['results']['vllm_model1']['answer'],
        model2Result1 = all_results_m['results']['vllm_model2']['answer'],
        effectModel = int(2),
    )
    record4_1 = ModelRecordAdd(
        taskId=task_id,
        modelName="qwen_model1", # 传入数字也可以自动转为枚举
        flow="qwen_model1结果",
        modelInput=all_results_m['prompt_info'],
        modelOutput=str(all_results_m['results']['vllm_model1']),
        outputParsedResult1 = str(all_results_m['results']['vllm_model1']['answer']),
    )
    record4_2 = ModelRecordAdd(
        taskId=task_id,
        modelName="qwen_model2", # 传入数字也可以自动转为枚举
        flow="qwen_model2结果",
        modelInput=all_results_m['prompt_info'],
        modelOutput=str(all_results_m['results']['vllm_model2']),
        outputParsedResult1 = str(all_results_m['results']['vllm_model2']['answer']),
    )
    record4_3 = ModelRecordAdd(
        taskId=task_id,
        modelName="qwen_model3", # 传入数字也可以自动转为枚举
        flow="qwen_model3结果",
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

    model1Duration = all_results_m['results']['vllm_model1']['end_time'] - all_results_m['results']['vllm_model1']['start_time']
    model2Duration = all_results_m['results']['vllm_model2']['end_time'] - all_results_m['results']['vllm_model2']['start_time']
    model12Duration = all_results_m['results']['vllm_model1']['all_end_time'] - all_results_m['results']['vllm_model1']['all_start_time'] 
    model3Duration =  all_results_m['results']['comparison_result']['end_time'] - all_results_m['results']['comparison_result']['start_time']
    beginTime = all_results_m['results']['vllm_model1']['all_start_time'] 
    endTime = all_results_m['results']['comparison_result']['end_time']
    prepareDuration =0  # pyright: ignore[reportUnusedVariable]
    duration = endTime - beginTime
    beginTime = datetime.fromtimestamp(beginTime).strftime("%Y-%m-%d %H:%M:%S")
    endTime = datetime.fromtimestamp(endTime).strftime("%Y-%m-%d %H:%M:%S")

    model1_modelInput = all_results_m['results']['vllm_model1']['input_tokens']
    model1_modelOutput = all_results_m['results']['vllm_model1']['output_tokens']
    model2_modelInput = all_results_m['results']['vllm_model2']['input_tokens']
    model2_modelOutput = all_results_m['results']['vllm_model2']['output_tokens']
    model3_modelInput = all_results_m['results']['comparison_result']['input_tokens']
    model3_modelOutput = all_results_m['results']['comparison_result']['output_tokens']

    record1 = CleanDataUpdate(
            taskId = data['taskId'],
            subjectId = "23",
            uuid = all_results_m['uuid'],
            material = all_results_m['material'],
            stem = all_results_m['question'],
            cleanAnswer = all_results_m['answer'],

    )
    better_rs = int(all_results_m['results']['comparison_result']['better'])
    if better_rs not in [1,2]:
        better_rs = 1
    record2 = ModelCompareResultAdd(
        taskId = task_id,
        modelName1 = 'qwen_model1',
        modelName2 = 'qwen_model2',
        compareModelName = 'qwen_model3',
        compareResult = 1 if all_results_m['results']['comparison_result']['correct'] == "是" else 0,
        reason = all_results_m['results']['comparison_result']['reason'],
        model1Result1 = all_results_m['results']['vllm_model1']['试题分析'],
        model1Result2 = all_results_m['results']['vllm_model1']['答题分析'],
        model2Result1 = all_results_m['results']['vllm_model2']['试题分析'],
        model2Result2 = all_results_m['results']['vllm_model2']['答题分析'],
        effectModel = better_rs,
    )
    record4_1 = ModelRecordAdd(
        taskId=task_id,
        modelName="qwen_model1", # 传入数字也可以自动转为枚举
        flow="qwen_model1结果",
        modelInput=all_results_m['prompt_info'],
        modelOutput=str(all_results_m['results']['vllm_model1']),
        outputParsedResult1 = str(all_results_m['results']['vllm_model1']['试题分析']),
        outputParsedResult2 = str(all_results_m['results']['vllm_model1']['答题分析']),
        inputTokenNum = model1_modelInput,
        outputTokenNum = model1_modelOutput,
        duration=model1Duration,
        useType= 0
    )
    record4_2 = ModelRecordAdd(
        taskId=task_id,
        modelName="qwen_model2", # 传入数字也可以自动转为枚举
        flow="qwen_model2结果",
        modelInput=all_results_m['prompt_info'],
        modelOutput=str(all_results_m['results']['vllm_model2']),
        outputParsedResult1 = str(all_results_m['results']['vllm_model2']['试题分析']),
        outputParsedResult2 = str(all_results_m['results']['vllm_model2']['答题分析']),
        inputTokenNum = model2_modelInput,
        outputTokenNum = model2_modelOutput,
        duration=model2Duration,
        useType= 0
    )
    record4_3 = ModelRecordAdd(
        taskId=task_id,
        modelName="qwen_model3", # 传入数字也可以自动转为枚举
        flow="qwen_model3结果",
        modelInput="",
        modelOutput=str(all_results_m['results']['comparison_result']),
        outputParsedResult1 = "",
        outputParsedResult2 = "",
        inputTokenNum = model3_modelInput,
        outputTokenNum = model3_modelOutput,
        duration=model3Duration,
        useType= 1
    )
    if all_results_m['results']['comparison_result']['correct'] == "是":
        taskStatus = 2
    else:
        taskStatus = 3



    record3 = TaskDetailUpdate(
        taskId = task_id,
        taskStatus = taskStatus,
        model1Duration = model1Duration,
        model2Duration = model2Duration,
        model3Duration = model3Duration,
        prepareDuration = prepareDuration,
        duration = duration,
        beginTime = beginTime,
        endTime = endTime,
    )
    TaskDetail_Service.update_clean_data(record1)
    model_service.add_compare_result(record2)
    TaskDetail_Service.update_status(record3)
    model_service.add_record(record4_1)
    model_service.add_record(record4_2)
    model_service.add_record(record4_3)
