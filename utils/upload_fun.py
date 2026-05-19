
import os
import sys

from typing import Dict, Any



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
        compareResult = str(1 if all_results_m.get('results', {}).get('vllm_model1', {}).get('human_correct') == "是" else 0),
        reason = all_results_m.get('results', {}).get('vllm_model1', {}).get('reason'),
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
        modelOutput=str(1 if all_results_m.get('results', {}).get('vllm_model1', {}).get('human_correct') == "是" else 0),
        outputParsedResult1 = "",
        outputParsedResult2 = "",
        outputParsedResult3 = ""
    )
    if all_results_m.get('results', {}).get('vllm_model1', {}).get('human_correct') == "是":
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
        compareResult = str(1 if all_results_m.get('results', {}).get('vllm_model1', {}).get('human_correct') == "是" else 0),
        reason = all_results_m.get('results', {}).get('vllm_model1', {}).get('reason'),
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
        modelOutput=str(1 if all_results_m.get('results', {}).get('vllm_model1', {}).get('human_correct') == "是" else 0),
        outputParsedResult1 = "",
        outputParsedResult2 = "",
        outputParsedResult3 = ""
    )
    if all_results_m.get('results', {}).get('vllm_model1', {}).get('human_correct') == "是":
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
