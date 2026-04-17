import os
import requests


def get_taskgroup_list(task_group_id, task_type,subjectid):
    """
    获取任务组列表
    
    接口路径: GET /api/taskgroup/list
    Mock地址: http://192.168.1.187:3000/mock/263/api/taskgroup/list
    
    Returns:
        dict: 返回接口响应数据，包含 total, rows, code, msg 等字段
    """
    url = f"http://192.168.1.210:8070/api/task/detail/list?taskGroupId={task_group_id}&subjectId={subjectid}&taskType={task_type}&pageSize=5000000&pageNum=1"
    
    no_proxy_backup = os.environ.get('NO_PROXY', '')
    os.environ['NO_PROXY'] = '192.168.1.210,192.168.1.187,localhost,127.0.0.1'
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {
            "error": str(e),
            "code": -1,
            "msg": "请求失败"
        }
    finally:
        os.environ['NO_PROXY'] = no_proxy_backup
import pandas as pd

def get_taskgroup(subject, task_type, area=None, series=None, question_type=None, status=None, page_num=None, page_size=None):
    """
    获取任务组列表
    
    接口路径: GET /api/taskgroup/list
    Mock地址: http://192.168.1.187:3000/mock/263/api/taskgroup/list
    
    Args:
        subject (str): 学科（必须）
        task_type (int): 任务类型（必须）1前置处理 5试题分析 6答题分析 7难易度 8评分量表
        area (str, optional): 区域
        series (str, optional): 系列
        question_type (str, optional): 题型
        status (int, optional): 任务组状态 0未开始 1进行中 2已完成
        page_num (int, optional): 页码
        page_size (int, optional): 页大小
    
    Returns:
        dict: 返回接口响应数据，包含 total, rows, code, msg 等字段
    """
    url = "http://192.168.1.210:8070/api/taskgroup/list"
    
    params = {
        "subject": subject,
        "taskType": task_type
    }
    
    if area is not None:
        params["area"] = area
    if series is not None:
        params["series"] = series
    if question_type is not None:
        params["questionType"] = question_type
    if status is not None:
        params["status"] = status
    if page_num is not None:
        params["pageNum"] = page_num
    if page_size is not None:
        params["pageSize"] = page_size
    
    no_proxy_backup = os.environ.get('NO_PROXY', '')
    os.environ['NO_PROXY'] = '192.168.1.210,192.168.1.187,localhost,127.0.0.1'
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {
            "error": str(e),
            "code": -1,
            "msg": "请求失败"
        }
    finally:
        os.environ['NO_PROXY'] = no_proxy_backup



# taskgroup['createTime']