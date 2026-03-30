import os
import requests


def get_taskgroup_list():
    """
    获取任务组列表
    
    接口路径: GET /api/taskgroup/list
    Mock地址: http://192.168.1.187:3000/mock/263/api/taskgroup/list
    
    Returns:
        dict: 返回接口响应数据，包含 total, rows, code, msg 等字段
    """
    url = "http://192.168.1.210:8070/api/task/detail/list?taskGroupId=7&pageSize=50000"
    
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
# taskgroup_list = get_taskgroup_list()
# print(taskgroup_list['rows'][0].keys())
