from http_request import *
from utils import upload_to_minio_math_folder
from bs4 import BeautifulSoup
import base64
import os 
import time 
from start_up import clean_text,clean_text_answer,save_base64_as_png
from llm_domain import process_three_tasks_in_one_chat
#from judegment_isdrawimg import our_model_judegment_drawimg,png2svg
#from GPT_domain import our_model_drawimg
from utils import upload_to_minio_math_folder
import re
from llm_domain import process_three_tasks_in_one_chat
import json 

save_base_path = '/mnt/data/weidu/our_DFJG/temp_image'
timestamp = time.strftime("%Y%m%d")
new_folder_path = os.path.join(save_base_path, timestamp)
os.makedirs(new_folder_path, exist_ok=True)
task_group_service = TaskGroupService()
query_params = TaskGroupQuery(subjectId=SubjectIdEnum.MATH,
                            taskType=TaskTypeEnum.ANSWER_CHECK,
                            area=AreaEnum.GUANG_DONG,
                            #questionType= "解答题",
                            series=SeriesEnum.YING_SHI)
task_res = task_group_service.get_list(query=query_params)
print(task_res.rows)
group_info = task_res.rows[0]
# for row in task_res.rows:
#     if row.id == 564:
#         group_info = row

print(f"group_info: \n{group_info}")

task_detail_service = TaskDetailService()
res = None
index = 1969
total = 1970  # 初始设为 1 进入循环
task_detail_query = TaskDetailQuery(
        taskGroupId=group_info.id,
        #taskGroupId=997,
        subjectId=SubjectIdEnum.MATH,
        taskType=TaskTypeEnum.PREPARE_TASK,
        #questionType="填空题",
        subTaskType = SubTaskTypeEnum.ANSWER_CHECK
    )



def clean_and_parse(value):
    # 1. 去掉 Markdown 的代码块标记
    if isinstance(value, str):
        value = re.sub(r'```json\s*|\s*```', '', value).strip()
    try:
        # 2. 进行解析（将转义字符串转为字典、列表等）
        return json.loads(value)
    except:
        return value
        
while index < total:
    # 每次调用都会自动处理：[内存取数据] 或 [跨页发起请求]
    #stem, res, total = task_detail_service.get_item_by_list(group_info, index, last_query_res=res)
    # 每次调用都会自动处理：[内存取数据] 或 [跨页发起请求]
    stem, res, total = task_detail_service.get_item_by_list(group_info, index, last_query_res=res,
                                                                task_detail_query=task_detail_query)
    total = int(total)  # 确保 total 是整数
    #total =1000 if total > 1000 else total
    if stem:
        index += 1
        print(f"进度: {index }/{total} - 处理任务: {stem.taskId}")
        image_full_path = ''
        imgs_list = []
        if stem.questionMaterial  is not None :
            meterial = stem.questionMaterial
            soup = BeautifulSoup(meterial, 'html.parser')
            imgs = soup.find_all('img', class_='dscimg')
            src_list = [img['src'] for img in imgs if img.has_attr('src')]
            if len(src_list)>0:
                    for i, img_str in enumerate(src_list):
                        image_full_path = save_base64_as_png(img_str, f"{stem.uuid}_tigan_image_{i}.png")
                        imgs_list.append(image_full_path)
                        upload_to_minio_math_folder(image_full_path)
            stem_ = stem.questionStem
            #判断需要需不需要处理图片
            soup = BeautifulSoup(stem_, 'html.parser')
            imgs = soup.find_all('img', class_='dscimg')
            src_list = [img['src'] for img in imgs if img.has_attr('src')]
            if len(src_list)>0:
                for i, img_str in enumerate(src_list):
                    image_full_path = save_base64_as_png(img_str, f"{stem.uuid}_image_{i}.png")
                    imgs_list.append(image_full_path)
                    upload_to_minio_math_folder(image_full_path)
            stem_answer = stem.answer
            stem_uuid = stem.uuid
            stem_knowledge = stem.knowledgeList if "knowledgeList" in stem.__dict__ else [stem.knowledge]  
            # 处理数据
            result = clean_text(meterial)+clean_text(stem_)
            result_answer = clean_text_answer(stem_answer)
            
        else: 
            stem_ = stem.questionStem
            #判断需要需不需要处理图片
            soup = BeautifulSoup(stem_, 'html.parser')
            imgs = soup.find_all('img', class_='dscimg')
            src_list = [img['src'] for img in imgs if img.has_attr('src')]
            if len(src_list)>0:
                for i, img_str in enumerate(src_list):
                    image_full_path = save_base64_as_png(img_str, f"{stem.uuid}_image_{i}.png")
                    imgs_list.append(image_full_path)
                    upload_to_minio_math_folder(image_full_path)
            stem_answer = stem.answer
            stem_uuid = stem.uuid
            stem_knowledge = stem.knowledgeList if "knowledgeList" in stem.__dict__ else [stem.knowledge] 
            # 处理数据
            result = clean_text(stem_)
            result_answer = clean_text_answer(stem_answer)
        # new_folder_path1 = '/data/weidu_new/code_25/0703/our_drawmath_img/our_image_draw/backup/work/svg/output.svg'
        # new_folder_path2 = '/data/weidu_new/code_25/0703/our_drawmath_img/our_image_draw/backup/work/svg'
        # task_config ={"save_img_dir":imgs_list,"uuid":stem_uuid,"name":stem_uuid,"execution_work_dir":new_folder_path1,"target_svg_path":new_folder_path2}
        # """判断是否需要画图"""
        log_path = os.path.join(new_folder_path, f"{stem_uuid}_log-答案.txt")
        if image_full_path and os.path.exists(image_full_path):
            path_to_write = image_full_path
        else:
            path_to_write = ""
        with open(log_path, "a", encoding="utf-8") as f:
                f.write("uuid: " + stem_uuid + "\n")
                f.write("\n")
                f.write("question: " + result + "\n")
                f.write("\n")
                f.write("answer: " + result_answer + "\n") 
                f.write("\n")
                f.write("题干配图: " + image_full_path + "\n")  
                f.write("\n") 
        
        step_result = process_three_tasks_in_one_chat(result,result_answer,image_full_path,log_path,stem) 
        print("answer",step_result)   
        with open(log_path, "a", encoding="utf-8") as f:
                f.write("模型结果: " + str(step_result) + "\n")
                f.write("\n")  
        try:
            subTask = ModelCompareResultAdd(
                    taskId=stem.taskId,
                    subTaskId=stem.subTaskId, # 传入数字也可以自动转为枚举
                    model1Result1=step_result['model1'],
                    modelName1 = "题库",
                    model2Result1=step_result['model2'],
                    modelName2 = "gemini3",
                    effectModel = step_result['better'],
                    compareResult = step_result['is_save'],
                    reason= step_result['reason']
                )
            ModelService().add_compare_result(data=subTask)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("上传内容: " + str(subTask) + "\n")
            #ModelService.add_compare_result(data=subTask)
            if step_result['is_save'] ==1:
                SubTask1 = TaskDetailUpdate(
                        taskId=stem.taskId, # 传入数字也可以自动转为枚举
                        taskStatus=2,
                        taskStep=1
                    )
                TaskDetailService().update_sub_status(SubTask1)
            else:
                SubTask1 = TaskDetailUpdate(
                        taskId=stem.taskId, # 传入数字也可以自动转为枚举
                        taskStatus=3,
                        taskStep=1
                    )
                TaskDetailService().update_sub_status(SubTask1)
        except Exception as e:
             print(f"Error occurred while adding compare result: {e}")
         
    

    
    