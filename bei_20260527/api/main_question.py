# ================================
# 试题向量库管理 主流程
# ================================
from http_request import QuestionVectorQuery, QuestionVectorItem, QuestionVectorService,\
    SubjectIdEnum
from http_milvus import QuestionVectorMilvusService

from pydantic import BaseModel, Field
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

collection_name_ch = "question_vector_chinese"
milvus_service_ch = QuestionVectorMilvusService(collection_name_ch)

def is_blank(check_str):
    if check_str is None:
        return True
    if check_str.strip() == "":
        return True
    return False

def is_not_blank(check_str):
    return not is_blank(check_str)

def _format_data(data: QuestionVectorItem):
    new_data = {}
    material = data.material
    stem = data.stem
    if is_not_blank(material):
        stem += f"\n{material}"
    data.questionVectorText = stem
    # print(f"question text: {data.questionVectorText}")
    new_data["uuid"] = data.uuid
    new_data["area"] = data.area
    new_data["series"] = data.series
    new_data["studySection"] = data.studySection
    new_data["subject"] = data.subject
    new_data["subjectId"] = 23
    new_data["knowledgeCode"] = data.knowledgeCode
    new_data["knowledge"] = data.knowledge
    new_data["questionVectorText"] = data.questionVectorText
    return new_data

def main_chinese_insert():
    collection_name = "question_vector_chinese"
    milvus_service = QuestionVectorMilvusService(collection_name)

    question_vector_service = QuestionVectorService()
    question_vector_query = QuestionVectorQuery(subjectId=SubjectIdEnum.CHINESE)

    res = None
    index = 0
    total = 1  # 初始设为 1 进入循环
    wait_update_lst = []
    while index < total:
        item, res, total = \
            question_vector_service.get_item_by_list(question_vector_query=question_vector_query,
                                                     last_query_res=res,
                                                     index=index,
                                                     page_size=100)
        index += 1
        print(f"insert index: {index}/{total} uuid: {item.uuid}")
        item_dict = _format_data(item)
        milvus_service.add_data(item_dict)
        wait_update_lst.append(item.uuid)
        # break
    update_res = question_vector_service.update_vectorized_uuid_list(wait_update_lst)
    # print(f"update_res: {update_res}")
    print("def main done...")

def main_chinese_query():
    collection_name = "question_vector_chinese"
    milvus_service = QuestionVectorMilvusService(collection_name)
    uuid = "07820800-dd1d-4b43-bfe6-485578705876"
    subject_id = 23
    milvus_service.query_data(uuid, subject_id)

app = FastAPI(title="vector", version="1.0")

@app.get("/queryVector")
async def query_vector(uuid: str,
                       subjectId: int):
    query_data_lst = milvus_service_ch.query_data(uuid=uuid, subject_id=subjectId)
    return {
        "code": 200,
        "data": query_data_lst[0]
    }

@app.get("/ping")
async def ping():
    print(f"get ping request.")
    return {
        "title": "MATH-MAIN",
        "ping":"pong"
    }

def main_uvicorn():
    print(f"start main uvicorn.")
    uvicorn.run(
        "main_question:app",
        host = "0.0.0.0",
        port = 11088,
        reload = False
    )

if __name__ == '__main__':

    main_chinese_insert()

    # main_chinese_query()

    # main_uvicorn()

    print(f"__main__ done...")

