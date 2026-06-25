import time

import requests
from typing import List, Dict, Any, Optional
from pymilvus import connections, Collection, utility, FieldSchema, DataType, CollectionSchema
import numpy as np


def calc_str_similarity(s1, s2, metric_type="COSINE"):
    v1 = get_bge_embeddings_item(s1)
    v2 = get_bge_embeddings_item(s2)
    # print(f"v1 type: {type(v1)} \n{v1}")
    # print(f"v2 type: {type(v2)} \n{v2}")
    return calc_milvus_similarity(v1, v2, metric_type)


def calc_milvus_similarity(v1: List[float], v2: List[float], metric_type="COSINE"):
    """
        本地模拟 Milvus 的相似度计算逻辑
        :param v1: 向量 A (list 或 np.array)
        :param v2: 向量 B (list 或 np.array)
        :param metric_type: "L2", "IP", "COSINE"
    """
    v1 = np.array(v1, dtype=np.float32)
    v2 = np.array(v2, dtype=np.float32)
    if metric_type == "L2":
        # 欧式距离：Milvus 返回的是未开根号的平方距离 (Squared L2)
        # 注意：Milvus 搜索返回通常是平方距离，如果你需要标准的欧式距离请开方
        return np.sum(np.square(v1 - v2))
    elif metric_type == "IP":
        # 内积：直接求和
        return np.dot(v1, v2)
    elif metric_type == "COSINE":
        # 余弦相似度：点积除以模长乘积
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return np.dot(v1, v2) / (norm_v1 * norm_v2)
    else:
        raise ValueError("Unsupported metric type")


"""milvus基础服务 """
class MilvusBaseService:
    """通用的 Milvus 基础服务类"""
    def __init__(self, host, port, collection_name, metric_type="COSINE"):
        if collection_name is None or collection_name == "":
            raise Exception("collection_name is None")
        self.collection_name = collection_name
        self.metric_type=metric_type
        self._connect(host, port)
        self.collection = self._get_collection()

    def _drop(self):
        """
        安全删除指定的 Collection
        :param collection_name: 集合名称
        """
        try:
            # 1. 检查是否存在
            if utility.has_collection(self.collection_name):
                # 2. 实例化 Collection 对象进行删除（也可以直接用 utility.drop_collection）
                col = Collection(self.collection_name)
                col.drop()
                print(f"成功删除 Collection: {self.collection_name}")
            else:
                print(f"删除失败: Collection '{self.collection_name}' 不存在")
        except Exception as e:
            print(f"删除过程中发生异常: {e}")

    def _connect(self, host, port):
        if not connections.has_connection("default"):
            connections.connect(alias="default", host=host, port=port)

    def _get_collection(self):
        if not utility.has_collection(self.collection_name):
            # 基类不负责创建，由具体子类定义 Schema 逻辑
            # raise ValueError(f"Collection {self.collection_name} 不存在，请先初始化。")
            return None
        col = Collection(self.collection_name)
        col.load()
        return col

    def insert(self, data: List[Dict[str, Any]]):
        res = self.collection.insert(data)
        return res.insert_count

    def search(self, vectors: List[List[float]], anns_field: str, param: Dict, limit: int, expr: str = None, output_fields: List[str] = None):

        return self.collection.search(
            data=vectors,
            anns_field=anns_field,
            param=param,
            limit=limit,
            expr=expr,
            output_fields=output_fields
        )

    def query(self, expr: str, output_fields: List[str] = None, limit: int = 10, offset: int = 0) -> List[
        Dict[str, Any]]:
        """
        根据非向量字段进行结构化查询 (Scalar Query)

        :param expr: 查询表达式，例如 "age > 25 and status == 1" 或 "id in [1, 2, 3]"
        :param output_fields: 需要返回的标量字段列表
        :param limit: 返回的最大记录数 (标量查询的 limit)
        :param offset: 偏移量，用于分页
        """
        # 1. 调用 Milvus 的 query 方法
        raw_results = self.collection.query(
            expr=expr,
            output_fields=output_fields,
            limit=limit,
            offset=offset
        )

        # 2. 结构化解析返回值
        # collection.query() 返回的已经是形如 [{"id": 1, "field1": "val"}, ...] 的列表
        formatted_results = []
        for entity in raw_results:
            # 如果指定了 output_fields，则过滤或确保字段存在；若未指定，Milvus 默认会返回主键和 output_fields 命中的字段
            if output_fields:
                data = {field: entity.get(field) for field in output_fields}
            else:
                data = entity.copy()

            # 补充：非向量查询没有 hit.distance / score，但主键 ID 已经在 entity 中了
            # 如果你的业务强依赖 "id" 键名，可以强制转换或保留
            # if "id" not in data and self.collection.schema.primary_field.name in entity:
            #     data["id"] = entity[self.collection.schema.primary_field.name]

            formatted_results.append(data)
        return formatted_results

    def calc_distance(self, vec1, vec2):
        self.collection(
            vectors_left=vec1,
            vectors_right=vec2,
            params={"metric_type": "COSINE"}  # 支持 "L2", "IP", "COSINE"
        )
        return None

"""英语试题服务 """
class EnglishQuestionService( MilvusBaseService):
    """英语题库特有业务服务"""
    def __init__(self, collection_name, host='172.16.50.167', port='19530', vector_dim=1024):
        # 固定该 Service 对应的 Collection 名字
        super().__init__(host, port, collection_name=collection_name)
        self.vector_dim = vector_dim
        if self.collection is None:
            self.collection = self._create_collection()

    def _init_collection(self):
        """内部方法：检查集合是否存在，不存在则创建，存在则加载"""
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            self.collection.load()
            print(f"✅ 集合 '{self.collection_name}' 已存在，加载完成。")
        else:
            self.collection = self._create_collection()
            print(f"✅ 集合 '{self.collection_name}' 创建并加载完成。")

    def _create_collection(self):
        """内部方法：定义 Schema 并创建集合及索引"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="uuid", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=128, nullable=True),
            FieldSchema(name="question_type", dtype=DataType.VARCHAR, max_length=128, nullable=True),
            FieldSchema(name="area", dtype=DataType.VARCHAR, max_length=128, nullable=True),
            FieldSchema(name="knowledge", dtype=DataType.VARCHAR, max_length=512, nullable=True),
            FieldSchema(name="target", dtype=DataType.VARCHAR, max_length=512, nullable=True),
            FieldSchema(name="plate", dtype=DataType.VARCHAR, max_length=128, nullable=True),
            FieldSchema(name="test_way", dtype=DataType.VARCHAR, max_length=128, nullable=True),
            FieldSchema(name="question_no", dtype=DataType.VARCHAR, max_length=128, nullable=True),
            FieldSchema(name="question_stem", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=65535, nullable=True),
            # 向量字段
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim)
        ]

        schema = CollectionSchema(fields=fields, description="题库向量集合", enable_dynamic_field=True)
        collection = Collection(name=self.collection_name, schema=schema)

        # 默认创建索引
        index_params = {
            "metric_type": self.metric_type,  # 如果你的模型使用余弦相似度，请改为 "COSINE"
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 64}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        collection.load()
        return collection

    def search_by_knowledge(self,
                            vector: List[float],
                            knowledge_point: str=None,
                            top_k: int = 5) -> List[Dict]:
        """
        特定业务方法：仅在某个知识点范围内搜索相似题
        """
        expr = None
        if knowledge_point is not None:
            expr = f'knowledge == "{knowledge_point}"'
        results = self.search(
            vectors=[vector],
            anns_field="embedding",
            param={
                "metric_type": self.metric_type,
                "params": {"ef": 64}
            },
            limit=top_k,
            expr=expr,
            output_fields=["uuid", "question_no", "question_stem", "answer", "knowledge"]
        )
        return self._format_results(results[0])

    def find_similar_reading_comprehension(self, vector: List[float], top_k: int = 3):
        """
        特定业务方法：专门搜索“阅读理解”类型的题目
        """
        expr = 'question_type == "阅读理解"'
        results = self.search(
            vectors=[vector],
            anns_field="embedding",
            param={
                "metric_type": self.metric_type,
                "params": {"ef": 64}
            },
            limit=top_k,
            expr=expr,
            output_fields=["question_stem", "parent_id"]
        )
        return self._format_results(results[0])

    @staticmethod
    def _format_results(milvus_hits) -> List[Dict]:
        """将 Milvus 原始对象转换为易读的 List[Dict]"""
        formatted = []
        for hit in milvus_hits:
            data = hit.entity.to_dict()
            data['distance'] = hit.distance
            data['id'] = hit.id
            formatted.append(data)
        return formatted

    def add_data(self, data):
        wait_data = []
        if data is None or data.get("question_stem") is None:
            print("data is None")
            return
        vector_item = get_bge_embeddings([data.get("question_stem")])[0]
        data["embedding"] = vector_item
        wait_data.append(data)
        self.insert(wait_data)
        print(f"add success, uuid:{data.get('uuid')}")

"""英语完型填空试题服务 """
class EnglishClozeQuestionService(MilvusBaseService):
    def __init__(self, collection_name, host='172.16.50.167', port='19530', vector_dim=1024):
        if collection_name is None:
            raise Exception("collection_name is None")
        # 固定该 Service 对应的 Collection 名字
        super().__init__(host, port, collection_name=collection_name)
        self.vector_dim = vector_dim
        if self.collection is None:
            self.collection = self._create_collection()

    def _init_collection(self):
        """内部方法：检查集合是否存在，不存在则创建，存在则加载"""
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            self.collection.load()
            print(f"✅ 集合 '{self.collection_name}' 已存在，加载完成。")
        else:
            self.collection = self._create_collection()
            print(f"✅ 集合 '{self.collection_name}' 创建并加载完成。")

    def _create_collection(self):
        """内部方法：定义 Schema 并创建集合及索引"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="uuid", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=128, nullable=True),
            FieldSchema(name="question_type", dtype=DataType.VARCHAR, max_length=128, nullable=True),
            FieldSchema(name="area", dtype=DataType.VARCHAR, max_length=128, nullable=True),
            FieldSchema(name="knowledge", dtype=DataType.VARCHAR, max_length=512, nullable=True),
            FieldSchema(name="target", dtype=DataType.VARCHAR, max_length=512, nullable=True),
            FieldSchema(name="plate", dtype=DataType.VARCHAR, max_length=128, nullable=True),
            FieldSchema(name="test_way", dtype=DataType.VARCHAR, max_length=128, nullable=True),
            FieldSchema(name="question_no", dtype=DataType.VARCHAR, max_length=128, nullable=True),
            FieldSchema(name="question_material", dtype=DataType.VARCHAR, max_length=65535, nullable=True),
            FieldSchema(name="question_stem", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=65535, nullable=True),
            # 向量字段
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim)
        ]

        schema = CollectionSchema(fields=fields, description="题库向量集合", enable_dynamic_field=True)
        collection = Collection(name=self.collection_name, schema=schema)

        # 默认创建索引
        index_params = {
            # "metric_type": "L2",  # 如果你的模型使用余弦相似度，请改为 "COSINE"
            "metric_type": self.metric_type,  # 如果你的模型使用余弦相似度，请改为 "COSINE"
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 64}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        collection.load()
        return collection

    def search_by_knowledge(self,
                            vector: List[float],
                            knowledge_point: str=None,
                            top_k: int = 5) -> List[Dict]:
        """
        特定业务方法：仅在某个知识点范围内搜索相似题
        """
        expr = None
        if knowledge_point is not None:
            expr = f'knowledge == "{knowledge_point}"'
        results = self.search(
            vectors=[vector],
            anns_field="embedding",
            param={
                "metric_type": self.metric_type,
                "params": {"ef": 64}
            },
            limit=top_k,
            expr=expr,
            output_fields=["uuid", "question_no", "question_stem", "answer", "knowledge"]
        )
        return self._format_results(results[0])

    def find_similar_reading_comprehension(self, vector: List[float], top_k: int = 3):
        """
        特定业务方法：专门搜索“阅读理解”类型的题目
        """
        expr = 'question_type == "阅读理解"'
        results = self.search(
            vectors=[vector],
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"ef": 64}},
            limit=top_k,
            expr=expr,
            output_fields=["question_stem", "parent_id"]
        )
        return self._format_results(results[0])

    @staticmethod
    def _format_results(milvus_hits) -> List[Dict]:
        """将 Milvus 原始对象转换为易读的 List[Dict]"""
        formatted = []
        for hit in milvus_hits:
            data = hit.entity.to_dict()
            data['distance'] = hit.distance
            data['id'] = hit.id
            formatted.append(data)
        return formatted

    def add_data(self, data):
        wait_data = []
        if data is None or data.get("question_stem") is None:
            print("data is None")
            return
        vector_item = get_bge_embeddings([data.get("question_stem")])[0]
        data["embedding"] = vector_item
        wait_data.append(data)
        self.insert(wait_data)
        print(f"add success, uuid:{data.get('uuid')}")

"""试题查重服务 """
class CompareQuestionService(MilvusBaseService):
    def __init__(self,
                 collection_name,
                 host='172.16.50.167',
                 port='19530',
                 need_del=False,
                 vector_dim=1024):
        # 固定该 Service 对应的 Collection 名字
        super().__init__(host, port, collection_name=collection_name)
        self.vector_dim = vector_dim
        self.need_del = need_del
        if self.collection is None:
            self.collection = self._create_collection()
        else:
            if self.need_del:
                self._init_collection()

    def _init_collection_bak(self):
        """内部方法：检查集合是否存在，不存在则创建，存在则加载"""
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            self.collection.load()
            print(f"✅ 集合 '{self.collection_name}' 已存在，加载完成。")
        else:
            self.collection = self._create_collection()
            print(f"✅ 集合 '{self.collection_name}' 创建并加载完成。")

    def _init_collection(self):
        """内部方法：检查集合是否存在，不存在则创建，存在则加载"""
        if utility.has_collection(self.collection_name):
            if self.need_del:
                self._drop()
                print(f"✅ 集合 '{self.collection_name}' 已存在，已删除完成。")
                time.sleep(0.5)
            else:
                self.collection = Collection(self.collection_name)
                self.collection.load()
                print(f"✅ 集合 '{self.collection_name}' 已存在，加载完成。")
        self.collection = self._create_collection()
        print(f"✅ 集合 '{self.collection_name}' 创建并加载完成。")

    def _create_collection(self):
        """内部方法：定义 Schema 并创建集合及索引"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="uuid", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="question_uuid", dtype=DataType.VARCHAR, max_length=128, nullable=True),
            FieldSchema(name="question_stem", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="only_question_stem", dtype=DataType.VARCHAR, max_length=65535),
            # 向量字段
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim)
        ]

        schema = CollectionSchema(fields=fields, description="题库向量集合", enable_dynamic_field=True)
        collection = Collection(name=self.collection_name, schema=schema)

        # 默认创建索引
        index_params = {
            # "metric_type": "L2",  # 如果你的模型使用余弦相似度，请改为 "COSINE"
            "metric_type": self.metric_type,  # 如果你的模型使用余弦相似度，请改为 "COSINE"
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 64}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        collection.load()
        return collection

    def search_by_data(self,
                        data_str: str,
                        top_k: int = 5) -> List[Dict]:
        """
        特定业务方法：仅在某个知识点范围内搜索相似题
        """
        vector = get_bge_embeddings_item(data_str)
        results = self.search(
            vectors=[vector],
            anns_field="embedding",
            param={
                "metric_type": self.metric_type,
                "params": {
                    "ef": 64
                }
            },
            limit=top_k,
            expr=None,
            output_fields=["uuid", "question_uuid", "question_stem", "only_question_stem"]
        )
        return self._format_results(results[0])


    @staticmethod
    def _format_results(milvus_hits) -> List[Dict]:
        """将 Milvus 原始对象转换为易读的 List[Dict]"""
        formatted = []
        for hit in milvus_hits:
            data = hit.entity.to_dict()
            data['distance'] = hit.distance
            data['id'] = hit.id
            formatted.append(data)
        return formatted

    def add_data(self, data):
        wait_data = []
        if data is None or data.get("question_stem") is None:
            print("data is None")
            return
        # print(f"question_stem: {data.get('question_stem')}")
        vector_item = get_bge_embeddings([data.get("question_stem")])[0]
        data["embedding"] = vector_item
        wait_data.append(data)
        self.insert(wait_data)
        # print(f"add success, uuid:{data.get('uuid')}")

"""试题样例服务 """
class QuestionVectorMilvusService(MilvusBaseService):
    def __init__(self, collection_name, host='172.16.50.167', port='19530', vector_dim=1024):
        if collection_name is None:
            raise Exception("collection_name is None")
        # 固定该 Service 对应的 Collection 名字
        super().__init__(host, port, collection_name=collection_name)
        self.vector_dim = vector_dim
        if self.collection is None:
            self.collection = self._create_collection()

    def _init_collection(self):
        """内部方法：检查集合是否存在，不存在则创建，存在则加载"""
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            self.collection.load()
            print(f"✅ 集合 '{self.collection_name}' 已存在，加载完成。")
        else:
            self.collection = self._create_collection()
            print(f"✅ 集合 '{self.collection_name}' 创建并加载完成。")

    def _create_collection(self):
        """内部方法：定义 Schema 并创建集合及索引"""
        fields = [
            FieldSchema(name="uuid", dtype=DataType.VARCHAR, max_length=128, is_primary=True),
            FieldSchema(name="area", dtype=DataType.VARCHAR, max_length=128, nullable=True),
            FieldSchema(name="series", dtype=DataType.VARCHAR, max_length=128, nullable=True),
            FieldSchema(name="studySection", dtype=DataType.VARCHAR, max_length=128, nullable=True),
            FieldSchema(name="subject", dtype=DataType.VARCHAR, max_length=128, nullable=True),
            FieldSchema(name="subjectId", dtype=DataType.INT64, nullable=True),
            FieldSchema(name="questionType", dtype=DataType.VARCHAR, max_length=128, nullable=True),
            FieldSchema(name="knowledgeCode", dtype=DataType.VARCHAR, max_length=128, nullable=True),
            FieldSchema(name="knowledge", dtype=DataType.VARCHAR, max_length=128, nullable=True),
            FieldSchema(name="questionVectorText", dtype=DataType.VARCHAR, max_length=65535, nullable=True),
            # 材料+题干 的向量字段
            FieldSchema(name="questionVector", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim)
        ]

        schema = CollectionSchema(fields=fields, description="题库向量集合", enable_dynamic_field=True)
        collection = Collection(name=self.collection_name, schema=schema)

        # 默认创建索引
        index_params = {
            "metric_type": self.metric_type,  # 如果你的模型使用余弦相似度，请改为 "COSINE"
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 64}
        }
        collection.create_index(field_name="questionVector", index_params=index_params)
        collection.load()
        return collection

    def add_data(self, data) -> bool:
        wait_data = []
        if data is None or data.get("questionVectorText") is None:
            print("question_vector_text is None")
            return False
        vector_item = get_bge_embeddings([data.get("questionVectorText")])[0]
        data["questionVector"] = vector_item
        wait_data.append(data)
        return self.insert(wait_data)

    def query_data(self, uuid, subject_id):
        query_expr = f"uuid=='{uuid}' AND subjectId=={subject_id}"
        query_res_lst =self.query(expr=query_expr,
                   output_fields=["uuid", "area", "subject","series", "studySection", "subjectId", "questionType", "questionVectorText", "questionVector"])
        return query_res_lst

def query_question_analysis(question_stem: str, knowledgeCode: str = None,questionType: str = None,area: str = None, subject: str = None,series: str = None,studySection: str = None, top_k: int = 1): 
    vector_item = get_bge_embeddings([question_stem])[0]
    
    expr_parts = []
    if questionType:
        expr_parts.append(f'questionType == "{questionType}"')
    if area:
        expr_parts.append(f'area == "{area}"')
    if subject:
        expr_parts.append(f'subject == "{subject}"')
    if knowledgeCode:
        expr_parts.append(f'knowledgeCode == "{knowledgeCode}"')
    if series:
        expr_parts.append(f'series == "{series}"')
    if studySection:
        expr_parts.append(f'studySection == "{studySection}"')
    
    expr = " AND ".join(expr_parts) if expr_parts else None
    
    search_params = {
        "metric_type": "COSINE",
    }
    collection_name_ch = "question_vector_chinese"
    milvus_service_ch = QuestionVectorMilvusService(collection_name_ch)
    milvus_service_ch._init_collection()
    query_res_lst = milvus_service_ch.collection.search(
        data=[vector_item],
        anns_field="questionVector",
        param=search_params,
        limit=top_k,
        expr=expr,
        output_fields=["uuid", "area", "subject","series", "studySection", "subjectId", "knowledgeCode", "knowledge", "questionType", "questionVectorText", "questionVector"]
    )
    
    if query_res_lst and query_res_lst[0]:
        results = []
        for hit in query_res_lst[0]:
            result = {
                "uuid": hit.entity.get("uuid"),
                "area": hit.entity.get("area"),
                "subject": hit.entity.get("subject"),
                "subjectId": hit.entity.get("subjectId"),
                "questionType": hit.entity.get("questionType"),
                "knowledgeCode": hit.entity.get("knowledgeCode"),
                "knowledge": hit.entity.get("knowledge"),
                "questionVectorText": hit.entity.get("questionVectorText"),
                "score": hit.distance
            }
            results.append(result)
        uuid = results[0]["uuid"] if results else None
        analysis = get_analysis_from_uuid(uuid, subjectId=23)
        return analysis
    return None

def get_analysis_from_uuid(uuid, subjectId):
    """
    获取试题向量信息
    
    接口路径: GET /mock/263/system/vector/getVector
    
    参数:
        uuid: 试题uuid (必须)
        subjectId: 学科ID (必须)
    
    Returns:
        dict: 返回接口响应数据，包含 code, msg, data 字段
    """
    url = f"http://192.168.1.210:8071/api/vector/example?uuid={uuid}&subjectId={subjectId}"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        if not response.text:
            return ""
        analysis = response.json()
        content = "【试题分析】"+ analysis['data']['analysis'] + "\n" + "【答题分析】"+ analysis['data']['answerAnalysis'] 
        return content
    except requests.exceptions.JSONDecodeError as e:
        return ""
    except requests.exceptions.RequestException as e:
        return ""

def get_background_text(subject,knowledgeCode, area, series, studySection , taskType=5):
    """
    获取试题向量信息
    
    接口路径: GET /mock/263/system/vector/getVector
    
    参数:
        uuid: 试题uuid (必须)
        subjectId: 学科ID (必须)
    
    Returns:
        dict: 返回接口响应数据，包含 code, msg, data 字段
    """
    url = f"http://192.168.1.210:8071/api/fodder/list?subject={subject}&area={area}&series={series}&studySection={studySection}&taskType={taskType}&knowledgeCode={knowledgeCode}"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        if not response.text:
            return ""
        res = response.json()
        return res['rows'][0].get('content')
    except requests.exceptions.JSONDecodeError as e:
        return ""
    except requests.exceptions.RequestException as e:
        return ""

def get_dict_text(subject,area, series, studySection):
    """
    获取试题向量信息
    
    接口路径: GET /mock/263/system/vector/getVector
    
    参数:
        uuid: 试题uuid (必须)
        subjectId: 学科ID (必须)
    
    Returns:
        dict: 返回接口响应数据，包含 code, msg, data 字段
    """
    url = f"http://192.168.1.210:8071/api/findDictList"
    
    payload = {
        "subject": subject,
        "area": area,
        "series": series,
        "studySection": studySection,
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        if not response.text:
            return {}
        content = response.json()['data']
        dict_content = get_deepest_type2(content)
    
        return dict_content
    except requests.exceptions.JSONDecodeError as e:
        return {}
    except requests.exceptions.RequestException as e:
        return {}

def get_deepest_type2(data):
    """
    提取:
    - 最小级 type=2 节点
    - 所属板块(type=1)

    返回:
    [
        {
            "section_code": "一",
            "section_name": "语言知识与应用",
            "code": "5-3-1",
            "name": "正确辨析近义词及关联词"
        }
    ]
    """

    result = []

    def dfs(node, section=None):

        # list
        if isinstance(node, list):
            for item in node:
                dfs(item, section)
            return

        # 防御
        if not isinstance(node, dict):
            return

        node_type = node.get("type")

        # 记录板块(type=1)
        if node_type == 1:
            section = {
                "section_code": node.get("code"),
                "section_name": node.get("name")
            }

        # 非 type=2 继续递归
        if node_type != 2:
            for child in node.get("list", []):
                dfs(child, section)
            return

        # 找更深层 type=2
        children = [
            child for child in node.get("list", [])
            if isinstance(child, dict) and child.get("type") == 2
        ]

        # 有更深层
        if children:
            for child in children:
                dfs(child, section)
        else:
            # 当前已是最小级
            result.append({
                "id": section["section_code"] if section else None,
                "section": section["section_name"] if section else None,
                "knowledgeCode": node.get("code"),
                "knowledge": node.get("name")
            })

    dfs(data)

    return result

# # 使用
# result = get_deepest_type2(content[0])

# print(result)
def get_bge_embeddings_item(text: str):
    vector_lst = get_bge_embeddings([text])
    return vector_lst[0]

"""生成bge 向量化数据 """
def get_bge_embeddings(
        texts: List[str],
        model_name: str = "BAAI/bge-m3",
        url: str = "http://172.16.50.167:10080/v1/embeddings"
) -> Optional[List[List[float]]]:
    """
    将文本列表转换为向量。
    参数:
        texts: 字符串列表，例如 ["你好", "Hello"]
        model_name: 模型名称
        url: TEI 服务地址
    返回:
        成功则返回向量列表 (List[List[float]])，失败则返回 None
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "input": texts,
        "model": model_name
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        # 检查 HTTP 状态码是否为 200
        response.raise_for_status()

        data = response.json()
        # 提取所有的 embedding 向量
        embeddings = [item["embedding"] for item in data["data"]]
        return embeddings
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求发生错误: {e}")
        if hasattr(e.response, 'text'):
            print(f"错误详情: {e.response.text}")
        return None


def _demo():
    data_lst = [{'id': 106294, 'uuid': '2e74a45c-1c60-4125-8202-ca6a5dd16bee', 'parent_id': None, 'question_type': '选择题', 'area': '广东', 'knowledge': '指令与要求', 'target': '能在语境及情境中正确运用日常交际用语', 'plate': '情景交际', 'test_way': '补全对话', 'orig_question_stem': '', 'orig_answer': '', 'question_no': '1.', 'question_stem': "M: Would you mind turning down the radio， Dave? It's a little loud.\t(  )\nW:  <u>    </u>. I'll do it at once.\nA. Yes\tB. I think so\nC. Not at all\tD. You're welcome", 'answer': 'C'}]

    # 进行向量化
    question_lst = [data_lst[0]["question_stem"]]
    vector_lst = get_bge_embeddings(question_lst, model_name="BAAI/bge-m3")
    vector_item = vector_lst[0]

    data_lst[0]["embedding"] = vector_item
    en_question_service = EnglishQuestionService()

    en_question_service.insert(data_lst)

    print("_demo_1 done...")

def _demo_bge():
    data_lst = [
        {'id': 106294, 'uuid': '2e74a45c-1c60-4125-8202-ca6a5dd16bee', 'parent_id': None, 'question_type': '选择题',
         'area': '广东', 'knowledge': '指令与要求', 'target': '能在语境及情境中正确运用日常交际用语',
         'plate': '情景交际', 'test_way': '补全对话', 'orig_question_stem': None, 'orig_answer': None,
         'question_no': '1.',
         'question_stem': "M: Would you mind turning down the radio， Dave? It's a little loud.\t(  )\nW:  <u>    </u>. I'll do it at once.\nA. Yes\tB. I think so\nC. Not at all\tD. You're welcome",
         'answer': 'C'}]
    data_str = [data_lst[0]["question_stem"]]
    res = get_bge_embeddings(data_str)

    print(f"res: {res}")

def _demo_select():
    en_question_service = EnglishQuestionService()
    vector_lst = get_bge_embeddings(["M: Would you mind turning down the radio， Dave? It's a little loud."])
    results = en_question_service.search_by_knowledge(vector=vector_lst[0], knowledge_point=None, top_k=5)
    print(f"results: {results}")
    print("_demo select done...")


if __name__ == '__main__':
    # dict_res = get_dict_text(subject="语文", area="广东省", series="应试", studySection="中职")
    # res = get_background_text(subject="语文",knowledgeCode="5-24-2", area="广东省", series="应试", studySection="中职", taskType=5)
    import time
    start_time = time.time()
    # uuid = query_question_analysis("这道题的字音对吗,还有其他独赢吗") 
    result = query_question_analysis(question_stem="补写下列句子中的空缺部分。 (1)沧海月明珠有泪", knowledgeCode="1-14",questionType="填空题",area="广东省",series="应试",studySection="中职")
    end_time = time.time()
    print(f"duration_seconds: {end_time - start_time:.2f}")
    print(result)

