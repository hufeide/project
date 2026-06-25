"""后端接口 """
import requests, inspect
from enum import Enum, IntEnum, StrEnum
from dataclasses import dataclass, is_dataclass, asdict, field
from typing import Optional, List, Dict, Any, TypeVar, Generic, Type, Union, Tuple

# ==================================#
# 配置
# ==================================#
CONFIG_BASE_URL = "http://192.168.1.210:8071"

"""基础 HTTP 客户端，处理通用请求逻辑 """
class BaseAPIClient:
    def __init__(self, base_url: str, timeout: int = 10):
        self.base_url = base_url.rstrip('/')  # 防止 URL 拼接时出现双斜杠
        self.timeout = timeout
        self.headers = {
            "Content-Type": "application/json"
        }

    def _clean_data(self, data: Any) -> Any:
        """
        内部辅助方法：清洗数据
        1. 将 dataclass 转为字典
        2. 过滤掉值为 None 的参数
        3. 自动提取 Enum 类型的真实值
        """
        if data is None:
            return {}

        # 如果传入的是 dataclass 对象，先转换为字典
        if is_dataclass(data):
            data_dict = asdict(data)
        elif isinstance(data, dict):
            data_dict = data
        elif isinstance(data, list):
            # 如果是列表，清洗列表内部的每个元素（或者直接返回）
            return [str(item).strip() for item in data]
        else:
            raise ValueError("参数必须是 dict 或 dataclass 对象")

        cleaned_data = {}
        for key, val in data_dict.items():
            if val is not None:
                # 自动解包枚举值
                cleaned_data[key] = val.value if isinstance(val, Enum) else val
        return cleaned_data

    def _request(self, method: str, endpoint: str, params: Any = None, data: Any = None) -> Dict[str, Any]:
        """
        通用请求处理方法
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        # 清洗 Query 参数 (通常用于 GET) 和 Body 参数 (通常用于 POST)
        cleaned_params = self._clean_data(params)
        cleaned_data = self._clean_data(data)

        try:
            response = requests.request(
                method=method,
                url=url,
                params=cleaned_params,
                json=cleaned_data if method.upper() in ["POST", "PUT", "PATCH"] else None,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            return {"code": 500, "msg": "请求超时，请检查网络", "data": None}
        except requests.exceptions.RequestException as e:
            return {"code": 500, "msg": f"请求异常: {str(e)}", "data": None}

    def _request_new(self, method: str, endpoint: str, params: Any = None, data: Any = None) -> Dict[str, Any]:
        """通用请求处理方法"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        # 核心改动：只有当参数不为 None 时才进行清洗和组装
        kwargs = {
            "headers": self.headers,
            "timeout": self.timeout
        }

        if params is not None:
            kwargs["params"] = self._clean_data(params)

        if data is not None:
            cleaned_data = self._clean_data(data)
            # 如果是 POST/PUT/PATCH，直接把清洗后的 List[str] 塞进 json 参数
            if method.upper() in ["POST", "PUT", "PATCH"]:
                kwargs["json"] = cleaned_data
            else:
                # 防止其他请求类型（如 GET）误传了 data
                kwargs["data"] = cleaned_data

        try:
            # 使用 **kwargs 动态解包参数，避免传递无意义的 None 或空 {}
            response = requests.request(method=method, url=url, **kwargs)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            return {"code": 500, "msg": "请求超时，请检查网络", "data": None}
        except requests.exceptions.RequestException as e:
            return {"code": 500, "msg": f"请求异常: {str(e)}", "data": None}

    def get(self, endpoint: str, params: Any = None) -> Dict[str, Any]:
        """封装 GET 请求"""
        return self._request("GET", endpoint, params=params)

    def post(self, endpoint: str, data: Any = None) -> Dict[str, Any]:
        """封装 POST 请求"""
        return self._request("POST", endpoint, data=data)

    def put(self, endpoint: str, data: Any = None) -> Dict[str, Any]:
        """封装 PUT 请求"""
        return self._request("PUT", endpoint, data=data)

T = TypeVar('T')
"""通用 API 返回结果泛型封装 """
@dataclass
class ApiQueryResponse(Generic[T]):
    code: int
    msg: str
    total: int = 0
    rows: List[T] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], item_class: Type[T]) -> "ApiQueryResponse[T]":
        code = data.get("code", 500)
        msg = data.get("msg", "未获取到返回信息")
        total = data.get("total", 0)

        raw_rows = data.get("rows", [])
        parsed_rows = []
        # for row in raw_rows:
        #     if hasattr(item_class, 'from_dict'):
        #         parsed_rows.append(item_class.from_dict(row))
        #     else:
        #         parsed_rows.append(item_class(**row) if isinstance(row, dict) else row)
        # return cls(code=code, msg=msg, total=total, rows=parsed_rows)
        # 1. 动态获取 item_class 构造函数支持的参数集合
        try:
            sig = inspect.signature(item_class)
            valid_params = set(sig.parameters.keys())
            # 检查类中是否显式定义了 **kwargs，如果定义了则可以直接全量传入
            has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        except ValueError:
            # 兼容某些无法获取 signature 的特殊类型
            valid_params = set()
            has_kwargs = True

        for row in raw_rows:
            if hasattr(item_class, 'from_dict'):
                parsed_rows.append(item_class.from_dict(row))
            else:
                if isinstance(row, dict):
                    # 2. 核心修改：如果目标类没有 **kwargs，则剔除它不认识的字段
                    if not has_kwargs:
                        filtered_row = {k: v for k, v in row.items() if k in valid_params}
                    else:
                        filtered_row = row

                    parsed_rows.append(item_class(**filtered_row))
                else:
                    parsed_rows.append(row)

        return cls(code=code, msg=msg, total=total, rows=parsed_rows)

"""通用新增、更新返回值封装 """
@dataclass
class ApiResponse:
    code: int
    msg: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApiResponse":
        code = data.get("code", 500)
        msg = data.get("msg", "未获取到返回信息")
        return cls(code=code, msg=msg)

# ==========================================
# 0. 枚举参数定义 (enum)
# ==========================================
"""学科"""
class SubjectEnum(StrEnum):
    CHINESE = "语文"
    MATH = "数学"
    ENGLISH = "英语"

"""学科ID """
class SubjectIdEnum(IntEnum):
    CHINESE = 23 # 语文
    MATH = 24 # 数学
    ENGLISH = 25 # 英语

"""地区 """
class AreaEnum(StrEnum):
    # ================= 华南地区 =================
    GUANG_DONG = "广东省"
    GUANG_XI = "广西省"
    HAI_NAN = "海南省"

    # ================= 华北地区 =================
    BEI_JING = "北京市"
    TIAN_JIN = "天津市"
    HE_BEI = "河北省"
    SHAN_XI = "山西省"
    NEI_MENG_GU = "内蒙古自治区"

    # ================= 东北地区 =================
    LIAO_NING = "辽宁省"
    JI_LIN = "吉林省"
    HEI_LONG_JIANG = "黑龙江省"

    # ================= 华东地区 =================
    SHANG_HAI = "上海市"
    JIANG_SU = "江苏省"
    ZHE_JIANG = "浙江省"
    AN_HUI = "安徽省"
    FU_JIAN = "福建省"
    JIANG_XI = "江西省"
    SHAN_DONG = "山东省"
    TAI_WAN = "台湾省"

    # ================= 华中地区 =================
    HE_NAN = "河南省"
    HU_BEI = "湖北省"
    HU_NAN = "湖南省"

    # ================= 西南地区 =================
    CHONG_QING = "重庆市"
    SI_CHUAN = "四川省"
    GUI_ZHOU = "贵州省"
    YUN_NAN = "云南省"
    XI_ZANG = "西藏自治区"

    # ================= 西北地区 =================
    SHAAN_XI = "陕西省"  # 注意：这里使用 SHAAN_XI 来区分山西省 (SHAN_XI)
    GAN_SU = "甘肃省"
    QING_HAI = "青海省"
    NING_XIA = "宁夏回族自治区"
    XIN_JIANG = "新疆维吾尔自治区"

"""任务类型 """
class TaskTypeEnum(IntEnum):
    PREPARE_TASK = 1 # 准备任务
    ANSWER_CHECK_TASK = 2
    KNOWLEDGE_CEHCK_TASK = 3

    ANALYSIS_TASK = 5 # 试题分析、答题分析任务
    DIFFICULTY_TASK = 6 # 难易度任务
    RUBRIC_TASK = 7 # 评分量表任务
    DEDUPLICATION_TASK = 8 # 去重任务

"""任务组状态 """
class TaskGroupStatusEnum(IntEnum):
    INIT = 0 # 初始化
    RUNNING = 1 # 进行中
    FINISHED = 2 # 已完成
    FAILED = 3 # 失败

"""任务状态 """
class TaskStatusEnum(IntEnum):
    INIT = 0 # 初始化
    RUNNING = 1 # 进行中
    CHECK_IS_SUCCESS = 2 # 校验通过待抽检
    CHECK_IS_FAILED = 3 # 校验不通过待审核
    FINISHED = 4 # 已完成
    SYNCHRONIZED = 5 # 已同步
    FAILED = 6 # 失败

"""任务步骤 """
class TaskStepEnum(IntEnum):
    DATA_CLEAN = 1 # 格式清理
    ANSWER_CHECK = 2 # 答案判断
    KNOWLEDGE_CHECK = 3 # 知识点判断
    QUESTION_TYPE_CHECK = 4 # 题型判断
    MATH_DRAW = 5 # 数学画图

"""系列状态 """
class SeriesEnum(StrEnum):
    YING_SHI = "应试"
    TONG_BU = "同步"

"""学段状态 """
class StudySectionEnum(StrEnum):
    ZHONG_ZHI = "中职"
    PU_GAO = "普高"

"""比较结果是否一致 """
class CompareResultEnum(IntEnum):
    NO = 0 # 不一致
    YES = 1 # 一致
    NO_RESULT = 2 # 无法判断

"""子任务类型 """
class SubTaskTypeEnum(IntEnum):
    ANSWER_CHECK = 1 # 答案判断
    KNOWLEDGE_CHECK = 2 # 知识点判断
    QUESTION_TYPE_CHECK = 3 # 题型判断
    MATH_DRAW = 4 # 数学画图

"""模型名称"""
class ModelNameEnum(StrEnum):
    ChatGPT = "ChatGPT"
    doubao = "doubao"
    Gemini = "Gemini"
    Deepseek = "Deepseek"
    Qwen = "Qwen"

class ParseStatusEnum(IntEnum):
    NO = 0
    YES = 1

"""能力层级映射 """
CH_ABILITY_LEVEL_DICT = {
    "A": "识记",
    "B": "理解",
    "C": "分析综合",
    "D": "鉴赏评价",
    "E": "表达应用", # 分为非作文 和 作文
    "F": "探究"
}

# ==========================================
# 1. 定义请求参数实体类 (Data Class)
# ==========================================
"""任务组请求 """
@dataclass
class TaskGroupQuery:
    """
    任务组查询参数封装
    """
    subjectId: SubjectIdEnum  # 学科
    taskType: TaskTypeEnum  # 任务类型

    area: Optional[str] = None  # 地区，如："广东省"
    series: Optional[str] = None  # 系列，如："应试"
    questionType: Optional[str] = None
    status: Optional[TaskGroupStatusEnum] = None  # 状态
    pageNum: Optional[int] = None
    pageSize: Optional[int] = None

"""任务请求 """
@dataclass
class TaskDetailQuery:
    """
    任务查询参数封装
    """
    taskGroupId: int # 任务组
    subjectId: int  # 学科
    taskType: int  # 任务类型
    pageNum: int = 1
    pageSize: int = 100

    taskStatus: Optional[TaskStatusEnum] = None  # 状态
    questionType: Optional[str] = None
    knowledge: Optional[str] = None  # 知识点
    uuid: Optional[str] = None  # uuid
    subTaskType: Optional[SubTaskTypeEnum] = None # 子任务类型

@dataclass
class TaskGroupItem:
    """任务组明细对象"""
    id: int
    subjectId: int
    subject: str
    area: str
    series: str
    studySection: str
    totalCount: int
    doneCount: int
    status: int
    taskType: int
    createTime: str
    updateTime: str

    # 可选/可能为空的字段
    createBy: Optional[str] = None
    updateBy: Optional[str] = None
    remark: Optional[str] = None
    plate: Optional[str] = None
    questionType: Optional[str] = None
    knowledgeCode: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskGroupItem":
        """
        将字典转化为对象。
        (内置过滤机制：防止后端接口新增了字段导致实例化报错)
        """
        valid_keys = cls.__dataclass_fields__.keys()
        # 只提取在数据类中定义过的字段
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)

    def to_dict(self) -> Dict[str, Any]:
        """
        将对象转化为字典。
        (支持深度递归转化嵌套的数据类)
        """
        return asdict(self)

"""任务明细/查询结果对象 """
@dataclass
class TaskDetailItem:
    # === 核心 ID 类 ===
    taskId: Optional[int] = None
    taskGroupId: Optional[int] = None
    subTaskId: Optional[int] = None
    questionId: Optional[int] = None
    subjectId: Optional[int] = None
    uuid: Optional[str] = None
    questionUuid: Optional[str] = None

    # === 任务状态与类型 ===
    taskType: Optional[int] = None
    taskStatus: Optional[int] = None
    taskStep: Optional[int] = None
    subTaskType: Optional[int] = None
    subTaskStatus: Optional[int] = None
    subTaskStep: Optional[int] = None

    # === 基础属性类 ===
    subject: Optional[str] = None           # 学科
    area: Optional[str] = None              # 地区
    series: Optional[str] = None            # 系列
    studySection: Optional[str] = None      # 学段
    questionType: Optional[str] = None      # 题型
    questionNo: Optional[str] = None        # 题号
    abilityLevel: Optional[str] = None      # 能力层级
    testWayName: Optional[str] = None       # 考法名称
    needDraw: Optional[int] = None          # 是否需要画图 (推断为 int 或 bool)

    # === 知识点与内容类 ===
    knowledgeCode: Optional[str] = None     # 知识点编码
    knowledge: Optional[str] = None         # 知识点名称
    material: Optional[str] = None           # 材料
    questionMaterial: Optional[str] = None   # 试题材料
    stem: Optional[str] = None              # 题干
    questionStem: Optional[str] = None      # 试题题干
    answer: Optional[str] = None            # 答案
    cleanAnswer: Optional[str] = None # 清洗后答案
    analysis: Optional[str] = None          # 解析
    answerAnalysis: Optional[str] = None    # 答案解析
    compareResult: Optional[str] = None     # 对比结果

    # === 时间记录 ===
    createTime: Optional[str] = None        # 创建时间 (ISO 8601 格式)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskDetailItem":
        """
        字典转对象，自动过滤掉不在数据类中定义的字段
        """
        if not data:
            return cls()
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)

    def to_dict(self) -> Dict[str, Any]:
        """
        将对象转化为字典。
        (支持深度递归转化嵌套的数据类)
        """
        return asdict(self)

"""任务组更新对象 """
@dataclass
class TaskGroupUpdate:
    taskGroupId: int
    groupStatus: Optional[TaskGroupStatusEnum] = None

"""任务更新对象 """
@dataclass
class TaskDetailUpdate:
    taskId: int
    taskStatus: Optional[TaskStatusEnum] = None
    taskStep: Optional[TaskStepEnum] = None

"""子任务更新对象 """
@dataclass
class SubTaskUpdate:
    subTaskId: int
    subTaskStatus: Optional[TaskStatusEnum] = None
    subTaskStep: Optional[int] = None

"""模型记录添加对象 """
@dataclass
class ModelRecordAdd:
    taskId: int
    modelName: str
    flow: str

    subTaskId: Optional[int] = None
    responseId: Optional[str] = None
    modelInput: Optional[str] = None
    modelThink: Optional[str] = None
    modelOutput: Optional[str] = None
    parsedStatus: Optional[int] = None
    outputParsedResult1: Optional[int] = None
    outputParsedResult2: Optional[int] = None

"""模型比对结果添加对象 """
@dataclass
class ModelCompareResultAdd:
    taskId: int

    subTaskId: Optional[int] = None # 子任务ID
    questionUuid: Optional[str] = None # 试题uuid
    modelName1: Optional[str] = None # 模型1名称
    modelName2: Optional[str] = None # 模型2名称
    compareModelName: Optional[str] = None # 比对模型名
    compareResult: Optional[CompareResultEnum] = None # 比对结果是否一致
    reason: Optional[str] = None # 原因
    model1Result1: Optional[str] = None # 模型1结果1
    model1Result2: Optional[str] = None # 模型1结果2
    model2Result1: Optional[str] = None # 模型2结果1
    model2Result2: Optional[str] = None # 模型2结果2
    effectModel: Optional[int] = None # 生效模型(1模型一,2模型二)

"""试题任务清晰数据更新对象 """
@dataclass
class CleanDataUpdate:
    taskId: int
    subjectId: SubjectIdEnum

    subTaskId: Optional[int] = None # 子任务ID
    uuid: Optional[str] = None
    material: Optional[str] = None # 清洗后材料
    stem: Optional[str] = None # 清洗后题干
    cleanAnswer: Optional[str] = None # 清理后答案

    needDraw: Optional[int] = None # 是否需要画图 0否 1是
    drawPath: Optional[str] = None # 画图路径
    docPath: Optional[str] = None # 文件路径（计算机）
    imagePath: Optional[str] = None # 图片路径（计算机）
    solveProblem: Optional[str] = None # 解题过程

"""提示词查询请求对象 """
@dataclass
class PromptQuery:
    subject: str # 学科
    area: str # 地区
    series: str # 系列
    studySection: str # 学段
    taskType: int # 任务类型

    knowledgeCode: Optional[str] = None # 知识点
    questionType: Optional[str] = None # 题型
    modelName: Optional[str] = None # 模型名称
    link: Optional[str] = None # 环节
    round: Optional[str] = None # 轮次
    testWayName: Optional[str] = None # 考查方式
    plate: Optional[str] = None # 板块
    discourseGenre: Optional[str] = None # 语篇体裁
    isDiscourse: Optional[int] = None # 是否包含材料
    version: Optional[str] = None # 版本号
    status: Optional[str] = None # 状态 0:启用;2:生产;3:停用;

"""提示词对象 """
@dataclass
class PromptItem:
    id: Optional[int] = None
    area: Optional[str] = None # 地区
    series: Optional[str] = None # 系列
    subject: Optional[str] = None # 学科
    studySection: Optional[str] = None # 学段
    taskType: Optional[str] = None # 任务类型
    knowledgeCode: Optional[str] = None # 知识点代码
    questionType: Optional[str] = None # 题型
    discourseGenre: Optional[str] = None # 语篇体裁
    plate: Optional[str] = None # 板块
    testWayName: Optional[str] = None # 考查方式
    isDiscourse: Optional[int] = None # 是否有材料
    modelName: Optional[str] = None # 模型名称
    link: Optional[str] = None # 环节
    round: Optional[str] = None # 轮次
    systemPromptContent: Optional[str] = None # 系统提示词内容
    taskPromptContent: Optional[str] = None # 任务提示词内容
    exampleContent: Optional[str] = None # 样例
    version: Optional[str] = None # 版本号
    status: Optional[str] = None # 状态
    createTime: Optional[str] = None # 创建时间
    updateTime: Optional[str] = None # 修改时间

"""素材查询请求对象 """
@dataclass
class FodderQuery:
    subject: str
    area: AreaEnum
    series: str
    studySection: str
    taskType: int
    knowledgeCode: Optional[str] = None

"""素材对象 """
@dataclass
class FodderItem:
    id: Optional[int] = None # 主键 ID
    area: Optional[str] = None # 地区
    series: Optional[str] = None # 系列
    studySection: Optional[str] = None # 学段
    taskType: Optional[int] = None # 任务类型
    knowledgeCode: Optional[str] = None # 知识点代码
    content: Optional[str] = None # 素材内容
    createTime: Optional[str] = None # 创建时间
    updateTime: Optional[str] = None # 更新时间

# ==========================================
# 2. 服务类封装
# ==========================================
"""任务组服务 继承自 BaseAPIClient """
class TaskGroupService(BaseAPIClient):
    def __init__(self, base_url: str = CONFIG_BASE_URL):
        # 调用父类初始化
        super().__init__(base_url=base_url)

    def get_list(self, query: TaskGroupQuery) -> ApiQueryResponse[TaskGroupItem]:
        """
        获取任务组列表 (GET 请求)
        """
        # 直接调用父类的 get 方法，传入 endpoint 和 dataclass 参数
        response_data = self.get("api/taskgroup/list", params=query)
        return ApiQueryResponse.from_dict(data= response_data, item_class=TaskGroupItem)

    def update_status(self, data: TaskGroupUpdate) -> ApiResponse:
        response_data = self.put("api/task/edit", data=data)
        return ApiResponse.from_dict(response_data)

"""任务服务 继承自 BaseAPIClient """
class TaskDetailService(BaseAPIClient):
    def __init__(self, base_url: str = CONFIG_BASE_URL):
        # 调用父类初始化
        super().__init__(base_url=base_url)

    def get_list(self, query: TaskDetailQuery) -> ApiQueryResponse[TaskDetailItem]:
        """
        获取任务组列表 (GET 请求)
        """
        # 直接调用父类的 get 方法，传入 endpoint 和 dataclass 参数
        response_data = self.get("api/task/detail/list", params=query)
        return ApiQueryResponse.from_dict(response_data, item_class=TaskDetailItem)

    def get_item_by_list(self,
                         task_group_item: TaskGroupItem,
                         index: int,
                         page_size: int = 100,
                         last_query_res: ApiQueryResponse[TaskDetailItem] = None,
                         task_detail_query: TaskDetailQuery = None) \
            -> Tuple[Optional[TaskDetailItem], ApiQueryResponse[TaskDetailItem], int]:
        """
        通过全局索引获取单条数据，返回: (数据项, 响应缓存, 总条数)
        """

        # 1. 计算目标页码和页内相对索引
        target_page = (index // page_size) + 1
        local_index = index % page_size

        # 2. 检查缓存是否有效 (是否存在且页码匹配)
        is_cache_valid = (
                last_query_res is not None and
                getattr(last_query_res, '_page_no', None) == target_page
        )

        if not is_cache_valid:
            # 构造分页查询对象
            if task_detail_query is None:
                task_detail_query = TaskDetailQuery(
                    taskGroupId=task_group_item.id,
                    subjectId=task_group_item.subjectId,
                    taskType=task_group_item.taskType
                )
            else:
                task_detail_query.taskGroupId=task_group_item.id
                task_detail_query.subjectId=task_group_item.subjectId
                task_detail_query.taskType=task_group_item.taskType
            task_detail_query.pageNum=target_page
            task_detail_query.pageSize=page_size
            # 发起网络请求
            last_query_res = self.get_list(task_detail_query)
            # 注入当前页码标识，用于下次调用时的缓存比对
            setattr(last_query_res, '_page_no', target_page)

        # 3. 获取总条数 (如果接口未返回则默认为 0)
        total_count = getattr(last_query_res, 'total', 0)

        # 4. 边界检查：如果 index 越界（超过总数或当前页实际返回数）
        if not last_query_res.rows or local_index >= len(last_query_res.rows):
            return None, last_query_res, total_count

        # 5. 返回结果：(数据, 响应对象, 总数)
        return last_query_res.rows[local_index], last_query_res, total_count


    def update_clean_data(self, data: CleanDataUpdate) -> ApiResponse:
        response_data = self.put("api/task/update", data=data)
        return ApiResponse.from_dict(response_data)

    def update_status(self, data: TaskDetailUpdate) -> ApiResponse:
        response_data = self.put("api/task/edit", data=data)
        return ApiResponse.from_dict(response_data)

    def update_sub_status(self, data: SubTaskUpdate) -> ApiResponse:
        response_data = self.put("api/task/edit", data=data)
        return ApiResponse.from_dict(response_data)

"""模型记录服务 """
class ModelService(BaseAPIClient):
    def __init__(self, base_url: str = CONFIG_BASE_URL):
        super().__init__(base_url)

    def add_record(self, data: ModelRecordAdd) -> ApiResponse:
        response_data = self.post("api/modelrecord/add", data=data)
        return ApiResponse.from_dict(response_data)

    def add_compare_result(self, data: ModelCompareResultAdd) -> ApiResponse:
        response_data = self.post("api/compare/add", data=data)
        return ApiResponse.from_dict(response_data)

"""提示词模板服务 """
class PromptService(BaseAPIClient):
    def __init__(self, base_url: str = CONFIG_BASE_URL):
        super().__init__(base_url)

    def get_list(self, query: PromptQuery) -> ApiQueryResponse[PromptItem]:
        response_data = self.get("api/prompt/list", params=query)
        return ApiQueryResponse.from_dict(response_data, item_class=PromptItem)

"""素材服务 """
class FodderService(BaseAPIClient):
    def __init__(self, base_url: str = CONFIG_BASE_URL):
        super().__init__(base_url)

    def get_list(self, query: FodderQuery) -> ApiQueryResponse[FodderItem]:
        response_data = self.get("api/fodder/list", params=query)
        return ApiQueryResponse.from_dict(response_data, item_class=FodderItem)

# ================================
# 试题向量库服务 开始
# ================================
"""素材查询请求对象 """
@dataclass
class QuestionVectorQuery:
    subjectId: SubjectIdEnum | int # 必须字段

    area: Optional[AreaEnum] = None # 地区
    series: Optional[SeriesEnum] = None # 系列状态
    studySection: Optional[StudySectionEnum] = None # 学段状态
    subject: Optional[SubjectEnum] = None # 学科

    pageNum: Optional[int] = None
    pageSize: Optional[int] = None

"""向量化 明细 """
@dataclass
class QuestionVectorItem:
    # ==========================================
    # 1. 核心 ID 与关联类
    # ==========================================
    taskId: Optional[int] = None  # 任务ID
    taskGroupId: Optional[int] = None  # 任务组ID
    subTaskId: Optional[int] = None  # 子任务ID
    uuid: Optional[str] = None  # 试题任务/通用唯一标识
    questionId: Optional[int] = None  # 试题ID
    questionUuid: Optional[str] = None  # 材料唯一标识

    # ==========================================
    # 2. 试题核心内容（文本/多媒体）
    # ==========================================
    questionMaterial: Optional[str] = None  # 原始试题材料
    material: Optional[str] = None  # 清洗后材料
    questionStem: Optional[str] = None  # 原始试题题干
    stem: Optional[str] = None  # 清洗后题干
    answer: Optional[str] = None  # 原始答案
    cleanAnswer: Optional[str] = None  # 清洗后答案
    verifyAnswer: Optional[str] = None  # 审核/验证后的答案
    analysis: Optional[str] = None  # 试题分析
    answerAnalysis: Optional[str] = None  # 答题分析
    scoringRubric: Optional[str] = None  # 评分标准/细则

    # ==========================================
    # 3. 试题分类与属性标签
    # ==========================================
    subject: Optional[str] = None  # 学科 (如: 数学、英语)
    subjectId: Optional[int] = None  # 学科ID
    studySection: Optional[str] = None  # 学段 (如: 小学、初n中、高中)
    area: Optional[str] = None  # 地区/地域
    series: Optional[str] = None  # 教材版本/系列
    questionNo: Optional[str] = None  # 题号
    questionType: Optional[str] = None  # 原始题型
    verifyQuestionType: Optional[str] = None  # 审核/验证后的题型
    abilityLevel: Optional[str] = None  # 能力层级/考核要求
    testWayName: Optional[str] = None  # 考查方式名称

    # ==========================================
    # 4. 知识点相关
    # ==========================================
    knowledgeCode: Optional[str] = None  # 原始知识点编码
    knowledge: Optional[str] = None  # 原始知识点名称
    verifyKnowledgeCode: Optional[str] = None  # 审核后的知识点编码
    verifyKnowledge: Optional[str] = None  # 审核后的知识点名称

    # ==========================================
    # 5. 流程、状态与控制类
    # ==========================================
    taskType: Optional[str] = None  # 任务类型
    subTaskType: Optional[str] = None  # 子任务类型
    taskStatus: Optional[int] = None  # 主任务状态
    subTaskStatus: Optional[int] = None  # 子任务状态
    taskStep: Optional[str] = None  # 主任务步骤/环节
    subTaskStep: Optional[str] = None  # 子任务步骤/环节
    auditStatus: Optional[int] = None  # 审核状态
    isSync: Optional[int] = None  # 是否同步/下发 (通常为 0/1)
    compareResult: Optional[str] = None  # 比对结果 (如查重或文本比对)
    manualReview: Optional[bool] = None  # 是否需要人工介入/人工审核

    # ==========================================
    # 6. 状态质检/校验细项
    # ==========================================
    knowledgeStatus: Optional[int] = None  # 知识点标注状态
    questionTypeStatus: Optional[int] = None  # 题型标注状态
    answerStatus: Optional[int] = None  # 答案标注状态

    # ==========================================
    # 7. 绘图相关
    # ==========================================
    needDraw: Optional[bool] = None  # 是否需要绘图/修图
    drawPath: Optional[str] = None  # 绘图文件路径/OSS地址

    # ==========================================
    # 8. 分值、难度与模型
    # ==========================================
    score: Optional[float] = None  # 试题分值
    answerStep: Optional[str] = None  # 踩分点/作答步骤
    difficulty: Optional[str] = None  # 难度文本 (如: 容易、较难)
    difficultyRatio: Optional[float] = None  # 难度系数 (如: 0.65)
    effectModel: Optional[str] = None  # 预估模型/影响模型

    # ==========================================
    # 9. 其他辅助
    # ==========================================
    tempPrompt: Optional[str] = None  # 临时提示词/大模型Prompt
    solveProblem: Optional[str] = None  # 解决的问题/备注
    createTime: Optional[str] = None  # 创建时间 (可以是 str 或 datetime)

"""试题向量库服务 """
class QuestionVectorService(BaseAPIClient):
    def __init__(self, base_url: str = CONFIG_BASE_URL):
        super().__init__(base_url)

    def get_list(self, data: QuestionVectorQuery) -> ApiQueryResponse[QuestionVectorItem]:
        response_data = self.get("api/vector/list", params=data)
        return ApiQueryResponse.from_dict(response_data, item_class=QuestionVectorItem)

    def update_vectorized_uuid_list(self, data: List[str]) -> ApiResponse:
        response_data = self.put("api/vector/update", data=data)
        return ApiResponse.from_dict(response_data)

    """ 通过全局索引获取单条数据，返回: (数据项, 响应缓存, 总条数)"""
    def get_item_by_list(self,
                         question_vector_query: QuestionVectorQuery,
                         last_query_res: ApiQueryResponse[QuestionVectorItem],
                         index: int,
                         page_size: int = 100) \
            -> Tuple[Optional[QuestionVectorItem], ApiQueryResponse[QuestionVectorItem], int]:
        # 1. 计算目标页码和页内相对索引
        target_page = (index // page_size) + 1
        local_index = index % page_size
        # 2. 检查缓存是否有效 (是否存在且页码匹配)
        is_cache_valid = (
            last_query_res is not None and
            getattr(last_query_res, '_page_no', None) == target_page
        )
        if not is_cache_valid:
            question_vector_query.pageNum=target_page
            question_vector_query.pageSize=page_size
            # 发起网络请求
            last_query_res = self.get_list(question_vector_query)
            # 注入当前页码标识，用于下次调用时的缓存比对
            setattr(last_query_res, '_page_no', target_page)
        # 3. 获取总条数 (如果接口未返回则默认为 0)
        total_count = getattr(last_query_res, 'total', 0)
        # 4. 边界检查：如果 index 越界（超过总数或当前页实际返回数）
        if not last_query_res.rows or local_index >= len(last_query_res.rows):
            return None, last_query_res, total_count
        # 5. 返回结果：(数据, 响应对象, 总数)
        return last_query_res.rows[local_index], last_query_res, total_count

# ================================
# 试题向量库服务 结束
# ================================

def _test_task_group_service():
    task_group_service = TaskGroupService()

    # 1. 正常实例化 (如果少传 subjectId 或 status，IDE 和 Python 都会报错)
    query_params = TaskGroupQuery(
        subjectId=SubjectIdEnum.CHINESE,
        taskType=TaskTypeEnum.RUBRIC_TASK,
    )

    # 2. 发起请求
    response_obj = task_group_service.get_list(query=query_params)
    print(f"response_obj: {response_obj}")
    for row in response_obj.rows:
        print(f"row type: {type(row)}")
        print(f"row id: {row.id}")

def _test_task_detail_service():
    task_detail_service = TaskDetailService()
    query_params = TaskDetailQuery(
        taskGroupId=570,
        subjectId=SubjectIdEnum.CHINESE,
        taskType=TaskTypeEnum.RUBRIC_TASK,
        pageNum=1,
        pageSize=100
    )
    response_obj = task_detail_service.get_list(query=query_params)
    print(f"response_obj: {response_obj}")

def _test_update_all_task_service():
    task_detail_service = TaskDetailService()

    upd_data = TaskDetailUpdate(
        taskId=2600,
        taskStatus=TaskStatusEnum.FINISHED
    )
    upd_res = task_detail_service.update_status(data=upd_data)
    print(f"upd_res: {upd_res}")

def _test_model_compare_service():
    pass

def _test_prompt_service():
    prompt_service = PromptService()
    query_params = PromptQuery(
        SubjectEnum.CHINESE,

    )

def _test_fodder_service():
    fodder_service = FodderService()
    fodder_query = FodderQuery(subject=SubjectEnum.CHINESE,
                               area=AreaEnum.GUANG_DONG,
                               # area="广东省",
                               series=SeriesEnum.YING_SHI,
                               studySection=StudySectionEnum.ZHONG_ZHI,
                               taskType=TaskTypeEnum.ANALYSIS_TASK)
    fodder_res = fodder_service.get_list(query=fodder_query)
    print(f"fodder_res: {fodder_res}")

def _test_get_item_by_list():
    import json
    task_group_service = TaskGroupService()
    query_params = TaskGroupQuery(subjectId=SubjectIdEnum.MATH,
                                  taskType=TaskTypeEnum.PREPARE_TASK,
                                  area=AreaEnum.GUANG_DONG,
                                  series=SeriesEnum.YING_SHI)
    task_res = task_group_service.get_list(query=query_params)

    group_info = task_res.rows[0]
    # for row in task_res.rows:
    #     if row.id == 564:
    #         group_info = row

    print(f"group_info: \n{group_info}")

    task_detail_service = TaskDetailService()
    res = None
    index = 0
    total = 1  # 初始设为 1 进入循环
    task_detail_query = TaskDetailQuery(
        taskGroupId=group_info.id,
        subjectId=SubjectIdEnum.MATH,
        taskType=TaskTypeEnum.ANALYSIS_TASK,
        questionType="填空题",
        subTaskType=SubTaskTypeEnum.ANSWER_CHECK
    )
    while index < total:
        # 每次调用都会自动处理：[内存取数据] 或 [跨页发起请求]
        item, res, total = \
            task_detail_service.get_item_by_list(group_info,
                                                 index,
                                                 last_query_res=res,
                                                 task_detail_query=task_detail_query,
                                                 page_size=100)
        index += 1
        # print(f"task_detail_query: {task_detail_query}")
        query_params = json.dumps(asdict(task_detail_query), ensure_ascii=False)
        print(f"task_detail_query: {query_params}")
        if item:
            print(f"进度: {index}/{total} - 处理任务: {item.taskId} subTaskId: {item.subTaskId}")
            # print(f"item: {item}")

            if index > 10:
                break
            # 执行业务逻辑


    print("全量数据轮询处理完毕！")

def _test_question_vector_list():
    vector_query = QuestionVectorQuery(subjectId=SubjectIdEnum.CHINESE)
    question_vector_service = QuestionVectorService()
    # vector_res = question_vector_service.get_list(data=vector_query)
    # print(f"vector_res total: {vector_res.total} \nrows: {len(vector_res.rows)}")
    # print(f"vector_res: {vector_res}")

    res = None
    index = 0
    total = 1  # 初始设为 1 进入循环
    while index < total:
        item, res, total = \
            question_vector_service.get_item_by_list(question_vector_query=vector_query,
                                                     last_query_res=res,
                                                     index=index,
                                                     page_size=100)
        index += 1


        # linux
        print(f"index: {index}/{total} uuid: {item.uuid}")


if __name__ == '__main__':

    # _test_task_group_service()
    # _test_task_detail_service()
    # _test_update_all_task_service()

    # _test_fodder_service()
    # _test_get_item_by_list()

    _test_question_vector_list()

    print(f"__main__ done...")
