
import json
import os
import sys
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(current_dir)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


class knowledge_md:
    def __init__(self):
        self.knowledge_point = pd.read_excel(os.path.join(os.path.dirname(current_dir),"data", "参考知识.xlsx"))

    def get_knowledge_point(self, knowledge_str):
        matched = self.knowledge_point[self.knowledge_point['知识代码'] == knowledge_str]
        if len(matched) == 0:
            return ""
        knowledge_point = matched['文件'].values[0]
        try:
            with open(f'{BASE_DIR}/data/knowledge/{knowledge_point}.md', 'r', encoding='utf-8') as f:
                knowledge_md = f.read()
            return knowledge_md
        except FileNotFoundError:
            return ""
def match_example(df, question_type, knowledgeCode):
    def split_codes(x):
        if pd.isna(x):
            return []
        return [i.strip() for i in str(x).strip("、").split("、") if i.strip()]
    df_filtered = df[df["题型"] == question_type].copy()
    df_filtered["code_list"] = df_filtered["对应广东字典库条目"].apply(split_codes)

    matched = df_filtered[df_filtered["code_list"].apply(lambda x: knowledgeCode in x)]

    if matched.empty:
        return None

    return "".join(matched["示例"])
def fill_example(row, df):
    result = match_example(df, row["questionType"], row["knowledgeCode"])

    # 👉 fallback：没有匹配就保留原值
    return result if result is not None else row["answer_type_example_one"]


def get_task_from_json():
    file_path = "/data/weidu_new/code_25/0703/dfjg_chinese_rec_v1/Template/exam_item_analysis/project/data/260323_chinese_json.json"
    def load_and_process_dfjg(file_path):
        tasks = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                for item in row.get('list', []):
                    if item.get('is_matrial') == '是':
                        material = item.get('questionStem', '')
                        for sub in item.get('list', []):
                            tasks.append({
                                "uuid": sub.get('uuid'),
                                "context": material,
                                "question": sub.get('questionStem', ''),
                                "answer": sub.get('answer', ''),
                                "type": sub.get('type'),
                                "knowledgeCode": sub.get('knowledge')
                            })
                    else:
                        tasks.append({
                            "uuid": item.get('uuid'),
                            "context": None,
                            "question": item.get('questionStem', ''),
                            "answer": item.get('answer', ''),
                            "type": item.get('type'),
                            "knowledgeCode": item.get('knowledge')
                        })
        return tasks
    tasks = load_and_process_dfjg(file_path)
    return tasks