
from typing import Any
from .json_validation import  is_list_of_list
from .image_utils import build_image_instruction

def prompt_answer_analysis(processed_data):
    knowledge_text = processed_data['knowledge']
    knowledge_point = processed_data['knowledge_point']
    level = processed_data['level']
    promote_out = processed_data['promote_out']
    answer_example = processed_data['answer_example']
    generated_text = processed_data['promote_head']
    question_type = processed_data['question_type']
    question = processed_data['question']
    answer = processed_data['answer']
    material = processed_data['material']
    if material:
        material_text = f"材料：/n{material}/n"
    else:
        material_text = ""
    
    if answer_example:
        answer_example = f"##【典型示例】：/n {answer_example}"
    else:
        answer_example = ""
    
    if knowledge_text:
        knowledge_text = f"背景知识参考：{knowledge_text} "
    else:
        knowledge_text = ""

    if is_list_of_list(processed_data['image_list']):
        image_rule_text = build_image_instruction(processed_data['image_list'])
    else:
        image_rule_text = build_image_instruction([processed_data['image_list']])

    if image_rule_text!="":
        image_rule_text = f"【图片使用规则（必须严格遵守）】/n {image_rule_text}"
    # image_rule_text = build_image_instruction([processed_data['image_list']])
    prompt = f"""
    ######################
    {knowledge_text}
    ###################### 
    
    # 试题分析答题分析要求：{promote_out}
    {answer_example}
    {image_rule_text}
    # 请对以下【题目】进行试题分析和答题分析：
    题型：{question_type}
    knowledge：{knowledge_point}
    level：{level}
    '{material_text}'
    题目内容：'{question}'
    正确答案：{answer}
    """
    sys_prompt = f"""
    {generated_text}
    注意：这是一道题，仅输出一个json格式的结果，不要因为题目信息的缺失而不输出内容。
    """
    return sys_prompt,prompt
