import gradio as gr
import json
import os

# 获取data目录下的所有JSON文件
data_dir = "/data/weidu_new/code_25/0703/dfjg_chinese_rec_v1/Template/exam_item_analysis/data"
json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]

def load_and_display(file_name):
    # 加载选中的JSON文件
    file_path = os.path.join(data_dir, file_name)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取需要展示的字段
    material = data.get('material', '无')
    question = data.get('question', '无')
    answer = data.get('answer', '无')
    res1 = data.get('res1', {})
    res2 = data.get('res2', {})
    prompt = data.get('prompt', '无')
    uuid = data.get('uuid', '无')
    
    # 格式化展示内容
    material_str = str(material)
    question_str = str(question)
    answer_str = str(answer)
    res1_str = json.dumps(res1, ensure_ascii=False, indent=2)
    res2_str = json.dumps(res2, ensure_ascii=False, indent=2)
    prompt_str = str(prompt)
    uuid_str = str(uuid)
    
    return material_str, question_str, answer_str, res1_str, res2_str, prompt_str, uuid_str

# 创建Gradio界面
with gr.Blocks() as app:
    gr.Markdown("# 试题分析展示")
    
    with gr.Row():
        # 左侧文件选择
        with gr.Column(scale=1):
            file_selector = gr.Dropdown(
                choices=json_files,
                label="选择JSON文件",
                value=json_files[0] if json_files else None
            )
            load_button = gr.Button("加载文件")
        
        # 右侧内容展示
        with gr.Column(scale=3):
            uuid_output = gr.Textbox(label="UUID", lines=1)
            material_output = gr.Textbox(label="材料", lines=5)
            question_output = gr.Textbox(label="问题", lines=5)
            answer_output = gr.Textbox(label="答案", lines=5)
            res1_output = gr.Textbox(label="分析结果1", lines=10)
            res2_output = gr.Textbox(label="分析结果2", lines=10)
            prompt_output = gr.Textbox(label="提示词", lines=10)
    
    # 绑定事件
    load_button.click(
        fn=load_and_display,
        inputs=file_selector,
        outputs=[material_output, question_output, answer_output, res1_output, res2_output, prompt_output, uuid_output]
    )
    
    # 初始化加载第一个文件
    if json_files:
        material_str, question_str, answer_str, res1_str, res2_str, prompt_str, uuid_str = load_and_display(json_files[0])
        material_output.value = material_str
        question_output.value = question_str
        answer_output.value = answer_str
        res1_output.value = res1_str
        res2_output.value = res2_str
        prompt_output.value = prompt_str
        uuid_output.value = uuid_str

# 启动应用
if __name__ == "__main__":
    app.launch(server_name="0.0.0.0",share=True)