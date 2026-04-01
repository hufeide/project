import os
import json
import pandas as pd
from glob import glob

RESULT_DIR = '/data/weidu_new/code_25/0703/dfjg_chinese_rec_v1/Template/exam_item_analysis/project/data/result'
OUTPUT_DIR = '/data/weidu_new/code_25/0703/dfjg_chinese_rec_v1/Template/exam_item_analysis/project/check/data'

FOLDERS = [
    'answer_analysis',
    'answer_correct',
    'answer_correct_gen',
    'answer_knowledge',
    'answer_knowledge_gen'
]


def extract_fields_from_json(json_path, folder_name):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    record = {'folder': folder_name}

    original_data = data.get('original_data', {})
    for key in original_data:
        if key != 'image_list':
            record[f'original_{key}'] = original_data.get(key)

    results = data.get('results', {})
    
    for model_key, model_result in results.items():
        prefix = f"{model_key}_"
        for k, v in model_result.items():
            if k != 'model_name':
                record[f'{prefix}{k}'] = v

    record['uuid'] = data.get('uuid', '')

    return record


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for folder in FOLDERS:
        folder_path = os.path.join(RESULT_DIR, folder)
        if not os.path.exists(folder_path):
            print(f"文件夹不存在: {folder_path}")
            continue

        json_files = glob(os.path.join(folder_path, '*.json'))
        print(f"处理文件夹: {folder}, 文件数: {len(json_files)}")

        records = []
        for json_file in json_files:
            try:
                record = extract_fields_from_json(json_file, folder)
                records.append(record)
            except Exception as e:
                print(f"处理文件失败 {json_file}: {e}")

        if records:
            df = pd.DataFrame(records)
            output_path = os.path.join(OUTPUT_DIR, f'{folder}.xlsx')
            df.to_excel(output_path, index=False, engine='openpyxl')
            print(f"  保存到 {output_path}")


if __name__ == '__main__':
    main()