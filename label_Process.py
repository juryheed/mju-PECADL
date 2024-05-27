import os
import json

# 입력 폴더 경로 설정
input_folder_path = 'D:/download/eye_data/datas/Validation/labels/diseaseA'
# 출력 폴더 경로 설정
output_folder_path = 'D:/download/eye_data/datas/Validation/processed_labels/diseaseA'

# 출력 폴더가 존재하지 않으면 생성
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# 모든 파일에 대해 반복
for filename in os.listdir(input_folder_path):
    if filename.endswith(".json"):
        input_file_path = os.path.join(input_folder_path, filename)

        # JSON 파일 읽기
        with open(input_file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)

        # 'CLASS' 항목만 남기기
        if 'ANNOTATION_INFO' in data and isinstance(data['ANNOTATION_INFO'], list):
            new_data = {
                "ANNOTATION_INFO": [
                    {"CLASS": annotation["CLASS"]}
                    for annotation in data['ANNOTATION_INFO']
                ]
            }
        else:
            new_data = {}

        # 출력 파일 경로 생성
        output_file_path = os.path.join(output_folder_path, filename)

        # 수정된 데이터로 JSON 파일 저장
        with open(output_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(new_data, json_file, indent=4, ensure_ascii=False)

        print(f"Processed file: {output_file_path}")

print("All JSON files have been processed.")
