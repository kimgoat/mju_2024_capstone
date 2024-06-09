import json

json_files = ['instances_default_484.json', 'instances_default_486.json', 'instances_default_489.json', 'instances_default_492.json']


# 병합된 데이터를 저장할 초기 딕셔너리
merged_data = {
    "licenses": [{"name": "", "id": 0, "url": ""}],
    "info": {"contributor": "", "date_created": "", "description": "", "url": "", "version": "", "year": ""},
    "categories": [
        {"id": 1, "name": "폴리머현수", "supercategory": ""},
        {"id": 2, "name": "접속개소", "supercategory": ""},
        {"id": 3, "name": "LA", "supercategory": ""},
        {"id": 4, "name": "TR", "supercategory": ""},
        {"id": 5, "name": "폴리머LP", "supercategory": ""}
    ],
    "images": [],
    "annotations": []
}

# 이미지와 어노테이션 ID의 시작값
current_image_id = 1
current_annotation_id = 1

# JSON 파일을 로드하고 병합
for json_file in json_files:
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

        # 이미지 섹션 병합
        for image in data['images']:
            image['id'] = current_image_id
            merged_data['images'].append(image)
            current_image_id += 1

        # 어노테이션 섹션 병합
        for annotation in data['annotations']:
            annotation['id'] = current_annotation_id
            # 새로운 이미지 ID를 사용하여 어노테이션의 image_id를 업데이트
            annotation['image_id'] = annotation['image_id'] + (current_image_id - len(data['images']) - 1)
            merged_data['annotations'].append(annotation)
            current_annotation_id += 1

# 병합된 데이터를 새로운 JSON 파일로 저장
with open('instances_default.json', 'w', encoding='utf-8') as outfile:
    json.dump(merged_data, outfile, ensure_ascii=False, indent=4)


