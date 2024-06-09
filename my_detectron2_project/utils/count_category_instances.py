import json
from collections import defaultdict

# JSON 파일 경로
train_data_json_file = '../pole_dataset/train/instances_default.json'
test_data_json_file = '../pole_dataset/test/instances_default.json'


def count_category_instances(json_file):
    category_instance_count = defaultdict(int)

    # JSON 파일을 로드하고 카테고리 별 인스턴스 수를 계산
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

        # 각 어노테이션의 카테고리 ID를 조사하여 인스턴스 수를 증가시킴
        for annotation in data['annotations']:
            category_id = annotation['category_id']
            category_instance_count[category_id] += 1

    # 카테고리 ID를 이름으로 변환
    category_names = {}
    for category in data['categories']:
        category_names[category['id']] = category['name']

    return category_instance_count, category_names


def print_category_instances(json_file):
    category_instance_count, category_names = count_category_instances(json_file)
    total_instances = sum(category_instance_count.values())  # 총 인스턴스 수 계산

    print(f"File: {json_file}")
    for category_id, count in category_instance_count.items():
        percentage = (count / total_instances) * 100  # 카테고리별 비율 계산
        print(f"Category: {category_names[category_id]}, Instances: {count}, Percentage: {percentage:.2f}%")
    print(f"Total instances: {total_instances}\n")  # 총 인스턴스 수 출력


# Train 데이터의 카테고리 별 인스턴스 수와 비율 출력
print("Train Data:")
print_category_instances(train_data_json_file)

# Test 데이터의 카테고리 별 인스턴스 수와 비율 출력
print("Test Data:")
print_category_instances(test_data_json_file)
