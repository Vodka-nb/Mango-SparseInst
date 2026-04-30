import os
import json
import random
import shutil
from PIL import Image

# 设置原始数据目录
data_images_dir = '/home/xxx/Data/Mango_Group_Aug/images'
data_annotations_dir = '/home/xxx/Data/Mango_Group_Aug/labels/json'
output_dir = '/home/xxx/Data/MangoGroup_Augmented_Data_split'

# 数据集划分比例 (train:val:test = 8:1:1)
split_ratios = {'train': 0.8, 'val': 0.1, 'test': 0.1}
random_seed = 42  # 固定随机种子确保可重复性

# 创建输出子目录结构
output_subdirs = ['train', 'val', 'test']
for subdir in output_subdirs:
    os.makedirs(os.path.join(output_dir, subdir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, subdir, 'annotations'), exist_ok=True)

# 初始化 COCO 数据结构（每个子集独立）
coco_data = {
    'train': {"images": [], "annotations": [], "categories": []},
    'val': {"images": [], "annotations": [], "categories": []},
    'test': {"images": [], "annotations": [], "categories": []}
}
category_map = {}
category_id = 1

# 获取所有 JSON 文件并随机划分
all_json_files = [f for f in os.listdir(data_annotations_dir) if f.endswith('.json')]
random.Random(random_seed).shuffle(all_json_files)

# 计算划分点
total = len(all_json_files)
train_end = int(total * split_ratios['train'])
val_end = train_end + int(total * split_ratios['val'])
splits = {
    'train': all_json_files[:train_end],
    'val': all_json_files[train_end:val_end],
    'test': all_json_files[val_end:]
}

# 处理每个子集
for split_name, split_files in splits.items():
    for json_file in split_files:
        # 读取 JSON 文件
        json_path = os.path.join(data_annotations_dir, json_file)
        with open(json_path, 'r') as f:
            data = json.load(f)

        # 获取图片路径
        image_filename = data['imagePath']
        original_image_path = os.path.join(data_images_dir, image_filename)
        if not os.path.exists(original_image_path):
            print(f"跳过 {image_filename}，图片文件不存在")
            continue

        # 读取实际图片尺寸
        with Image.open(original_image_path) as img:
            actual_width, actual_height = img.size

        # 检查并校正标注坐标
        json_width = data.get('imageWidth', actual_width)
        json_height = data.get('imageHeight', actual_height)
        if (json_width, json_height) != (actual_width, actual_height):
            scale_x = actual_width / json_width
            scale_y = actual_height / json_height
            for shape in data['shapes']:
                points = shape['points']
                shape['points'] = [[x * scale_x, y * scale_y] for (x, y) in points]
            print(f"已校正 {json_file} 的标注坐标")

        # 更新 JSON 尺寸并清理冗余字段
        data.update({
            'imageWidth': actual_width,
            'imageHeight': actual_height,
            'imageData': None
        })

        # 保存处理后的文件到对应子集目录
        output_split_dir = os.path.join(output_dir, split_name)
        shutil.copy(original_image_path, os.path.join(output_split_dir, 'images', image_filename))
        with open(os.path.join(output_split_dir, 'annotations', json_file), 'w') as f:
            json.dump(data, f, indent=2)

        # 构建 COCO 数据
        image_id = len(coco_data[split_name]['images']) + 1
        image_info = {
            "id": image_id,
            "file_name": image_filename,
            "width": actual_width,
            "height": actual_height
        }
        coco_data[split_name]['images'].append(image_info)

        # 处理标注
        for shape in data['shapes']:
            label = shape['label']
            if label not in category_map:
                category_map[label] = category_id
                category_id += 1

            # 计算 bbox
            points = shape['points']
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            x_min, y_min = min(x_coords), min(y_coords)
            bbox = [x_min, y_min, max(x_coords) - x_min, max(y_coords) - y_min]

            # 添加标注
            annotation = {
                "id": len(coco_data[split_name]['annotations']) + 1,
                "image_id": image_id,
                "category_id": category_map[label],
                "segmentation": [sum(points, [])],
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0
            }
            coco_data[split_name]['annotations'].append(annotation)

# 统一类别信息
for split in coco_data.values():
    for label, cat_id in category_map.items():
        split['categories'].append({
            "id": cat_id,
            "name": label,
            "supercategory": "none"
        })

# 保存各子集的 COCO 文件
for split_name in split_ratios.keys():
    output_path = os.path.join(output_dir, split_name, f'annotations_{split_name}.json')
    with open(output_path, 'w') as f:
        json.dump(coco_data[split_name], f, indent=2)

print("数据处理完成！数据集已划分为 train/val/test。")
