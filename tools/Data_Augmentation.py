"""
Mango Dataset Augmentation with Original Preservation
Created: 2025-03-03
Author: AI Assistant
"""
import os
import json
import cv2
import shutil
import albumentations as A
import numpy as np
from datetime import datetime
from tqdm import tqdm

# ========== 配置区 ==========
class Config:
    # 输入输出路径
    image_dir = "/home/xxx/Data/Mango_1300_V3/train/images"
    json_dir = "/home/xxx/Data/Mango_1300_V3/train/annotations"
    dest_dir = "/home/xxx/Data/Mango_1300_V3_Aug/train"

    # 增强参数
    augmentation_times = 7  # 每张图生成的增强版本数（不含原始图）
    min_polygon_area = 30  # 多边形最小有效面积
    epsilon = 1e-6  # 浮点精度阈值

    # 增强流水线配置
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.4),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.03,
            rotate_limit=15,
            interpolation=cv2.INTER_NEAREST,  # 保持边缘锐利
            border_mode=cv2.BORDER_REPLICATE,  # 边缘复制代替填充
            p=0.4
        ),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussianBlur(blur_limit=(2, 4), p=0.3),
        A.CLAHE(p=0.2),
        A.ColorJitter(
            brightness=0.11,  # 亮度波动±10%(原0.2)
            contrast=0.11,  # 对比度波动±10%
            saturation=0.05,  # 饱和度波动±5%
            hue=0.02,  # 色相波动±2度
            p=0.5
        ),
        A.ToGray(p=0.05),  # 低概率灰度化
        A.ChannelShuffle(p=0.05)  # 通道混洗
    ], keypoint_params=A.KeypointParams(
        format='xy',
        remove_invisible=False, #保留不可见关键点
        angle_in_degrees=True
    ))

# ========== 核心函数 ==========
def validate_polygon(points: list, min_area: float) -> bool:
    """验证多边形有效性（基于Shoelace公式）"""
    if len(points) < 3:
        return False
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return area >= min_area

def safe_convert(value):
    """安全转换numpy类型为Python原生类型"""
    if isinstance(value, np.generic):
        return value.item()
    return value

def process_annotation(original_ann: dict, keypoints: list, new_size: tuple) -> dict:
    """标注文件转换（包含类型转换和精度处理）"""
    new_shapes = []
    idx = 0

    # 确保new_size为原生int类型
    new_size = (int(new_size[0]), int(new_size[1]))

    for shape in original_ann["shapes"]:
        # 新增标注结构验证
        if "shapes" not in original_ann:
            print(f"⚠️ 无效标注结构：缺少shapes字段")
            return None
        if not isinstance(original_ann["shapes"], list):
            print(f"⚠️ 无效标注结构：shapes字段非列表类型")
            return None
        num_points = len(shape["points"])
        new_points = keypoints[idx:idx + num_points]
        idx += num_points

        valid_points = []
        for p in new_points:
            # 坐标裁剪和类型转换
            x = np.clip(p[0], 0, new_size[0] - 1)
            y = np.clip(p[1], 0, new_size[1] - 1)

            # 处理浮点精度误差
            x = safe_convert(x)
            y = safe_convert(y)
            x = x if abs(x) > Config.epsilon else 0.0
            y = y if abs(y) > Config.epsilon else 0.0

            valid_points.append([x, y])

        if validate_polygon(valid_points, Config.min_polygon_area):
            new_shapes.append({
                "label": shape["label"],
                "points": valid_points,
                "group_id": safe_convert(shape.get("group_id")),
                "shape_type": "polygon",
                "flags": shape.get("flags", {})
            })

    return {
        "version": original_ann["version"],
        "flags": original_ann["flags"],
        "imagePath": original_ann["imagePath"],  # 后续会覆盖
        "imageData": None,
        "imageHeight": new_size[1],
        "imageWidth": new_size[0],
        "shapes": new_shapes
    }

# ========== 主流程 ==========
def main():
    # 创建目录结构
    os.makedirs(os.path.join(Config.dest_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(Config.dest_dir, "labels", "json"), exist_ok=True)

    # 获取原始文件列表
    image_files = [f for f in os.listdir(Config.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in tqdm(image_files):
        # 原始文件路径
        image_path = os.path.join(Config.image_dir, image_file)
        json_base = os.path.splitext(image_file)[0]
        json_path = os.path.join(Config.json_dir, json_base + ".json")

        # 新增文件校验逻辑
        if not os.path.exists(json_path):
            print(f"⚠️ 标注文件缺失：{json_path}，跳过该图像处理")
            continue
        if os.path.getsize(json_path) == 0:
            print(f"⛔ 空标注文件：{json_path}，跳过处理")
            continue
        # 读取原始数据
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        if image is None:
            print(f"无法读取图像文件：{image_path}")
            continue
        # 修改后：
        try:
            with open(json_path, 'r', encoding='utf-8-sig') as f:  # 处理BOM头[4,8](@ref)
                raw_content = f.read().strip()
                if not raw_content:
                    raise ValueError("文件内容为空")
                original_ann = json.loads(raw_content)
        except json.JSONDecodeError as e:
            print(f"❌ JSON格式错误：{json_path}\n错误详情：{str(e)}\n错误上下文：{raw_content[:200]}...")
            continue
        except Exception as e:
            print(f"❌ 未知错误：{json_path}\n错误类型：{type(e).__name__}\n错误信息：{str(e)}")
            continue

        # 准备关键点数据
        original_keypoints = []
        for shape in original_ann["shapes"]:
            original_keypoints.extend([[p[0], p[1]] for p in shape["points"]])

        # 生成增强版本
        for aug_idx in range(Config.augmentation_times + 1):
            try:
                # 应用数据增强
                if aug_idx == 0:  # 保留原始版本
                    transformed_image = image
                    transformed_keypoints = original_keypoints
                else:
                    augmented = Config.transform(image=image, keypoints=original_keypoints)
                    transformed_image = augmented["image"]
                    transformed_keypoints = augmented["keypoints"]

                # 转换标注信息
                new_ann = process_annotation(
                    original_ann=original_ann,
                    keypoints=transformed_keypoints,
                    new_size=(transformed_image.shape[1], transformed_image.shape[0])
                )

                # 保存增强结果
                suffix = f"_{aug_idx}" if aug_idx > 0 else ""
                new_filename = os.path.splitext(image_file)[0] + suffix + ".jpg"

                # 保存图像
                cv2.imwrite(
                    os.path.join(Config.dest_dir, "images", new_filename),
                    cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
                )

                # 保存标注
                new_ann["imagePath"] = new_filename
                with open(os.path.join(Config.dest_dir, "labels", "json", new_filename.replace(".jpg", ".json")), "w") as f:
                    json.dump(new_ann, f, indent=2)

            except Exception as e:
                print(f"Error processing {image_file} augmentation {aug_idx}: {str(e)}")
                continue

if __name__ == "__main__":
    main()
