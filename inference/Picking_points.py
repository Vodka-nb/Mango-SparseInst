import cv2
import numpy as np
import os
import math
import traceback
from skimage.morphology import skeletonize
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fill_labels_with_white(mask_img):
    """
    #检测掩膜图上截断主干枝的矩形标签，并用白色填充
    """
    result = mask_img.copy()
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    
    # 边缘检测寻找文本标签的外框
    edges = cv2.Canny(gray, 50, 150)
    
    # 膨胀边缘使其连通成矩形块
    kernel = np.ones((5, 15), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # 查找轮廓
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # 标签框通常是宽度较大、高度适中的矩形
        if w > 20 and 10 < h < 80:
            # 用纯白色填充整个标签区域
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 255, 255), -1)
            
    return result

def draw_horizontal_guide_line(img, harvest_point, label_pos):
    if label_pos[0] > harvest_point[0]:  # 标签在右侧
        end_point = (label_pos[0] - 15, harvest_point[1])
        start_point = (harvest_point[0] + 25, harvest_point[1])
        cv2.line(img, start_point, end_point, (0, 0, 255), 3)
    else:  # 标签在左侧
        end_point = (label_pos[0] + 15, harvest_point[1])
        start_point = (harvest_point[0] - 25, harvest_point[1])
        cv2.line(img, start_point, end_point, (0, 0, 255), 3)


def draw_transparent_label(img, text, position):
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.0
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    label_pos = (position[0] + 150, position[1])
    if label_pos[0] + text_size[0] > img.shape[1]:
        label_pos = (position[0] - text_size[0] - 150, position[1])
    if label_pos[1] - text_size[1] < 0:
        label_pos = (label_pos[0], position[1] + 50)
    if label_pos[1] + text_size[1] > img.shape[0]:
        label_pos = (label_pos[0], position[1] - 50)

    overlay = img.copy()
    cv2.rectangle(overlay,
                  (label_pos[0] - 15, label_pos[1] - text_size[1] - 15),
                  (label_pos[0] + text_size[0] + 15, label_pos[1] + 15),
                  (0, 0, 0), -1)
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    cv2.putText(img, text, label_pos, font, font_scale, (255, 255, 255), thickness)
    return label_pos


def calculate_curvature(skeleton, point, window=10):
    x, y = point
    points = []

    # 向前追踪
    cx, cy = x, y
    for _ in range(window):
        found = False
        for dx, dy in [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < skeleton.shape[1] and 0 <= ny < skeleton.shape[0] and skeleton[ny, nx] > 0:
                points.append((nx, ny))
                cx, cy = nx, ny
                found = True
                break
        if not found: break

    # 向后追踪
    cx, cy = x, y
    for _ in range(window):
        found = False
        for dx, dy in [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < skeleton.shape[1] and 0 <= ny < skeleton.shape[0] and skeleton[ny, nx] > 0:
                points.append((nx, ny))
                cx, cy = nx, ny
                found = True
                break
        if not found: break

    # 计算曲率
    if len(points) > 2:
        points = np.array(points)
        x_vals = points[:, 0]
        y_vals = points[:, 1]
        A = np.vstack([x_vals, np.ones(len(x_vals))]).T
        m, c = np.linalg.lstsq(A, y_vals, rcond=None)[0]
        distances = np.abs(y_vals - (m * x_vals + c)) / np.sqrt(m ** 2 + 1)
        return np.mean(distances)
    return 0


def calculate_energy(width, dist_junction, dist_fruit, curvature,
                     width_weight=0.65, dist_weight=0.1, fruit_weight=0.05, curve_weight=0.2):
    """计算能量函数，优先选择细枝干位置"""
    width_penalty = min(1.0, width / 25.0) if width > 0 else 1.0

    dist_penalty = 0.0
    if dist_junction < 280:
        dist_penalty = min(1.0, (280 - dist_junction) / 280)
    elif dist_junction > 380:
        dist_penalty = min(1.0, (dist_junction - 380) / 380)

    fruit_penalty = min(0.5, dist_fruit / 200.0) if dist_fruit > 0 else 0.0
    curve_penalty = min(1.0, curvature * 15) if curvature > 0 else 0.0

    return (
            width_weight * width_penalty +
            dist_weight * dist_penalty +
            fruit_weight * fruit_penalty +
            curve_weight * curve_penalty
    )


def get_min_axis_width(component, px, py):
    if py >= component.shape[0] or px >= component.shape[1] or component[py, px] == 0:
        return float('inf')

    row = py
    col_start, col_end = px, px
    while col_start > 0 and component[row, col_start] > 0: col_start -= 1
    while col_end < component.shape[1] - 1 and component[row, col_end] > 0: col_end += 1
    width_h = col_end - col_start

    col = px
    row_start, row_end = py, py
    while row_start > 0 and component[row_start, col] > 0: row_start -= 1
    while row_end < component.shape[0] - 1 and component[row_end, col] > 0: row_end += 1
    width_v = row_end - row_start

    return min(width_h, width_v)


def get_local_mask_area(component, px, py, radius=3):
    if py >= component.shape[0] or px >= component.shape[1]:
        return float('inf')

    y_min = max(0, py - radius)
    y_max = min(component.shape[0], py + radius + 1)
    x_min = max(0, px - radius)
    x_max = min(component.shape[1], px + radius + 1)

    roi = component[y_min:y_max, x_min:x_max]
    return cv2.countNonZero(roi)


def is_y_junction(skeleton, x, y):
    if y >= skeleton.shape[0] or x >= skeleton.shape[1] or skeleton[y, x] == 0:
        return False

    top_found, left_found, right_found = False, False, False

    for dy in range(1, 15):
        if y - dy < 0: break
        if skeleton[y - dy, x] > 0: top_found = True; break

    for dx in range(1, 15):
        if x - dx < 0: break
        if skeleton[y, x - dx] > 0: left_found = True; break

    for dx in range(1, 15):
        if x + dx >= skeleton.shape[1]: break
        if skeleton[y, x + dx] > 0: right_found = True; break

    return top_found and (left_found and right_found)


def trace_upwards_from_junction(skeleton, junction_point, max_trace_length=400):
    x, y = junction_point
    path = [(x, y)]
    visited = set([(x, y)])

    primary_directions = [(0, -1), (-1, -1), (1, -1), (0, -2), (-1, -2), (1, -2)]
    secondary_directions = [(-1, 0), (1, 0), (-2, 0), (2, 0)]
    current_x, current_y = x, y

    for step in range(max_trace_length):
        next_point = None
        for dx, dy in primary_directions:
            nx, ny = current_x + dx, current_y + dy
            if (0 <= nx < skeleton.shape[1] and 0 <= ny < skeleton.shape[0] and
                    skeleton[ny, nx] > 0 and (nx, ny) not in visited):
                next_point = (nx, ny)
                break

        if next_point is None:
            for dx, dy in secondary_directions:
                nx, ny = current_x + dx, current_y + dy
                if (0 <= nx < skeleton.shape[1] and 0 <= ny < skeleton.shape[0] and
                        skeleton[ny, nx] > 0 and (nx, ny) not in visited):
                    next_point = (nx, ny)
                    break

        if next_point:
            current_x, current_y = next_point
            path.append(next_point)
            visited.add(next_point)
        else:
            found = False
            for radius in range(1, 5):
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        if dx == 0 and dy == 0: continue
                        nx, ny = current_x + dx, current_y + dy
                        if (0 <= nx < skeleton.shape[1] and 0 <= ny < skeleton.shape[0] and
                                skeleton[ny, nx] > 0 and (nx, ny) not in visited):
                            next_point = (nx, ny)
                            found = True
                            break
                    if found: break
                if found: break

            if found:
                current_x, current_y = next_point
                path.append(next_point)
                visited.add(next_point)
            else:
                break

    return path


def find_optimal_harvest_point(component, skeleton, junction_point, fruit_point, search_range=400):
    path = trace_upwards_from_junction(skeleton, junction_point, max_trace_length=search_range)

    # 优化1：严格确保采摘点在最顶部分叉点的上方（y值更小，至少预留30像素空间）
    valid_path = [pt for pt in path if pt[1] < junction_point[1] - 30]

    if not valid_path or len(valid_path) < 10:
        return (junction_point[0], max(0, junction_point[1] - 60))

    min_energy = float('inf')
    best_point = valid_path[-1]

    for i in range(len(valid_path)):
        px, py = valid_path[i]
        if py >= component.shape[0] or px >= component.shape[1]: continue

        try:
            row = component[py, :]
            if np.count_nonzero(row) == 0: continue

            mask_indices = np.where(row > 0)[0]
            center_x = int((mask_indices.min() + mask_indices.max()) // 2)
            center_y = py
            cx, cy = center_x, center_y

            if not (0 <= cx < component.shape[1] and component[cy, cx] > 0): continue

            width = get_min_axis_width(component, cx, cy)
            dist_junction = math.sqrt((cx - junction_point[0]) ** 2 + (cy - junction_point[1]) ** 2)
            if dist_junction < 280: continue
            dist_fruit = math.sqrt((cx - fruit_point[0]) ** 2 + (cy - fruit_point[1]) ** 2)
            curvature = calculate_curvature(skeleton, (cx, cy))

            energy = calculate_energy(width, dist_junction, dist_fruit, curvature,
                                      width_weight=0.7, dist_weight=0.05, fruit_weight=0.05, curve_weight=0.2)

            if energy < min_energy:
                min_energy = energy
                best_point = (cx, cy)
        except Exception as e:
            continue

    return best_point

def find_upper_junction(skeleton, fruit_point):
    if fruit_point[1] >= skeleton.shape[0] or fruit_point[0] >= skeleton.shape[1]: return None
    height, width = skeleton.shape
    current_y = max(0, fruit_point[1] - 1)
    best_junction = None
    max_distance = 0

    while current_y >= 0:
        for x in range(width):
            if skeleton[current_y, x] > 0:
                if is_y_junction(skeleton, x, current_y):
                    distance = fruit_point[1] - current_y
                    if distance > max_distance:
                        max_distance = distance
                        best_junction = (x, current_y)
        current_y -= 1

    return best_junction


def process_images(mask_path, original_path, output_dir):
    img_key = os.path.splitext(os.path.basename(original_path))[0]
    img_output_dir = os.path.join(output_dir, f"{img_key}_processing")
    os.makedirs(img_output_dir, exist_ok=True)
    logger.info(f"处理图像: {img_key}, 结果保存在: {img_output_dir}")

    #  读取图像并处理截断标签
    mask_img = cv2.imread(mask_path)
    mask_img = fill_labels_with_white(mask_img)
    
    hsv = cv2.cvtColor(mask_img, cv2.COLOR_BGR2HSV)
    original_img = cv2.imread(original_path)

    if mask_img is None or original_img is None:
        logger.error("错误: 无法读取图像")
        return []

    cv2.imwrite(os.path.join(img_output_dir, f"{img_key}_original.png"), original_img)
    cv2.imwrite(os.path.join(img_output_dir, f"{img_key}_mask_repaired.png"), mask_img)

    # 主干枝提取与形态学加强缝合
    lower_green = np.array([35, 100, 50])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # 稍微膨胀白色掩膜，确保它和绿色掩膜的边缘完美咬合，不留1像素的缝隙
    white_mask = cv2.dilate(white_mask, np.ones((5, 5), np.uint8), iterations=1)
    green_mask = cv2.bitwise_or(green_mask, white_mask)
    
    # 将闭运算核放大，增强全局连通性，弥合因为光照等导致的小断裂
    kernel_close = np.ones((15, 15), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel_close)
    cv2.imwrite(os.path.join(img_output_dir, f"{img_key}_green_mask.png"), green_mask)

    # 3. 果实区域检测
    lower_fruit1 = np.array([0, 100, 50])
    upper_fruit1 = np.array([20, 255, 255])
    lower_fruit2 = np.array([20, 100, 50])
    upper_fruit2 = np.array([35, 255, 255])
    fruit_mask = cv2.bitwise_or(cv2.inRange(hsv, lower_fruit1, upper_fruit1), 
                                cv2.inRange(hsv, lower_fruit2, upper_fruit2))
    cv2.imwrite(os.path.join(img_output_dir, f"{img_key}_fruit_mask.png"), fruit_mask)

    # 4. 果实-主干连接区域
    kernel = np.ones((25, 25), np.uint8)
    expanded_fruit = cv2.dilate(fruit_mask, kernel, iterations=1)
    seed_mask = cv2.bitwise_and(green_mask, expanded_fruit)
    cv2.imwrite(os.path.join(img_output_dir, f"{img_key}_seed_mask.png"), seed_mask)

    # 5. 提取主干枝组件（加入动态面积过滤，剔除噪点）
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(green_mask, connectivity=8)
    final_stem_mask = np.zeros_like(green_mask)
    stem_components = []
    stem_centroids = []
    
    # 用 candidates 暂存碰到了果实的连通域
    candidates = []
    for label in range(1, num_labels):
        component_mask = (labels == label).astype(np.uint8) * 255
        if cv2.countNonZero(cv2.bitwise_and(component_mask, seed_mask)) > 0:
            area = stats[label, cv2.CC_STAT_AREA]
            candidates.append((area, component_mask, centroids[label].astype(int)))
            
    # 按面积从大到小排序
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # 动态过滤：只保留最大的主干，或者面积至少达到最大主干20%且大于500的连通域。
    # 这样就能彻底屏蔽掉碰到了果实掩膜、但面积只有一百多的碎叶子和噪点。
    if candidates:
        max_area = candidates[0][0]
        for area, comp, cent in candidates:
            if area > max_area * 0.2 and area > 500:
                final_stem_mask = cv2.bitwise_or(final_stem_mask, comp)
                stem_components.append(comp)
                stem_centroids.append(cent)

    cv2.imwrite(os.path.join(img_output_dir, f"{img_key}_stem_mask.png"), final_stem_mask)

    #  主干处理核心
    harvest_points_img = original_img.copy()
    harvest_points = []
    debug_img = original_img.copy()

    for idx, (component, centroid) in enumerate(zip(stem_components, stem_centroids)):
        best_point_candidate = None
        
        try:
            stem_seed_mask = cv2.bitwise_and(component, seed_mask)
            seed_points = cv2.findNonZero(stem_seed_mask)

            if seed_points is None or seed_points.size == 0:
                logger.info(f"主干枝{idx}无果实连接点，使用质心寻找")
                try:
                    kernel_dilate = np.ones((3, 3), np.uint8)
                    component_dilated = cv2.dilate(component, kernel_dilate, iterations=1)
                    skeleton = skeletonize(component_dilated.astype(bool)).astype(np.uint8) * 255
                    skeleton_8u = skeleton.astype(np.uint8)
                    harvest_point = find_optimal_harvest_point_without_fruit(component, skeleton_8u, centroid)
                except:
                    harvest_point = None

                if harvest_point is None:
                    harvest_point = (int(centroid[0]), int(centroid[1]))
                best_point_candidate = harvest_point

            else:
                fruit_x = int(np.mean(seed_points[:, :, 0]))
                fruit_y = int(np.mean(seed_points[:, :, 1]))
                fruit_point = (fruit_x, fruit_y)

                kernel_dilate = np.ones((3, 3), np.uint8)
                component_dilated = cv2.dilate(component, kernel_dilate, iterations=1)
                try:
                    skeleton = skeletonize(component_dilated.astype(bool)).astype(np.uint8) * 255
                except:
                    skeleton = component
                skeleton_8u = skeleton.astype(np.uint8)

                junction_point = find_upper_junction(skeleton_8u, fruit_point)

                if junction_point:
                    logger.info(f"主干枝{idx}: 找到最顶部分叉点 {junction_point}")
                    best_point = find_optimal_harvest_point(component, skeleton_8u, junction_point, fruit_point)
                else:
                    best_point = (int(centroid[0]), int(centroid[1]))
                    logger.info(f"主干枝{idx}: 未找到分叉点，使用质心")
                
                best_point_candidate = best_point

        except Exception as e:
            logger.error(f"主干枝{idx}处理出错: {str(e)}")
            best_point_candidate = (int(centroid[0]), int(centroid[1]))

        # 使用距离阈值（120像素）去重，防截断导致的二次识别
        if best_point_candidate:
            is_duplicate = False
            for existing_pt in harvest_points:
                if math.hypot(best_point_candidate[0] - existing_pt[0], best_point_candidate[1] - existing_pt[1]) < 120:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                text = f"({best_point_candidate[0]}, {best_point_candidate[1]})"
                label_pos = draw_transparent_label(harvest_points_img, text, best_point_candidate)
                draw_horizontal_guide_line(harvest_points_img, best_point_candidate, label_pos)
                cv2.circle(harvest_points_img, best_point_candidate, 40, (20, 230, 240), 8)
                cv2.circle(harvest_points_img, best_point_candidate, 30, (144, 238, 144), 6)
                cv2.circle(harvest_points_img, best_point_candidate, 15, (255, 255, 255), -1)
                harvest_points.append(best_point_candidate)

    # 7. 保存结果
    result_path = os.path.join(img_output_dir, f"{img_key}_harvest_points.jpg")
    cv2.imwrite(result_path, harvest_points_img)
    logger.info(f"处理完成！找到{len(harvest_points)}个采摘点")
    return harvest_points

def batch_process_images(mask_folder, orig_folder, output_parent_dir):
    os.makedirs(output_parent_dir, exist_ok=True)
    mask_files = [f for f in os.listdir(mask_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_images = len(mask_files)
    logger.info(f"找到 {total_images} 张掩膜图像，开始批量处理...")
    
    for i, mask_file in enumerate(mask_files, 1):
        mask_path = os.path.join(mask_folder, mask_file)
        orig_file = mask_file
        orig_path = os.path.join(orig_folder, orig_file)
        
        if not os.path.exists(orig_path):
            continue
            
        try:
            harvest_points = process_images(mask_path, orig_path, output_parent_dir)
            result_file = os.path.join(output_parent_dir, f"{os.path.splitext(mask_file)[0]}_points.txt")
            with open(result_file, 'w') as f:
                f.write(f"Image: {mask_file}\nHarvest Points: {harvest_points}\n")
        except Exception as e:
            logger.error(f"❌ 处理失败: {mask_file} | {str(e)}")

if __name__ == "__main__":
    mask_folder = "xxx"
    orig_folder = "xxx"
    output_parent_dir = "xxx"
    batch_process_images(mask_folder, orig_folder, output_parent_dir)
