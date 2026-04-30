#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
采摘点推理脚本（FPEA高召回精准定位版 v3）
1. 投票机制：分叉点只选得分最高（被经过次数最多），不考虑Y坐标
2. 分叉检测：严格识别"倒Y型"分叉（上方1支 + 下方左右各1支）
3. 水平居中：采摘点确定后，强制调整至该行主干枝的水平中心
"""

import os
import cv2
import numpy as np
import math
import json
import torch
from pathlib import Path
from tqdm import tqdm
from skimage.morphology import skeletonize
from collections import defaultdict
from rfdetr import RFDETRSegNano

# ================= 配置区域 =================
CHECKPOINT_PATH = "/home/xxx/Litchi/model/output/checkpoint_best_ema.pth"
TEST_IMAGE_DIR  = "/home/xxx/Data/litchi_coco_rf_detr/test"
OUTPUT_DIR      = "./litchi_harvest_results"

CONF_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLS_NAMES = ["branch", "fruit", "fruits cluster"]

CLASS_COLOR_MAP = {
    'branch': (0, 255, 255), 'branchs': (0, 255, 255),
    'fruit': (0, 0, 255), 'fruits': (0, 0, 255), 'fruits cluster': (0, 0, 255)
}
DEFAULT_COLOR = (255, 0, 0)

LOWER_YELLOW, UPPER_YELLOW = (0, 200, 200), (50, 255, 255)
LOWER_RED, UPPER_RED = (0, 0, 200), (100, 100, 255)

# FPEA能量权重
W_WIDTH = 0.55
W_CURVE = 0.25
W_DIST  = 0.20

# 位置约束
MIN_DIST_JUNC = 25
MAX_DIST_JUNC = 320
MIN_WIDTH = 6
RESCUE_RADIUS = 50
BRANCH_TRACE_LEN = 12  # 分叉分支追踪长度

FOLDERS = {
    'original': os.path.join(OUTPUT_DIR, '0_original'),
    'color_mask': os.path.join(OUTPUT_DIR, '1_color_mask'),
    'binary_mask': os.path.join(OUTPUT_DIR, '2_binary_mask'),
    'result': os.path.join(OUTPUT_DIR, '3_harvest_points')
}

# ================= 核心几何函数 =================
def get_horizontal_width(mask, pt):
    x, y = pt
    h, w = mask.shape
    if not (0 <= x < w and 0 <= y < h and mask[y, x] > 0):
        return np.inf
    lx, rx = x, x
    while lx > 0 and mask[y, lx-1] > 0: lx -= 1
    while rx < w-1 and mask[y, rx+1] > 0: rx += 1
    return rx - lx + 1

def snap_to_horizontal_center(mask, pt):
    """将点调整至该行的水平中心（宽度中点）"""
    x, y = pt
    h, w = mask.shape
    if not (0 <= y < h):
        return pt
    
    # 获取该行的所有掩膜像素
    row = mask[y, :]
    indices = np.where(row > 0)[0]
    
    if len(indices) > 0:
        # 计算水平中心（中位数或平均值）
        center_x = int(np.median(indices))
        return (center_x, y)
    return pt

def calc_local_curvature(skel, pt):
    x, y = pt
    pts = [(x, y)]
    for dx, dy in [(0,-1),(-1,-1),(1,-1),(-1,0),(1,0),(0,1),(-1,1),(1,1)]:
        nx, ny = x+dx, y+dy
        if 0 <= nx < skel.shape[1] and 0 <= ny < skel.shape[0] and skel[ny, nx] > 0:
            pts.append((nx, ny)); break
    if len(pts) < 3: return 0.0
    pts = np.array(pts)
    A = np.vstack([pts[:, 0], np.ones(len(pts))]).T
    try:
        m, c = np.linalg.lstsq(A, pts[:, 1], rcond=None)[0]
        return np.mean(np.abs(pts[:, 1] - (m * pts[:, 0] + c)) / np.sqrt(m**2 + 1))
    except: return 0.0

def fpea_energy(width, curvature, dist_junc):
    w_pen = min(1.0, max(0, (width - MIN_WIDTH) / 45.0))
    c_pen = min(1.0, curvature * 10.0)
    if dist_junc < MIN_DIST_JUNC:
        d_pen = 1.0 - (dist_junc / MIN_DIST_JUNC)
    elif dist_junc > MAX_DIST_JUNC:
        d_pen = min(1.0, (dist_junc - MAX_DIST_JUNC) / 200.0)
    else:
        d_pen = 0.0
    return W_WIDTH * w_pen + W_CURVE * c_pen + W_DIST * d_pen

def trace_main_branch_up(skel, start_pt, max_len=400):
    x, y = start_pt
    path = [(x, y)]
    visited = {(x, y)}
    curr = (x, y)
    h, w = skel.shape
    dirs = [(0,-1), (-1,-1), (1,-1), (-1,0), (1,0), (0,1), (-1,1), (1,1)]
    for _ in range(max_len):
        nxt = None
        for dx, dy in dirs:
            nx, ny = curr[0]+dx, curr[1]+dy
            if 0<=nx<w and 0<=ny<h and skel[ny, nx]>0 and (nx, ny) not in visited:
                nxt = (nx, ny); break
        if nxt:
            curr = nxt; path.append(curr); visited.add(curr)
        else:
            break
    return path

def trace_branch_direction(skel, x, y, dx_dir, dy_dir, max_len=BRANCH_TRACE_LEN):
    """
    沿指定方向追踪骨架分支
    返回：追踪到的像素数
    """
    h, w = skel.shape
    cx, cy = x + dx_dir, y + dy_dir
    count = 0
    
    for _ in range(max_len):
        if 0 <= cx < w and 0 <= cy < h and skel[cy, cx] > 0:
            count += 1
            cx += dx_dir
            cy += dy_dir
        else:
            break
    return count

def find_inverted_y_junctions(skel):
    """
    严格识别"倒Y型"分叉点
    特征：上方1个分支 + 左下方1个分支 + 右下方1个分支
    """
    h, w = skel.shape
    bin_skel = (skel > 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    neighbor_count = cv2.filter2D(bin_skel, -1, kernel)
    
    # 初步筛选：连通度≥4的候选点
    candidates = np.argwhere((neighbor_count >= 3) & (bin_skel == 1))
    
    inverted_y_junctions = []
    
    for y, x in candidates:
        # 检测三个方向的分支存在性
        # 1. 上方分支（Y减小方向）
        up_branch = trace_branch_direction(skel, x, y, 0, -1, BRANCH_TRACE_LEN)
        
        # 2. 左下方分支（X减小，Y增大）
        left_down_branch = trace_branch_direction(skel, x, y, -1, 1, BRANCH_TRACE_LEN)
        
        # 3. 右下方分支（X增大，Y增大）
        right_down_branch = trace_branch_direction(skel, x, y, 1, 1, BRANCH_TRACE_LEN)
        
        # 倒Y型判定：三个方向均有足够长度的分支（≥6像素）
        min_branch_len = BRANCH_TRACE_LEN // 2
        if (up_branch >= min_branch_len and 
            left_down_branch >= min_branch_len and 
            right_down_branch >= min_branch_len):
            inverted_y_junctions.append((x, y))
    
    return inverted_y_junctions

def vote_junctions_upward(skel, fruit_centers, valid_juncs):
    """
    多果实向上投票，只选得分最高（不考虑Y坐标）
    """
    h, w = skel.shape
    junc_set = set(valid_juncs) if valid_juncs else set()
    scores = defaultdict(int)
    trace_dirs = [(0,-1), (-1,-1), (1,-1), (-1,0), (1,0), (0,1), (-1,1), (1,1)]
    
    for fc in fruit_centers:
        # 找果实最近骨架点
        start = None
        for r in range(1, 50):
            for dy in range(-r, r+1):
                for dx in [-r, r] if abs(dy)!=r else range(-r+1, r):
                    nx, ny = int(fc[0])+dx, int(fc[1])+dy
                    if 0<=nx<w and 0<=ny<h and skel[ny, nx]>0:
                        start = (nx, ny); break
                if start: break
            if start: break
        if not start: continue
        
        # 向上追踪投票
        curr = start; visited = set()
        for _ in range(450):
            if curr in visited: break
            visited.add(curr)
            if curr in junc_set:
                scores[curr] += 1  # 每经过一次+1分
            nxt = None
            for dx, dy in trace_dirs:
                nx, ny = curr[0]+dx, curr[1]+dy
                if 0<=nx<w and 0<=ny<h and skel[ny, nx]>0 and (nx, ny) not in visited:
                    nxt = (nx, ny); break
            if nxt: curr = nxt
            else: break
    
    if scores:
        # 直接返回得分最高的节点（被经过次数最多）
        return max(scores, key=scores.get)
    
    # 无分叉点时：返回果实最近骨架点作为备选
    if fruit_centers is not None and len(fruit_centers) > 0:
        fc = tuple(np.mean(fruit_centers, axis=0).astype(int))
        for r in range(1, 60):
            for dy in range(-r, r+1):
                for dx in [-r, r] if abs(dy)!=r else range(-r+1, r):
                    nx, ny = fc[0]+dx, fc[1]+dy
                    if 0<=nx<w and 0<=ny<h and skel[ny, nx]>0:
                        return (nx, ny)
    return None

def project_to_mask_center(mask, pt, radius=15):
    """投影至掩膜局部几何中轴（距离变换极值）"""
    x, y = pt; h, w = mask.shape
    y1, y2 = max(0, y-radius), min(h, y+radius+1)
    x1, x2 = max(0, x-radius), min(w, x+radius+1)
    roi = mask[y1:y2, x1:x2]
    if np.any(roi):
        dist = cv2.distanceTransform(roi, cv2.DIST_L2, 3)
        my, mx = np.unravel_index(np.argmax(dist), dist.shape)
        return (mx + x1, my + y1)
    return pt

def strict_mask_check(mask, pt):
    x, y = pt; h, w = mask.shape
    return 0 <= x < w and 0 <= y < h and mask[y, x] > 0

def rescue_off_mask_point(mask, skel, off_pt):
    """越界救援：最近邻掩膜像素 + 中轴投影"""
    h, w = mask.shape; x, y = off_pt
    nearest = None; min_d = np.inf
    for r in range(1, RESCUE_RADIUS + 1):
        for dy in range(-r, r+1):
            for dx in [-r, r] if abs(dy)!=r else range(-r+1, r):
                nx, ny = x+dx, y+dy
                if 0<=nx<w and 0<=ny<h and mask[ny, nx]>0:
                    d = math.hypot(dx, dy)
                    if d < min_d: min_d, nearest = d, (nx, ny)
            if nearest: break
        if nearest: break
    if nearest is None: return None
    rescued = project_to_mask_center(mask, nearest, radius=12)
    return rescued if strict_mask_check(mask, rescued) else None

def guaranteed_fallback(comp_mask, skel, fruit_pt, junction=None):
    """终极保底策略"""
    h, w = comp_mask.shape
    
    start = junction
    if start is None:
        fx, fy = int(fruit_pt[0]), int(fruit_pt[1])
        for r in range(1, 70):
            for dy in range(-r, r+1):
                for dx in [-r, r] if abs(dy)!=r else range(-r+1, r):
                    nx, ny = fx+dx, fy+dy
                    if 0<=nx<w and 0<=ny<h and skel[ny, nx]>0 and comp_mask[ny, nx]>0:
                        start = (nx, ny); break
                if start: break
            if start: break
    
    if start is None:
        ys, xs = np.where(skel > 0)
        if len(xs) > 0:
            idx = np.argmin(ys)
            start = (int(xs[idx]), int(ys[idx]))
        else:
            return None
    
    path = trace_main_branch_up(skel, start, max_len=450)
    junc_y = junction[1] if junction else start[1]
    
    for pt in path:
        px, py = pt
        if py >= junc_y: continue
        if not strict_mask_check(comp_mask, pt): continue
        width = get_horizontal_width(comp_mask, pt)
        if width > 60: continue
        # 应用水平居中
        final = snap_to_horizontal_center(comp_mask, pt)
        if strict_mask_check(comp_mask, final):
            return final
    
    for pt in path:
        if strict_mask_check(comp_mask, pt):
            return snap_to_horizontal_center(comp_mask, pt)
    
    return snap_to_horizontal_center(comp_mask, start) if strict_mask_check(comp_mask, start) else None

# ================= 可视化模块 =================
def draw_picking_point_aligned(img, pt, text):
    font = cv2.FONT_HERSHEY_SIMPLEX; scale, thick = 0.55, 1
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    pad = 5; box_h = th + 2*pad + bl; box_w = tw + 2*pad
    box_y = pt[1] - box_h // 2; box_x = pt[0] + 35
    if box_x + box_w >= img.shape[1]: box_x = pt[0] - 35 - box_w
    box_y = max(0, min(box_y, img.shape[0] - box_h))
    
    cv2.rectangle(img, (box_x, box_y), (box_x+box_w, box_y+box_h), (0,0,0), -1)
    tx = box_x + pad
    ty = box_y + th + pad + (box_h - th - bl) // 2 - bl//2
    cv2.putText(img, text, (tx, ty), font, scale, (255,255,255), thick, cv2.LINE_AA)
    
    ly = pt[1]
    if pt[0] < box_x: cv2.line(img, (pt[0]+7, ly), (box_x-1, ly), (0,0,255), 2)
    else: cv2.line(img, (box_x+box_w+1, ly), (pt[0]-7, ly), (0,0,255), 2)
        
    cv2.circle(img, pt, 7, (20, 230, 240), 2)
    cv2.circle(img, pt, 3, (255,255,255), -1)

# ================= 主处理流程 =================
def process_image(img_path, model, out_dirs):
    orig = cv2.imread(str(img_path))
    if orig is None: return []
    h, w = orig.shape[:2]; name = Path(img_path).stem
    cv2.imwrite(os.path.join(out_dirs['original'], f"{name}.jpg"), orig)
    
    res = model.predict(str(img_path), threshold=CONF_THRESHOLD)
    det = res[0] if isinstance(res, list) else res
    if not hasattr(det, 'mask') or det.mask is None: return []
    
    # 生成彩色掩膜
    color_mask = np.zeros_like(orig)
    for m, cid in zip(det.mask, det.class_id):
        cname = CLS_NAMES[cid] if cid < len(CLS_NAMES) else 'other'
        color_mask[(m>0.5).astype(bool)] = CLASS_COLOR_MAP.get(cname, DEFAULT_COLOR)
    cv2.imwrite(os.path.join(out_dirs['color_mask'], f"{name}.png"), color_mask)
    
    # 提取二值掩膜
    branch_bin = cv2.inRange(color_mask, LOWER_YELLOW, UPPER_YELLOW)
    branch_bin = cv2.morphologyEx(branch_bin, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    fruit_bin = cv2.dilate(cv2.inRange(color_mask, LOWER_RED, UPPER_RED), np.ones((25,25), np.uint8), iterations=1)
    seed = cv2.bitwise_and(branch_bin, fruit_bin)
    cv2.imwrite(os.path.join(out_dirs['binary_mask'], f"{name}.png"), branch_bin)
    
    # 连通域筛选
    _, labels, stats, _ = cv2.connectedComponentsWithStats(branch_bin, connectivity=8)
    valid_comps = []
    for i in range(1, stats.shape[0]):
        if stats[i, cv2.CC_STAT_AREA] > 400:
            comp = (labels == i).astype(np.uint8) * 255
            if cv2.countNonZero(cv2.bitwise_and(comp, seed)) > 0:
                valid_comps.append(comp)
    if not valid_comps: return []
    
    results = []; vis_img = orig.copy()
    
    for comp in valid_comps:
        skel = skeletonize((comp>0).astype(bool)).astype(np.uint8) * 255
        skel = cv2.dilate(skel, np.ones((3,3), np.uint8), iterations=1)
        
        fruit_pts = cv2.findNonZero(cv2.bitwise_and(comp, seed))
        if fruit_pts is None: continue
        fruit_centers = fruit_pts[:, 0, :]
        fruit_pt = tuple(np.mean(fruit_centers, axis=0).astype(int))
        
        pick_pt = None
        
        # === 倒Y型分叉检测 + 投票选起点（只选最高分）===
        valid_juncs = find_inverted_y_junctions(skel)
        junction = vote_junctions_upward(skel, fruit_centers, valid_juncs)
        
        # === 2. FPEA能量评估 ===
        if junction:
            path = trace_main_branch_up(skel, junction, max_len=400)
            candidates = []
            junc_x, junc_y = junction
            
            for pt in path:
                px, py = pt
                if py >= junc_y: continue
                if not strict_mask_check(comp, pt): continue
                dist = math.hypot(px - junc_x, py - junc_y)
                if dist < MIN_DIST_JUNC or dist > MAX_DIST_JUNC: continue
                width = get_horizontal_width(comp, pt)
                curve = calc_local_curvature(skel, pt)
                energy = fpea_energy(width, curve, dist)
                candidates.append((energy, pt))
            
            if candidates:
                candidates.sort(key=lambda x: x[0])
                best = candidates[0][1]
                pick_pt = project_to_mask_center(comp, best, radius=12)
                if not strict_mask_check(comp, pick_pt):
                    pick_pt = rescue_off_mask_point(comp, skel, pick_pt)
        
        # === 3. 终极保底 ===
        if pick_pt is None or not strict_mask_check(comp, pick_pt):
            pick_pt = guaranteed_fallback(comp, skel, fruit_pt, junction)
        
        # === 4. 强制水平居中 + 输出校验 ===
        if pick_pt and strict_mask_check(comp, pick_pt):
            # 最终确保水平居中
            pick_pt = snap_to_horizontal_center(comp, pick_pt)
            
            results.append({"x": int(pick_pt[0]), "y": int(pick_pt[1])})
            draw_picking_point_aligned(vis_img, pick_pt, f"({pick_pt[0]},{pick_pt[1]})")
    
    # 组件级去重
    if len(results) > 1:
        filtered = []
        for p in results:
            if not any(math.hypot(p['x']-k['x'], p['y']-k['y']) < 55 for k in filtered):
                filtered.append(p)
        results = filtered
        
    cv2.imwrite(os.path.join(out_dirs['result'], f"{name}_harvest.jpg"), vis_img)
    with open(os.path.join(out_dirs['result'], f"{name}_pts.json"), 'w') as f:
        json.dump({"image": name, "width": w, "height": h, "points": results}, f, indent=2)
    return results

def main():
    print(f"[INFO] Loading model: {CHECKPOINT_PATH} | Device: {DEVICE}")
    model = RFDETRSegNano(pretrain_weights=CHECKPOINT_PATH)
    for d in FOLDERS.values(): os.makedirs(d, exist_ok=True)
    imgs = sorted(list(Path(TEST_IMAGE_DIR).glob('*.jpg')) + list(Path(TEST_IMAGE_DIR).glob('*.png')))
    print(f"[INFO] Found {len(imgs)} images. Starting inference...")
    
    total, success = 0, 0
    for img_p in tqdm(imgs, desc="FPEA Positioning"):
        try:
            pts = process_image(img_p, model, FOLDERS)
            total += 1
            if pts: success += 1
        except Exception as e:
            print(f"[ERROR] {img_p}: {e}")
    print(f"\n[DONE] Total: {total} | Positioned: {success} | Rate: {success/total*100:.1f}%")

if __name__ == "__main__":
    main()
