import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# ====================== 【论文绝对核心参数】======================
N_FRAMES = 30  # 论文：30帧统计
RD_THRESH = 0.31
BMS_THRESH = 0.28

# 论文训练集得到的专家权重
W_DE = 1.0000
W_SE = 0.8572
W_VE = 0.9842

# 论文帧间相似度阈值
SE_LOW = 0.62
SE_HIGH = 0.88

# 论文RGB三规则阈值
R_T = 125
S_T = 55

# 论文区域跟踪：质心距离阈值
TRACKING_DIST_THRESH = 50

# ====================== 1. RGB-HIS转换 + 论文火焰候选区三规则 ======================
def rgb2his(rgb_frame):
    frame = rgb_frame.astype(np.float32) / 255.0
    B, G, R = cv2.split(frame)
    I = (R + G + B) / 3.0
    min_rgb = np.minimum(np.minimum(R, G), B)
    S = 1.0 - (min_rgb / (I + 1e-6))
    S = S * 255
    return R * 255, G * 255, B * 255, S

def get_flame_candidate_paper(frame):
    R, G, B, S = rgb2his(frame)
    rule1 = R > R_T
    rule2 = (R >= G) & (G > B)
    rule3 = S >= ((255 - R) * S_T) / R_T
    mask = np.zeros_like(R, dtype=np.uint8)
    mask[rule1 & rule2 & rule3] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

# ====================== 2. 工具函数：提取所有目标轮廓+质心+面积 ======================
def get_all_contours_centroid_area(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    targets = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        targets.append((cnt, (cx, cy), area))
    return targets

# ====================== 3. 论文区域跟踪算法：质心最小距离匹配同一目标 ======================
def match_targets(prev_targets, curr_targets):
    matched = []
    used_prev = set()

    for curr_idx, (_, curr_c, _) in enumerate(curr_targets):
        min_dist = float('inf')
        match_idx = -1

        for prev_idx, hist in enumerate(prev_targets):
            if prev_idx in used_prev:
                continue
            prev_c = hist["centroid"][-1]
            dist = np.sqrt((curr_c[0] - prev_c[0]) ** 2 + (curr_c[1] - prev_c[1]) ** 2)
            if dist < min_dist and dist < TRACKING_DIST_THRESH:
                min_dist = dist
                match_idx = prev_idx

        matched.append(match_idx)
        if match_idx != -1:
            used_prev.add(match_idx)
    return matched

# ====================== 4. 三大专家模块（100%对齐论文）======================
def DE_Expert(frame, candidate_mask):
    B, _, _ = cv2.split(frame)
    blue_roi = B[candidate_mask == 255]
    if len(blue_roi) == 0:
        return False
    blue_std = np.std(blue_roi)
    return blue_std > 11

def SE_Expert(current_single_mask, prev_single_mask):
    if prev_single_mask is None:
        return True, np.nan
    intersection = cv2.countNonZero(cv2.bitwise_and(current_single_mask, prev_single_mask))
    union = cv2.countNonZero(cv2.bitwise_or(current_single_mask, prev_single_mask))
    if union == 0:
        return False, 0.0
    iou = intersection / union
    se_res = SE_LOW <= iou <= SE_HIGH
    return se_res, iou

def VE_Expert(centroid_queue, area_queue):
    if len(centroid_queue) < N_FRAMES or len(area_queue) < N_FRAMES:
        return False, 0.0, 0.0

    coords = list(centroid_queue)
    ZD = 0.0
    for i in range(1, len(coords)):
        dx = coords[i][0] - coords[i - 1][0]
        dy = coords[i][1] - coords[i - 1][1]
        ZD += np.sqrt(dx ** 2 + dy ** 2)
    DS = np.sqrt((coords[0][0] - coords[-1][0]) ** 2 + (coords[0][1] - coords[-1][1]) ** 2)
    RD = DS / ZD if ZD != 0 else 0.0
    MS = np.mean(area_queue)
    BMS = ZD / (N_FRAMES * np.sqrt(MS)) if MS > 0 else 0.0
    is_flame_ve = (RD < RD_THRESH) and (BMS > BMS_THRESH)
    return is_flame_ve, round(RD, 2), round(BMS, 2)

# ====================== 5. 论文多专家系统（MES）加权投票融合 ======================
def MES_Fusion(de_res, se_res, ve_res):
    total_score = de_res * W_DE + se_res * W_SE + ve_res * W_VE
    max_possible_score = W_DE + W_SE + W_VE
    return total_score / max_possible_score > 0.6

# ====================== 6. 论文风格质心轨迹绘制函数 ======================
def plot_centroid_trajectories(trajectory_dict):
    if not trajectory_dict:
        print("无有效质心轨迹数据，跳过绘图")
        return

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['figure.dpi'] = 100

    arrow_colors = ['#8B4513', '#4682B4', '#D4AF37', '#5F9EA0', '#9370DB', '#CD853F', '#483D8B']
    plot_list = list(trajectory_dict.items())[:7]
    n_plots = len(plot_list)
    n_rows = 3 if n_plots > 3 else 1 if n_plots ==1 else 2
    n_cols = min(n_plots, 3)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax_idx, (target_id, centroid_data) in enumerate(plot_list):
        ax = axes[ax_idx]
        n_frames = len(centroid_data)
        if n_frames < 2:
            continue

        x_min, x_max = np.min(centroid_data[:, 0]), np.max(centroid_data[:, 0])
        y_min, y_max = np.min(centroid_data[:, 1]), np.max(centroid_data[:, 1])
        x_pad = (x_max - x_min) * 0.1 if (x_max - x_min) > 0 else 10
        y_pad = (y_max - y_min) * 0.1 if (y_max - y_min) > 0 else 10
        x_range = (x_max + x_pad) - (x_min - x_pad)

        for i in range(n_frames - 1):
            x_prev, y_prev = centroid_data[i]
            x_curr, y_curr = centroid_data[i+1]
            color = arrow_colors[i % len(arrow_colors)]
            head_width = x_range * 0.012
            head_length = x_range * 0.015

            arrow = FancyArrowPatch(
                (x_prev, y_prev), (x_curr, y_curr),
                arrowstyle=f'->, head_width={head_width}, head_length={head_length}',
                color=color, linewidth=1.2, zorder=2
            )
            ax.add_patch(arrow)

        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.invert_yaxis()
        ax.set_xlabel("Pixels", fontsize=12)
        ax.set_ylabel("Pixels", fontsize=12)
        ax.set_title(f"Target ID: {target_id}", fontsize=12, y=-0.18)
        ax.tick_params(axis='both', labelsize=10)

    plt.subplots_adjust(left=0.08, right=0.92, bottom=0.15, top=0.92, wspace=0.25, hspace=0.35)
    plt.savefig("flame_centroid_trajectories.png", dpi=300, bbox_inches='tight')
    print("质心轨迹图已保存为 flame_centroid_trajectories.png")
    plt.show()

# ====================== 7. 论文相似度时序散点图绘制函数 ======================
def plot_similarity_scatter(similarity_dict):
    if not similarity_dict:
        print("无有效相似度数据，跳过绘图")
        return

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['figure.dpi'] = 120

    subplot_titles = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)']
    marker_style = '*'
    marker_color = '#00008B'
    marker_size = 8

    plot_list = list(similarity_dict.items())[:7]
    n_plots = len(plot_list)
    n_rows = 3
    n_cols = 3

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 10))
    axes = axes.flatten()

    for ax_idx in range(n_plots, n_rows*n_cols):
        axes[ax_idx].axis('off')

    for ax_idx, (target_id, sim_data) in enumerate(plot_list):
        ax = axes[ax_idx]
        valid_data = sim_data[~np.isnan(sim_data[:, 1])]
        if len(valid_data) == 0:
            continue

        frame_nums = valid_data[:, 0]
        sim_values = valid_data[:, 1]

        ax.scatter(frame_nums, sim_values, marker=marker_style, color=marker_color, s=marker_size, linewidths=0.5)

        ax.set_xlim(0, max(300, np.max(frame_nums)))
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set_xlabel("Frame Number", fontsize=10)
        ax.set_ylabel("Similarity", fontsize=10)
        ax.set_title(subplot_titles[ax_idx], fontsize=11, y=-0.2)
        ax.tick_params(axis='both', labelsize=9)
        ax.grid(False)

    plt.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.95, wspace=0.22, hspace=0.35)
    plt.savefig("flame_similarity_scatter.png", dpi=300, bbox_inches='tight')
    print("帧间相似度散点图已保存为 flame_similarity_scatter.png")
    plt.show()

# ====================== 主函数：纯视频流处理 ======================
def flame_detection(video_path):
    cap = cv2.VideoCapture(video_path)
    tracked_targets = {}
    target_id = 0
    frame_count = 0

    # 数据存储字典
    full_trajectories = {}
    similarity_records = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        display_frame = frame.copy()

        # 步骤1：论文RGB-HIS模型提取火焰候选区
        candidate_mask = get_flame_candidate_paper(frame)
        # 步骤2：提取所有候选目标的轮廓、质心、面积
        current_targets = get_all_contours_centroid_area(candidate_mask)
        # 步骤3：论文区域跟踪算法，匹配连续帧的同一目标
        prev_target_list = list(tracked_targets.values())
        matched_indices = match_targets(prev_target_list, current_targets)

        new_tracked = {}
        # 步骤4：对每个目标独立计算特征、专家判定
        for idx, (cnt, (cx, cy), area) in enumerate(current_targets):
            match_idx = matched_indices[idx]

            # 匹配到历史目标，继承队列与上一帧掩码
            if match_idx != -1:
                target_hist = prev_target_list[match_idx]
                centroid_queue = target_hist["centroid"]
                area_queue = target_hist["area"]
                prev_single_mask = target_hist["last_mask"]
                t_id = target_hist["id"]
            # 新目标，初始化队列
            else:
                centroid_queue = deque(maxlen=N_FRAMES)
                area_queue = deque(maxlen=N_FRAMES)
                prev_single_mask = None
                t_id = target_id
                target_id += 1
                # 为新目标初始化数据存储
                full_trajectories[t_id] = []
                similarity_records[t_id] = []

            # 更新目标队列
            centroid_queue.append((cx, cy))
            area_queue.append(area)
            full_trajectories[t_id].append((cx, cy))

            # 生成当前目标的真实轮廓掩码
            current_single_mask = np.zeros_like(candidate_mask)
            cv2.drawContours(current_single_mask, [cnt], -1, 255, -1)

            # 三大专家独立判定
            de_res = DE_Expert(frame, current_single_mask)
            se_res, iou_value = SE_Expert(current_single_mask, prev_single_mask)
            ve_res, RD, BMS = VE_Expert(centroid_queue, area_queue)

            # 记录当前帧的相似度数据
            similarity_records[t_id].append([frame_count, iou_value])

            # 多专家加权融合最终判定
            final_res = MES_Fusion(de_res, se_res, ve_res)

            # 保存目标跟踪信息
            new_tracked[t_id] = {
                "id": t_id,
                "centroid": centroid_queue,
                "area": area_queue,
                "last_mask": current_single_mask,
                "contour": cnt,
                "result": final_res,
                "RD": RD,
                "BMS": BMS,
                "cx": cx,
                "cy": cy
            }

            # 可视化
            color = (0, 255, 0) if final_res else (0, 0, 255)
            cv2.drawContours(display_frame, [cnt], -1, color, 2)
            cv2.putText(display_frame, f"ID{t_id}", (cx - 15, cy - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(display_frame, f"RD:{RD} BMS:{BMS}", (cx - 30, cy + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 更新跟踪器与上一帧信息
        tracked_targets = new_tracked

        # 显示结果
        cv2.imshow("Paper-Exact Flame Detection", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 检测结束后，自动绘制两张论文图
    np_trajectories = {tid: np.array(traj) for tid, traj in full_trajectories.items() if len(traj) >= 5}
    plot_centroid_trajectories(np_trajectories)

    np_similarity = {tid: np.array(sim) for tid, sim in similarity_records.items() if len(sim) >= 5}
    plot_similarity_scatter(np_similarity)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 替换为你的测试视频路径
    flame_detection("test_video.mp4")