import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import csv

# ====================== 【调试开关 & 核心参数】======================
ENABLE_MOTION_FILTER = True
N_FRAMES = 30
RD_THRESH = 0.31
BMS_THRESH = 0.28
W_DE = 1.0000
W_SE = 0.8572
W_VE = 0.9842
SE_LOW = 0.62
SE_HIGH = 0.88

# 核心参数（完全保留你的设置）
R_T = 150
S_T = 65
MIN_PLOT_FRAME = 3
TRACKING_DIST_THRESH = 150  # 🔥 修复1：放宽跟踪距离阈值，适应火焰移动
MIN_AREA_THRESH = 20
MAX_LOST_FRAMES = 8  # 🔥 修复2：增加丢失容忍帧数，适应火焰短暂消失
# 🔥 修复3：面积变化率阈值从1.2改为3.0（±200%，覆盖火焰燃烧的剧烈波动）
MAX_AREA_CHANGE_RATIO = 3.0
WHITE_THRESHOLD = 210

# ====================== 1. RGB-HIS转换 ======================
def rgb2his(rgb_frame):
    frame = rgb_frame.astype(np.float32) / 255.0
    B, G, R = cv2.split(frame)
    I = (R + G + B) / 3.0
    min_rgb = np.minimum(np.minimum(R, G), B)
    S = 1.0 - (min_rgb / (I + 1e-6))
    S = S * 255
    return R * 255, G * 255, B * 255, S

# ====================== 2. 火焰候选区提取 ======================
def get_flame_candidate_paper(frame):
    R, G, B, S = rgb2his(frame)
    rule1 = R > R_T
    rule2 = (R >= G) & (G > B)
    rule3 = S >= ((255 - R) * S_T) / R_T
    mask = np.zeros_like(R, dtype=np.uint8)
    mask[rule1 & rule2 & rule3] = 255

    # 形态学处理（完全保留你的设置）
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    return mask

# ====================== 3. 蓝色分量离散度过滤 ======================
def get_blue_dispersion_filtered_mask(frame, candidate_mask):
    filtered_mask = np.zeros_like(candidate_mask)
    contours, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    B, _, _ = cv2.split(frame)
    for cnt in contours:
        single_mask = np.zeros_like(candidate_mask)
        cv2.drawContours(single_mask, [cnt], -1, 255, -1)
        blue_roi = B[single_mask == 255]
        if len(blue_roi) == 0:
            continue
        blue_std = np.std(blue_roi)
        if blue_std > 11:
            filtered_mask = cv2.bitwise_or(filtered_mask, single_mask)
    return filtered_mask

# ====================== 4. 论文三图拼接 ======================
def generate_paper_figure(original_frame, his_mask, filtered_mask, save_path=None):
    h, w = original_frame.shape[:2]
    his_3ch = cv2.cvtColor(his_mask, cv2.COLOR_GRAY2BGR)
    filtered_3ch = cv2.cvtColor(filtered_mask, cv2.COLOR_GRAY2BGR)
    combined = np.hstack([original_frame, his_3ch, filtered_3ch])

    title_font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = 1.2
    title_thickness = 2
    title_y = h + 60
    canvas = np.zeros((h + 100, w * 3, 3), dtype=np.uint8)
    canvas[:h, :, :] = combined

    cv2.putText(canvas, "(a) Original RGB image", (w // 2 - 200, title_y), title_font, title_scale, (255, 255, 255), title_thickness)
    cv2.putText(canvas, "(b) Detection result of RGB-HIS fire model", (w + w // 2 - 350, title_y), title_font, title_scale, (255, 255, 255), title_thickness)
    cv2.putText(canvas, "(c) Detection result of the Blue dispersion", (w * 2 + w // 2 - 350, title_y), title_font, title_scale, (255, 255, 255), title_thickness)

    if save_path:
        cv2.imwrite(save_path, canvas)
    return canvas

# ====================== 5. 帧差分静态干扰排除 ======================
def calculate_frame_diff(prev_gray, curr_gray):
    if prev_gray is None:
        return None
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, diff_mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    return diff_mask

def check_region_motion(diff_mask, cnt):
    if not ENABLE_MOTION_FILTER or diff_mask is None:
        return True
    region_mask = np.zeros_like(diff_mask)
    cv2.drawContours(region_mask, [cnt], -1, 255, -1)
    motion_pixels = cv2.countNonZero(cv2.bitwise_and(diff_mask, region_mask))
    return motion_pixels > 0

# ====================== 6. 工具函数：提取目标轮廓+质心+面积 ======================
def get_all_contours_centroid_area(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    targets = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA_THRESH:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        targets.append((cnt, (cx, cy), area))
    return targets

# ====================== 🔥【终极修复版】跟踪匹配函数 ======================
def match_targets(tracked_targets_dict, curr_targets):
    matched = [-1] * len(curr_targets)
    used_ids = set()

    # 🔥 修复4：匈牙利算法最优匹配，彻底解决贪心匹配的ID抢占问题
    # 构建代价矩阵：代价 = 距离 + 面积变化率（越小越优）
    cost_matrix = []
    for curr_idx, (_, curr_c, curr_area) in enumerate(curr_targets):
        row = []
        for tid, hist in tracked_targets_dict.items():
            prev_c = hist["centroid"][-1]
            dist = np.hypot(curr_c[0] - prev_c[0], curr_c[1] - prev_c[1])
            prev_area = hist["area"][-1] if len(hist["area"]) > 0 else curr_area
            area_ratio = abs(curr_area - prev_area) / (prev_area + 1e-6)
            # 综合代价：距离权重0.7，面积权重0.3，优先保证位置连续
            cost = 0.7 * dist + 0.3 * area_ratio * 100  # 面积归一化
            row.append(cost)
        cost_matrix.append(row)

    # 匈牙利算法求解最优匹配
    from scipy.optimize import linear_sum_assignment
    if len(cost_matrix) > 0 and len(cost_matrix[0]) > 0:
        curr_indices, prev_indices = linear_sum_assignment(cost_matrix)
        # 遍历匹配结果，过滤不符合约束的匹配
        for curr_idx, prev_idx in zip(curr_indices, prev_indices):
            tid = list(tracked_targets_dict.keys())[prev_idx]
            if tid in used_ids:
                continue
            # 双重约束验证
            _, curr_c, curr_area = curr_targets[curr_idx]
            hist = tracked_targets_dict[tid]
            prev_c = hist["centroid"][-1]
            dist = np.hypot(curr_c[0] - prev_c[0], curr_c[1] - prev_c[1])
            prev_area = hist["area"][-1] if len(hist["area"]) > 0 else curr_area
            area_ratio = abs(curr_area - prev_area) / (prev_area + 1e-6)
            if dist < TRACKING_DIST_THRESH and area_ratio < MAX_AREA_CHANGE_RATIO:
                matched[curr_idx] = tid
                used_ids.add(tid)

    return matched

# ====================== 7. 三大专家模块 ======================
def DE_Expert(frame, candidate_mask):
    B, _, _ = cv2.split(frame)
    blue_roi = B[candidate_mask == 255]
    if len(blue_roi) == 0:
        return False
    return np.std(blue_roi) > 11

def SE_Expert(current_single_mask, prev_single_mask):
    if prev_single_mask is None:
        return True, np.nan
    intersection = cv2.countNonZero(cv2.bitwise_and(current_single_mask, prev_single_mask))
    union = cv2.countNonZero(cv2.bitwise_or(current_single_mask, prev_single_mask))
    if union == 0:
        return False, 0.0
    iou = intersection / union
    return SE_LOW <= iou <= SE_HIGH, iou

def VE_Expert(centroid_queue, area_queue):
    if len(centroid_queue) < N_FRAMES or len(area_queue) < N_FRAMES:
        return None, 0.0, 0.0
    coords = list(centroid_queue)
    ZD = 0.0
    for i in range(1, len(coords)):
        ZD += np.hypot(coords[i][0] - coords[i-1][0], coords[i][1] - coords[i-1][1])
    DS = np.hypot(coords[0][0] - coords[-1][0], coords[0][1] - coords[-1][1])
    RD = DS / ZD if ZD != 0 else 0.0
    MS = np.mean(area_queue)
    BMS = ZD / (N_FRAMES * np.sqrt(MS)) if MS > 0 else 0.0
    return (RD < RD_THRESH) and (BMS > BMS_THRESH), round(RD, 2), round(BMS, 2)

# ====================== 8. 动态多专家融合 ======================
def MES_Fusion(de_res, se_res, ve_res):
    valid_experts = []
    valid_experts.append((de_res, W_DE))
    valid_experts.append((se_res, W_SE))
    if ve_res is not None:
        valid_experts.append((ve_res, W_VE))

    total_score = 0.0
    total_weight = 0.0
    for res, weight in valid_experts:
        total_score += res * weight
        total_weight += weight

    return total_score / total_weight > 0.6 if total_weight > 0 else False

# ====================== 9. 论文风格质心轨迹绘制 ======================
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
    n_rows = 3 if n_plots > 3 else 1 if n_plots == 1 else 2
    n_cols = min(n_plots, 3)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(6 * n_cols, 5 * n_rows))
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
            x_curr, y_curr = centroid_data[i + 1]
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

# ====================== 10. 论文相似度时序散点图绘制 ======================
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

    for ax_idx in range(n_plots, n_rows * n_cols):
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

# ====================== 主函数：逐帧调试模式 + CSV保存 ======================
def flame_detection(video_path):
    cap = cv2.VideoCapture(video_path)
    tracked_targets = {}
    target_id = 0
    frame_count = 0
    save_count = 0

    prev_gray = None
    full_trajectories = {}
    similarity_records = {}

    # CSV初始化（完全保留你的设置）
    csv_file = open("melt.csv", "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "Frame", "Target_ID", "CX", "CY", "Area", "IoU_Similarity",
        "RD", "BMS", "Final_Result"
    ])

    ret, frame = cap.read()
    if not ret:
        print("无法读取视频")
        return

    while True:
        frame_count += 1
        display_frame = frame.copy()

        # 帧差分计算
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff_mask = calculate_frame_diff(prev_gray, curr_gray)

        # 论文三图生成
        his_mask = get_flame_candidate_paper(frame)
        filtered_mask = get_blue_dispersion_filtered_mask(frame, his_mask)
        paper_canvas = generate_paper_figure(frame, his_mask, filtered_mask)

        # 提取候选目标
        current_targets = get_all_contours_centroid_area(his_mask)
        matched_indices = match_targets(tracked_targets, current_targets)

        new_tracked = {}
        # 1. 处理本帧成功检测到的目标
        for idx, (cnt, (cx, cy), area) in enumerate(current_targets):
            if not check_region_motion(diff_mask, cnt):
                continue

            match_id = matched_indices[idx]

            # 如果匹配到了历史目标
            if match_id != -1 and match_id in tracked_targets:
                t_data = tracked_targets[match_id]
                t_data["lost_frames"] = 0  # 重置丢失帧数
                centroid_queue = t_data["centroid"]
                area_queue = t_data["area"]
                prev_single_mask = t_data["last_mask"]  # 保留历史掩码
                t_id = match_id
            # 如果是全新的目标
            else:
                centroid_queue = deque(maxlen=N_FRAMES)
                area_queue = deque(maxlen=N_FRAMES)
                prev_single_mask = None
                t_id = target_id
                target_id += 1
                full_trajectories[t_id] = []
                similarity_records[t_id] = []

            # 更新队列和数据记录
            centroid_queue.append((cx, cy))
            area_queue.append(area)
            full_trajectories[t_id].append((cx, cy))

            current_single_mask = np.zeros_like(his_mask)
            cv2.drawContours(current_single_mask, [cnt], -1, 255, -1)

            # 专家判定
            de_res = DE_Expert(frame, current_single_mask)
            se_res, iou_value = SE_Expert(current_single_mask, prev_single_mask)
            ve_res, RD, BMS = VE_Expert(centroid_queue, area_queue)

            # 记录相似度数据
            similarity_records[t_id].append([frame_count, iou_value])
            final_res = MES_Fusion(de_res, se_res, ve_res)

            new_tracked[t_id] = {
                "id": t_id,
                "centroid": centroid_queue,
                "area": area_queue,
                "last_mask": current_single_mask,  # 更新最新掩码
                "contour": cnt,
                "result": final_res,
                "RD": RD,
                "BMS": BMS,
                "cx": cx,
                "cy": cy,
                "lost_frames": 0
            }

            # 写入CSV
            csv_writer.writerow([
                frame_count, t_id, cx, cy, round(area, 2),
                round(iou_value, 4) if not np.isnan(iou_value) else -1,
                RD, BMS, 1 if final_res else 0
            ])

            # 实时可视化
            color = (0, 255, 0) if final_res else (0, 0, 255)
            cv2.drawContours(display_frame, [cnt], -1, color, 2)
            cv2.putText(display_frame, f"ID{t_id}", (cx - 15, cy - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            ve_status = f"RD:{RD} BMS:{BMS}" if ve_res is not None else "VE: WAITING..."
            cv2.putText(display_frame, ve_status, (cx - 40, cy + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 2. 处理本帧未检测到的目标（丢失容忍逻辑）
        for t_id, t_data in tracked_targets.items():
            if t_id not in new_tracked:
                t_data["lost_frames"] += 1
                if t_data["lost_frames"] <= MAX_LOST_FRAMES:
                    new_tracked[t_id] = t_data

        # 逐帧调试提示信息
        cv2.putText(display_frame, f"Frame: {frame_count} | SPACE: Next | Q: Quit | S: Save", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 终端调试信息
        active_tracked_num = sum(1 for t in new_tracked.values() if t["lost_frames"] == 0)
        print(f"帧号:{frame_count} | 活跃目标数:{active_tracked_num}", end="")
        for tid, traj in full_trajectories.items():
            if tid in new_tracked:
                print(f" | ID{tid}轨迹长度:{len(traj)}", end="")
        print("")

        # 更新跟踪器
        tracked_targets = new_tracked
        prev_gray = curr_gray.copy()

        # 显示窗口
        cv2.imshow("Real-time Flame Detection", display_frame)
        cv2.imshow("Paper Figure (a)(b)(c)", paper_canvas)

        # 逐帧按键控制
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            ret, frame = cap.read()
            if not ret:
                print("视频播放完毕")
                break
        elif key == ord('s'):
            save_path = f"paper_figure_{save_count}_frame_{frame_count}.png"
            cv2.imwrite(save_path, paper_canvas)
            print(f"论文图片已保存为 {save_path}")
            save_count += 1

    # 关闭CSV文件
    csv_file.close()
    print("\n✅ 所有目标数据已保存到 melt.csv")

    # 检测结束后数据统计
    print("\n" + "=" * 60)
    print(f"视频总帧数:{frame_count} | 总跟踪目标数:{len(full_trajectories)}")
    for tid, traj in full_trajectories.items():
        print(f"目标ID{tid} | 总帧数:{len(traj)} | 相似度记录数:{len(similarity_records[tid])}")
    print("=" * 60 + "\n")

    # 绘图
    np_trajectories = {tid: np.array(traj) for tid, traj in full_trajectories.items() if len(traj) >= MIN_PLOT_FRAME}
    plot_centroid_trajectories(np_trajectories)

    np_similarity = {tid: np.array(sim) for tid, sim in similarity_records.items() if len(sim) >= MIN_PLOT_FRAME}
    plot_similarity_scatter(np_similarity)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    flame_detection("part_mp4\\melt2.mp4")