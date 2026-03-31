import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import csv  # 仅新增导入，不影响原有代码

# ====================== 【调试开关 & 核心参数】======================
# 调试开关：关闭帧差分过滤（测试时设为False，正式实验设为True）
ENABLE_MOTION_FILTER = False
# 论文核心参数
N_FRAMES = 30  # 论文：30帧统计
RD_THRESH = 0.31
BMS_THRESH = 0.28
W_DE = 1.0000
W_SE = 0.8572
W_VE = 0.9842
SE_LOW = 0.62
SE_HIGH = 0.88

# 核心修改部分
R_T = 110
S_T = 70

# 【优化】放宽跟踪距离阈值，减少ID刷新
TRACKING_DIST_THRESH = 120
# 【优化1】调高最小面积阈值，过滤微小噪点（可根据视频调整）
MIN_AREA_THRESH = 20
# 绘图最小帧数要求
MIN_PLOT_FRAME = 3

# 【新增】匹配机制优化参数：面积变化率阈值（±60%以内才允许匹配）
MAX_AREA_CHANGE_RATIO = 0.6
#
WHITE_THRESHOLD = 254  # 论文中白色核心的RGB阈值，保持不变


# ====================== 1. RGB-HIS转换 + 论文火焰候选区三规则 ======================
def rgb2his(rgb_frame):
    frame = rgb_frame.astype(np.float32) / 255.0
    B, G, R = cv2.split(frame)
    I = (R + G + B) / 3.0
    min_rgb = np.minimum(np.minimum(R, G), B)
    S = 1.0 - (min_rgb / (I + 1e-6))
    S = S * 255
    return R * 255, G * 255, B * 255, S


# 【核心优化2】重构候选区提取的形态学流程，解决连通性+噪点问题
def get_flame_candidate_paper(frame):
    R, G, B, S = rgb2his(frame)
    rule1 = R > R_T
    rule2 = (R >= G) & (G > B)
    rule3 = S >= ((255 - R) * S_T) / R_T
    paper_rules = rule1 & rule2 & rule3

    white_core = (R > WHITE_THRESHOLD) & (G > WHITE_THRESHOLD) & (B > WHITE_THRESHOLD)
    mask = np.zeros_like(R, dtype=np.uint8)
    mask[paper_rules | white_core] = 255

    # -------------------------- 优化形态学处理 --------------------------
    # 1. 小核开运算：彻底去除微小噪点（解决小干扰像素问题）
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

    # 2. 稍大核闭运算：填充火焰内部空洞、连接相邻火焰区域（解决连通性问题）
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)

    # 3. 最终小开运算：清除闭运算后残留的微小噪点
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    # -------------------------------------------------------------------

    return mask


# ====================== 2. 蓝色分量离散度过滤函数 ======================
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


# ====================== 3. 论文三图拼接+保存函数 ======================
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

    cv2.putText(canvas, "(a) Original RGB image", (w // 2 - 200, title_y), title_font, title_scale, (255, 255, 255),
                title_thickness)
    cv2.putText(canvas, "(b) Detection result of RGB-HIS fire model", (w + w // 2 - 350, title_y), title_font,
                title_scale, (255, 255, 255), title_thickness)
    cv2.putText(canvas, "(c) Detection result of the Blue dispersion", (w * 2 + w // 2 - 350, title_y), title_font,
                title_scale, (255, 255, 255), title_thickness)

    if save_path:
        cv2.imwrite(save_path, canvas)
    return canvas


# ====================== 4. 帧差分静态干扰排除 ======================
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


# ====================== 5. 工具函数：提取所有目标轮廓+质心+面积 ======================
def get_all_contours_centroid_area(mask):
    # 【修改】CHAIN_APPROX_NONE：保留所有轮廓点，让轮廓完全贴合边缘
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


# ====================== 【核心修改】6. 论文区域跟踪算法：质心距离+面积变化双重约束 ======================
def match_targets(prev_targets, curr_targets):
    matched = []
    used_prev = set()

    for curr_idx, (_, curr_c, curr_area) in enumerate(curr_targets):
        min_dist = float('inf')
        match_idx = -1

        for prev_idx, hist in enumerate(prev_targets):
            if prev_idx in used_prev:
                continue

            # 1. 计算质心距离
            prev_c = hist["centroid"][-1]
            dist = np.sqrt((curr_c[0] - prev_c[0]) ** 2 + (curr_c[1] - prev_c[1]) ** 2)

            # 2. 【新增】计算面积变化率
            prev_area = hist["area"][-1] if len(hist["area"]) > 0 else curr_area
            area_change_ratio = abs(curr_area - prev_area) / (prev_area + 1e-6)  # 加小值防除零

            # 3. 双重约束：质心距离足够近 + 面积变化率在合理范围内
            if (dist < min_dist
                    and dist < TRACKING_DIST_THRESH
                    and area_change_ratio < MAX_AREA_CHANGE_RATIO):
                min_dist = dist
                match_idx = prev_idx

        matched.append(match_idx)
        if match_idx != -1:
            used_prev.add(match_idx)
    return matched


# ====================== 7. 三大专家模块 ======================
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
        return None, 0.0, 0.0

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

    return total_score / total_weight > 0.6


# ====================== 9. 论文风格质心轨迹绘制函数 ======================
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


# ====================== 10. 论文相似度时序散点图绘制函数 ======================
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

    # ====================== 【新增】CSV初始化 ======================
    csv_file = open("car.csv", "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "Frame", "Target_ID", "CX", "CY", "Area", "IoU_Similarity",
        "RD", "BMS", "Final_Result"
    ])

    # 逐帧调试：先读取第一帧
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
        prev_target_list = list(tracked_targets.values())
        matched_indices = match_targets(prev_target_list, current_targets)

        new_tracked = {}
        for idx, (cnt, (cx, cy), area) in enumerate(current_targets):
            # 静态干扰排除
            if not check_region_motion(diff_mask, cnt):
                continue

            match_idx = matched_indices[idx]
            if match_idx != -1:
                target_hist = prev_target_list[match_idx]
                centroid_queue = target_hist["centroid"]
                area_queue = target_hist["area"]
                prev_single_mask = target_hist["last_mask"]
                t_id = target_hist["id"]
            else:
                centroid_queue = deque(maxlen=N_FRAMES)
                area_queue = deque(maxlen=N_FRAMES)
                prev_single_mask = None
                t_id = target_id
                target_id += 1
                # 新目标初始化数据存储
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
                "last_mask": current_single_mask,
                "contour": cnt,
                "result": final_res,
                "RD": RD,
                "BMS": BMS,
                "cx": cx,
                "cy": cy
            }

            # ====================== 【新增】写入CSV ======================
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

        # 逐帧调试提示信息
        cv2.putText(display_frame, f"Frame: {frame_count} | SPACE: Next | Q: Quit | S: Save", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 终端调试信息
        tracked_num = len(new_tracked)
        print(f"帧号:{frame_count} | 跟踪目标数:{tracked_num}", end="")
        for tid, traj in full_trajectories.items():
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
            # 空格键前进一帧
            ret, frame = cap.read()
            if not ret:
                print("视频播放完毕")
                break
        elif key == ord('s'):
            # s键保存当前论文图
            save_path = f"paper_figure_{save_count}_frame_{frame_count}.png"
            cv2.imwrite(save_path, paper_canvas)
            print(f"论文图片已保存为 {save_path}")
            save_count += 1

    # 关闭CSV文件
    csv_file.close()
    print("\n✅ 所有目标数据已保存到 flame_data.csv")

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
    # 替换为你的测试视频路径
    flame_detection("part_mp4\car.mp4")