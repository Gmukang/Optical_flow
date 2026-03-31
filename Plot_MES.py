import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def plot_single_target_trajectory(target_id, df):
    """
    绘制单个目标的质心运动轨迹图（论文风格）
    """
    # 筛选数据
    target_data = df[df['Target_ID'] == target_id].sort_values('Frame')
    coords = target_data[['CX', 'CY']].values

    if len(coords) < 2:
        print(f"目标ID {target_id} 的数据不足2帧，无法绘制轨迹图")
        return

    # 论文风格设置
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['figure.dpi'] = 100

    arrow_colors = ['#8B4513', '#4682B4', '#D4AF37', '#5F9EA0', '#9370DB', '#CD853F', '#483D8B']

    fig, ax = plt.subplots(figsize=(6, 5))
    n_frames = len(coords)

    x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
    y_min, y_max = np.min(coords[:, 1]), np.max(coords[:, 1])
    x_pad = (x_max - x_min) * 0.1 if (x_max - x_min) > 0 else 10
    y_pad = (y_max - y_min) * 0.1 if (y_max - y_min) > 0 else 10
    x_range = (x_max + x_pad) - (x_min - x_pad)

    # 绘制箭头轨迹
    for i in range(n_frames - 1):
        x_prev, y_prev = coords[i]
        x_curr, y_curr = coords[i + 1]
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

    plt.subplots_adjust(left=0.15, right=0.92, bottom=0.15, top=0.92)
    save_path = f"single_target_{target_id}_trajectory.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 目标ID {target_id} 的质心轨迹图已保存为 {save_path}")
    plt.show()


def plot_single_target_similarity(target_id, df):
    """
    绘制单个目标的相似度时序散点图（论文风格）
    """
    # 筛选数据
    target_data = df[df['Target_ID'] == target_id].sort_values('Frame')
    valid_data = target_data[target_data['IoU_Similarity'] != -1]  # 过滤掉第一帧的NaN

    if len(valid_data) == 0:
        print(f"目标ID {target_id} 无有效的相似度数据")
        return

    # 论文风格设置
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['figure.dpi'] = 120

    marker_style = '*'
    marker_color = '#00008B'
    marker_size = 8

    fig, ax = plt.subplots(figsize=(6, 5))

    frame_nums = valid_data['Frame'].values
    sim_values = valid_data['IoU_Similarity'].values

    ax.scatter(frame_nums, sim_values, marker=marker_style, color=marker_color, s=marker_size, linewidths=0.5)

    ax.set_xlim(0, max(300, np.max(frame_nums)))
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xlabel("Frame Number", fontsize=10)
    ax.set_ylabel("Similarity", fontsize=10)
    ax.set_title("(a)", fontsize=11, y=-0.2)
    ax.tick_params(axis='both', labelsize=9)
    ax.grid(False)

    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.92)
    save_path = f"single_target_{target_id}_similarity.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 目标ID {target_id} 的相似度散点图已保存为 {save_path}")
    plt.show()


if __name__ == "__main__":
    # 1. 读取CSV文件
    csv_path = "flame_data.csv"
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ 成功读取 {csv_path}")
        print(f"数据包含的目标ID有: {sorted(df['Target_ID'].unique())}")
    except FileNotFoundError:
        print(f"❌ 找不到文件 {csv_path}，请先运行火焰检测代码生成数据")
        exit()

    # 2. 用户输入目标ID
    while True:
        try:
            target_id_input = input("\n请输入你要分析的 Target_ID (输入 q 退出): ")
            if target_id_input.lower() == 'q':
                break
            target_id = int(target_id_input)

            if target_id not in df['Target_ID'].values:
                print(f"❌ 目标ID {target_id} 不存在，请从以下列表中选择: {sorted(df['Target_ID'].unique())}")
                continue

            # 3. 绘图
            print(f"\n正在绘制目标ID {target_id} 的图表...")
            plot_single_target_trajectory(target_id, df)
            plot_single_target_similarity(target_id, df)

        except ValueError:
            print("❌ 请输入有效的数字ID")