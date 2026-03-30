import cv2
import numpy as np
import scipy.sparse
import scipy.ndimage
import scipy.stats
import csv
import os
from tqdm import tqdm  # 进度条，可选安装：pip install tqdm
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde



def draw_optical_flow_field(img, u, v, step=16, scale=50, color=(255, 0, 0)):
    """
    在图像上绘制光流场箭头图（类似论文图8的效果）
    :param img: 输入图像（通常是缩放后的帧）
    :param u: 光流水平分量
    :param v: 光流垂直分量
    :param step: 箭头间距（越大越快，默认16）
    :param scale: 箭头长度缩放因子
    :param color: 箭头颜色 (BGR)
    :return: 绘制好的图像
    """
    # 复制图像，避免覆盖原图
    flow_img = img.copy()
    h, w = img.shape[:2]

    #修复，过滤无效光流，只保留有运动的部分
    # u = np.nan_to_num(u, 0)
    # v = np.nan_to_num(v, 0)

    # 遍历网格点
    for y in range(0, h, step):
        for x in range(0, w, step):
            # 获取当前像素的光流向量
            flow_u = u[y, x] * scale
            flow_v = v[y, x] * scale

            # 计算箭头终点（简化版向量叠加）
            x2 = int(x + flow_u)
            y2 = int(y + flow_v)

            # 跳过极小光流，避免无效绘制
            # if abs(flow_u) < 1 and abs(flow_v) < 1:
            #     continue

            # 绘制箭头
            # 1. 绘制箭头主体线
            cv2.line(flow_img, (x, y), (x2, y2), color, thickness=1)
            # 2. 绘制箭头尖（简化版：画两条短斜线）
            # 这里用简单的端点替代，更接近论文里的点划线风格
            cv2.circle(flow_img, (x2, y2), 1, color, -1)

    return flow_img

def plot_flow_kde_histogram(u_nsd, v_nsd, essential_mask, title_suffix="Fire", save_path=None, frame_idx=None):
    """
    绘制论文同款光流2D KDE等高线直方图（对应论文图8）
    :param u_nsd: NSD光流水平分量 (H×W)
    :param v_nsd: NSD光流垂直分量 (H×W)
    :param essential_mask: 有效像素掩码 (H×W, bool)
    :param title_suffix: 标题类型，"Fire"（火焰）/ "Rigid"（刚体）
    :param save_path: 保存路径（如"fire_kde_hist.png"），None则仅显示
    :param frame_idx: 帧号，用于按帧保存避免覆盖
    """
    # -------------------------- 步骤1：提取有效运动像素（论文Ω_e，严格遵循） --------------------------
    u_ess = u_nsd[essential_mask]
    v_ess = v_nsd[essential_mask]
    if len(u_ess) == 0:
        print("警告：无有效运动像素，无法绘制直方图")
        return

    # -------------------------- 步骤2：计算2D核密度估计（KDE，论文h(r,φ)的平滑实现） --------------------------
    flow_data = np.vstack([u_ess, v_ess])
    # 用scott自动选择带宽，匹配论文的平滑效果
    kde = gaussian_kde(flow_data, bw_method='scott')

    # -------------------------- 步骤3：生成u-v平面网格（完全匹配论文的-0.5~0.5范围） --------------------------
    u_grid = np.linspace(-0.6, 0.6, 100)  # 扩展范围避免截断，覆盖论文的-0.5~0.5
    v_grid = np.linspace(-0.6, 0.6, 100)
    U, V = np.meshgrid(u_grid, v_grid)

    # 计算每个网格点的KDE密度值
    grid_points = np.vstack([U.ravel(), V.ravel()])
    Z = kde(grid_points).reshape(U.shape)

    # -------------------------- 步骤4：绘制填充等高线图（1:1复刻论文风格） --------------------------
    plt.figure(figsize=(6, 5), dpi=150)  # 高分辨率，适合论文投稿
    # levels=10对应论文的多层灰度，cmap='gray'实现黑=高密度、白=低密度，vmin/vmax匹配论文颜色条
    contour = plt.contourf(U, V, Z, levels=10, cmap='gray', vmin=0, vmax=1.5)
    # 添加原点(0,0)的叉号×，完全匹配论文
    plt.scatter(0, 0, marker='x', color='k', s=30, linewidth=1.5)
    # 添加颜色条，刻度完全匹配论文
    cbar = plt.colorbar(contour)
    cbar.set_ticks([0, 0.5, 1, 1.5])
    # 轴标签1:1复刻论文
    plt.xlabel('flow vector component u', fontsize=12)
    plt.ylabel('flow vector component v', fontsize=12)
    # 标题完全匹配论文的(a)/(b)格式
    if title_suffix == "Fire":
        plt.title('(a) Fire motion histogram.', fontsize=14, pad=15)
        save_name = save_path or f"fire_flow_hist_{frame_idx}.png" if frame_idx else "fire_flow_hist.png"
    else:
        plt.title('(b) Rigid motion histogram.', fontsize=14, pad=15)
        save_name = save_path or f"rigid_flow_hist_{frame_idx}.png" if frame_idx else "rigid_flow_hist.png"

    # 轴范围严格匹配论文
    plt.xlim(-0.6, 0.6)
    plt.ylim(-0.6, 0.6)
    plt.tight_layout()  # 避免标签截断

    # -------------------------- 步骤5：保存/显示 --------------------------
    plt.savefig(save_name, bbox_inches='tight', dpi=300)
    print(f"直方图已保存到：{save_name}")
    plt.close()


# ====================== 论文核心特征提取函数（完全保留原版逻辑，无修改） ======================
def rgb_to_generalized_mass(img_bgr):
    """
    论文Section II-B2: 将RGB图像转换为广义质量图
    输入: img_bgr - OpenCV格式的BGR图像 (uint8)
    输出: mass - 广义质量图 (float32, [0,1])
    """
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    H = img_hsv[:, :, 0] / 179.0
    S = img_hsv[:, :, 1] / 255.0
    V = img_hsv[:, :, 2] / 255.0

    Hc = 0.083
    a = 100.0
    b = 0.11

    hue_dist = np.minimum(np.abs(Hc - H), 1.0 - np.abs(Hc - H))
    f_hue = 1.0 - 1.0 / (1.0 + np.exp(-a * (hue_dist - b)))
    mass = f_hue * S * V

    mass = np.clip(mass, 1e-6, 1.0)

    return mass.astype(np.float32)


def compute_nsd_optical_flow(I1, I2, alpha=0.1):
    """
    论文Section II-C: 计算Non-Smooth Data (NSD)光流
    输入: I1, I2 - 连续两帧的标量图像 (float32)
          alpha - 正则化参数
    输出: u_nsd, v_nsd - NSD光流的水平/垂直分量 (float32)
    """
    Ix = cv2.Sobel(I2, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(I2, cv2.CV_32F, 0, 1, ksize=3)
    It = I2 - I1

    grad_sq = Ix ** 2 + Iy ** 2
    denom = grad_sq + alpha

    u_nsd = - (Ix * It) / denom
    v_nsd = - (Iy * It) / denom
    return u_nsd, v_nsd


def compute_omt_optical_flow(I0, I1, alpha=20):
    """
    论文Section II-B: 计算Optimal Mass Transport (OMT)光流
    输入: I0, I1 - 连续两帧的广义质量图 (float32)
          alpha - 正则化参数
    输出: u_omt, v_omt - OMT光流的水平/垂直分量 (float32)
    """
    h, w = I0.shape
    N = h * w

    def build_derivative_matrix(h, w, direction='x'):
        rows, cols, vals = [], [], []
        for y in range(h):
            for x in range(w):
                i = y * w + x
                if direction == 'x':
                    if 0 < x < w - 1:
                        rows.extend([i, i])
                        cols.extend([i - 1, i + 1])
                        vals.extend([-0.5, 0.5])
                    elif x == 0:
                        rows.extend([i, i])
                        cols.extend([i, i + 1])
                        vals.extend([-1.0, 1.0])
                    else:
                        rows.extend([i, i])
                        cols.extend([i - 1, i])
                        vals.extend([-1.0, 1.0])
                else:
                    if 0 < y < h - 1:
                        rows.extend([i, i])
                        cols.extend([i - w, i + w])
                        vals.extend([-0.5, 0.5])
                    elif y == 0:
                        rows.extend([i, i])
                        cols.extend([i, i + w])
                        vals.extend([-1.0, 1.0])
                    else:
                        rows.extend([i, i])
                        cols.extend([i - w, i])
                        vals.extend([-1.0, 1.0])
        return scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))

    D_x = build_derivative_matrix(h, w, direction='x')
    D_y = build_derivative_matrix(h, w, direction='y')

    I_avg = (I0.flatten() + I1.flatten()) / 2.0

    I_avg = np.clip(I_avg, 1e-6, 1.0)
    I_hat = scipy.sparse.diags(I_avg, 0, shape=(N, N))

    D_x_I = D_x @ I_hat
    D_y_I = D_y @ I_hat
    A = scipy.sparse.hstack([D_x_I, D_y_I])

    It = I1.flatten() - I0.flatten()
    b = -It

    I_hat_block = scipy.sparse.block_diag([I_hat, I_hat])
    left_side = alpha * I_hat_block + A.T @ A
    right_side = A.T @ b

    try:
        u_vec = scipy.sparse.linalg.spsolve(left_side, right_side)
    except:
        u_vec = np.zeros(2 * N, dtype=np.float32)

    u_omt = u_vec[:N].reshape(h, w)
    v_omt = u_vec[N:].reshape(h, w)

    # 🔥 修复：过滤NaN/inf
    u_omt = np.nan_to_num(u_omt, nan=0.0, posinf=0.0, neginf=0.0)
    v_omt = np.nan_to_num(v_omt, nan=0.0, posinf=0.0, neginf=0.0)

    return u_omt.astype(np.float32), v_omt.astype(np.float32)


def get_essential_pixels(u, v, c=0.2):
    """
    论文Section III-A: 筛选运动显著的Essential Pixels
    输入: u, v - 光流分量
          c - 阈值系数 (论文默认0.2)
    输出: mask - Essential Pixels的布尔掩码
    """
    mag = np.sqrt(u ** 2 + v ** 2)
    max_mag = np.max(mag)
    if max_mag < 1e-8:
        return np.zeros_like(mag, dtype=bool)
    return mag > c * max_mag


def compute_source_match(u_omt, v_omt):
    """
    论文Section III-B3: 计算OMT源匹配特征 f3
    输入: u_omt, v_omt - OMT光流分量
    输出: f3 - 源匹配特征值
    """
    if np.max(np.abs(u_omt)) < 1e-8 and np.max(np.abs(v_omt)) < 1e-8:
        return 0.0

    h, w = u_omt.shape
    y, x = np.mgrid[0:h, 0:w]
    yc, xc = h // 2, w // 2
    y_rel = y - yc
    x_rel = x - xc
    r = np.sqrt(x_rel ** 2 + y_rel ** 2)
    r = np.maximum(r, 1e-8)

    u_T = np.exp(-r) * x_rel
    v_T = np.exp(-r) * y_rel

    mag_omt = np.sqrt(u_omt ** 2 + v_omt ** 2)
    mag_omt = np.maximum(mag_omt, 1e-8)
    u_norm = u_omt / mag_omt
    v_norm = v_omt / mag_omt

    conv_u = scipy.ndimage.convolve(u_norm, u_T, mode='constant')
    conv_v = scipy.ndimage.convolve(v_norm, v_T, mode='constant')

    res = np.max(np.abs(conv_u + conv_v))

    return np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)


def compute_directional_variance(u_nsd, v_nsd, essential_mask, n_bins=8):
    """
    论文Section III-B4: 计算NSD方向方差特征 f4
    输入: u_nsd, v_nsd - NSD光流分量
          essential_mask - Essential Pixels掩码
          n_bins - 方向分箱数
    输出: f4 - 方向方差特征值
    """
    u_ess = u_nsd[essential_mask]
    v_ess = v_nsd[essential_mask]
    if len(u_ess) == 0:
        return 0.0

    angles = np.arctan2(v_ess, u_ess)
    angles[angles < 0] += 2 * np.pi

    kde = scipy.stats.gaussian_kde(angles)
    bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
    s = np.zeros(n_bins)

    for i in range(n_bins):
        theta_samples = np.linspace(bin_edges[i], bin_edges[i + 1], 100)
        kde_vals = kde(theta_samples)
        s[i] = np.trapz(kde_vals, theta_samples)

    s = s / np.sum(s)
    f4 = np.var(s)
    return f4


def extract_karasev_features(frame1_bgr, frame2_bgr, alpha_nsd=0.1, alpha_omt=20, c_essential=0.2, resize_max=640,
                             frame_index = None, draw_kde=False):
    """
    基于Mueller et al. (2013)论文提取完整的4D火灾特征向量
    新增resize_max参数：限制最大边长，大幅提升大分辨率视频的处理速度
    """
    # 可选缩放：限制最大边长，加速计算（论文验证过分辨率≥70x70不影响特征有效性）
    h, w = frame1_bgr.shape[:2]
    if max(h, w) > resize_max:
        scale = resize_max / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        frame1_bgr = cv2.resize(frame1_bgr, (new_w, new_h))
        frame2_bgr = cv2.resize(frame2_bgr, (new_w, new_h))

    # 1. 颜色预处理
    mass1 = rgb_to_generalized_mass(frame1_bgr)
    mass2 = rgb_to_generalized_mass(frame2_bgr)
    gray1 = cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray2 = cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # 2. 双光流计算
    u_nsd, v_nsd = compute_nsd_optical_flow(gray1, gray2, alpha=alpha_nsd)
    u_omt, v_omt = compute_omt_optical_flow(mass1, mass2, alpha=alpha_omt)

    # 绘制 NSD 光流场（和你发的论文图1:1匹配）
    flow_visual = draw_optical_flow_field(
        img=frame1_bgr,  # 用原始帧绘图
        u=u_nsd,
        v=v_nsd,
        step=4,  # 原始分辨率用16，疏密合适、不卡顿
        scale=100,  # 箭头放大倍数
        color=(255, 0, 0)  # 蓝色箭头（论文风格）
    )
    # 显示光流图
    cv2.imshow("NSD Optical Flow Field (论文同款)", flow_visual)
    cv2.waitKey(1)  # 视频实时显示
    # ==================================================================================

    # 3. 有效像素筛选
    essential_mask = get_essential_pixels(u_nsd, v_nsd, c=c_essential)
    if not np.any(essential_mask):
        return np.array([0.0, 0.0, 0.0, 0.0])

    # 4. 4D特征提取
    transport_energy = (mass2 / 2.0) * (u_omt ** 2 + v_omt ** 2)
    f1 = np.mean(transport_energy[essential_mask])

    nsd_magnitude = (1.0 / 2.0) * (u_nsd ** 2 + v_nsd ** 2)
    f2 = np.mean(nsd_magnitude[essential_mask])

    f3 = compute_source_match(u_omt, v_omt)

    f4 = compute_directional_variance(u_nsd, v_nsd, essential_mask)

    # ======================================
    # 🟢 【插入绘图代码：位置100%正确】
    # 自动根据f4判断：f4>0.1为火焰，否则为刚体（可根据你的数据调整阈值）
    # ======================================
    if draw_kde:
        title_suffix = "Fire" if f4 > 0.1 else "Rigid"
        plot_flow_kde_histogram(
            u_nsd=u_nsd,
            v_nsd=v_nsd,
            essential_mask=essential_mask,
            title_suffix=title_suffix, # 按帧号保存，避免覆盖
            frame_idx=frame_index
        )

    return np.array([f1, f2, f3, f4])


# ====================== 新增：视频批量处理核心逻辑 ======================
def process_video_fire_features(
        video_path,
        output_csv_path=None,
        frame_interval=1,
        resize_max=640,
        visualize=False
):
    """
    处理输入视频，逐帧提取论文4D火焰特征，支持.mp4/.avi格式
    :param video_path: 输入视频路径（.mp4/.avi）
    :param output_csv_path: 特征保存路径（.csv），默认保存到视频同目录
    :param frame_interval: 帧间隔，默认1（连续两帧计算），越大处理越快
    :param resize_max: 帧最大边长，限制分辨率加速计算，默认640
    :param visualize: 是否实时显示处理画面和特征值，默认False
    :return: 特征数组、csv保存路径
    """
    # 1. 视频读取初始化
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件：{video_path}，请检查路径或格式是否正确")

    # 视频基础信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # 输出csv路径默认设置
    if output_csv_path is None:
        output_csv_path = f"{video_name}_fire_features.csv"

    # 2. 初始化变量
    feature_list = []
    prev_frame = None
    frame_idx = 0

    # 3. 逐帧处理循环（带进度条）
    pbar = tqdm(total=total_frames, desc=f"处理视频 {video_name}")
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break  # 视频读取完毕

        # 跳过间隔帧，减少计算量
        if frame_idx % frame_interval == 0:
            if prev_frame is not None:
                # 提取4D特征
                features = extract_karasev_features(
                    prev_frame, curr_frame,
                    resize_max=resize_max,
                    frame_index=frame_idx,
                    draw_kde=False
                )
                f1, f2, f3, f4 = features

                # 计算对应时间戳
                timestamp = frame_idx / fps

                # 保存特征和元信息
                feature_item = [
                    frame_idx - frame_interval,  # 起始帧号
                    frame_idx,  # 结束帧号
                    round(timestamp, 3),  # 时间戳(秒)
                    round(f1, 6), round(f2, 6),
                    round(f3, 6), round(f4, 6)
                ]
                feature_list.append(feature_item)

                # 实时可视化
                if visualize:
                    display_frame = curr_frame.copy()
                    # 叠加特征文本
                    text = [
                        f"Time: {timestamp:.2f}s",
                        f"f1(OMT Energy): {f1:.4e}",
                        f"f2(NSD Mag): {f2:.4e}",
                        f"f3(Source Match): {f3:.4e}",
                        f"f4(Direction Var): {f4:.4f}"
                    ]
                    for i, line in enumerate(text):
                        cv2.putText(display_frame, line, (10, 30 + i * 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.imshow("Fire Feature Extraction", display_frame)
                    # 按ESC退出可视化
                    if cv2.waitKey(1) & 0xFF == 27:
                        visualize = False
                        cv2.destroyAllWindows()

        # 更新前一帧和帧号
        prev_frame = curr_frame.copy()
        frame_idx += 1
        pbar.update(1)

    # 4. 资源释放
    pbar.close()
    cap.release()
    cv2.destroyAllWindows()

    # 5. 保存特征到CSV
    csv_header = ["start_frame", "end_frame", "timestamp_s", "f1_omt_energy", "f2_nsd_magnitude", "f3_source_match",
                  "f4_direction_variance"]
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(feature_list)

    print(f"处理完成！共提取 {len(feature_list)} 组特征，已保存到：{output_csv_path}")
    return np.array(feature_list), output_csv_path


# ====================== 使用示例 ======================
if __name__ == "__main__":
    # --------------------------
    # 只需修改这里的视频路径即可运行
    # 支持.mp4/.avi格式，相对路径/绝对路径均可
    # --------------------------
    INPUT_VIDEO_PATH = "part_mp4\car.mp4"  # 替换为你的视频路径
    # INPUT_VIDEO_PATH = "petrochemical_flame.avi"

    # 处理视频
    try:
        feature_array, csv_save_path = process_video_fire_features(
            video_path=INPUT_VIDEO_PATH,
            frame_interval=2,  # 每隔2帧处理一次，平衡速度和精度
            resize_max=640,  # 限制最大分辨率，加速计算
            visualize=True  # 开启实时可视化，按ESC关闭
        )

        # 打印前5组特征示例
        print("\n前5组特征示例：")
        print("帧号\t时间(s)\tf1\t\tf2\t\tf3\t\tf4")
        for item in feature_array[:5]:
            print(f"{int(item[1])}\t{item[2]}\t{item[3]:.4e}\t{item[4]:.4e}\t{item[5]:.4e}\t{item[6]:.4f}")

    except Exception as e:
        print(f"处理失败：{str(e)}")