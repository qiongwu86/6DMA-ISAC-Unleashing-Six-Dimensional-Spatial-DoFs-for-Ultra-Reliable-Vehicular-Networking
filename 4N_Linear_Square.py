import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# 1. 环境变量与参数定义
# ==========================================
class ISAC_6DMA_Config:
    def __init__(self):
        # 核心物理参数
        self.lam = 0.125  # 波长 = 0.125m
        self.d = 0.5 * self.lam  # 阵元间距 = 0.0625m (半波长)
        self.k = 2 * np.pi / self.lam  # 波数

        # 阵列规模：正方形 2x2 意味着垂直方向有 2 个阵元
        self.N_vert = 2
        self.min_db = -40  # 绘图动态范围极限


# ==========================================
# 2. 垂直增益计算函数
# ==========================================
def compute_vertical_gain(config, theta_vec):
    """
    计算垂直切面的阵列方向图
    theta_vec: 弧度制角度向量
    """
    # 1. 阵列因子 (Array Factor)
    # 对于侧射阵(Broadside)，相位差 psi = k * d * sin(theta)
    # 当 theta = 0 时，指向水平正前方
    psi = config.k * config.d * np.sin(theta_vec)

    # 典型的均匀直线阵 (ULA) 公式
    numerator = np.sin(config.N_vert * psi / 2)
    denominator = config.N_vert * np.sin(psi / 2) + 1e-12
    af = np.abs(numerator / denominator)

    # 2. 单元因子 (Element Factor)
    # 模拟实际天线单元的定向性：只向前方(x轴正半轴)辐射
    # cos(theta) 模型是业界常用的垂直面单阵元模拟方式
    angles_norm = np.arctan2(np.sin(theta_vec), np.cos(theta_vec))
    is_front = np.abs(angles_norm) <= (np.pi / 2)

    # 给予背面信号极大的衰减，主瓣指向 0 度
    ef = np.where(is_front, np.cos(angles_norm) ** 1.5, 1e-4)

    # 3. 总增益计算 (归一化 dB)
    total_gain = af * ef
    gain_db = 20 * np.log10(np.clip(total_gain, 1e-6, 1.0))

    return np.maximum(gain_db, config.min_db)


# ==========================================
# 3. 绘图与可视化
# ==========================================
def plot_vertical_pattern():
    config = ISAC_6DMA_Config()
    theta = np.linspace(0, 2 * np.pi, 1200)

    # 计算 2x2 正方形布局的垂直增益
    gain_2x2 = compute_vertical_gain(config, theta)

    # 为了对比，计算 1x4 线性布局(垂直只有1层)的增益
    # N_vert = 1 时的垂直方向图
    temp_n = config.N_vert
    config.N_vert = 1
    gain_1x4 = compute_vertical_gain(config, theta)
    config.N_vert = temp_n  # 恢复

    # 开始绘图
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(8, 8), dpi=120)
    ax = fig.add_subplot(111, projection='polar')

    # 绘制线性排布 (垂直方向无聚焦)
    ax.plot(theta, gain_1x4, color='gray', linestyle='--', linewidth=2, label='线性排布 (1x4) - 垂直无增益')

    # 绘制正方形排布 (垂直方向 2 层聚焦)
    ax.plot(theta, gain_2x2, color='#d62728', linewidth=3, label='正方形排布 (2x2) - 垂直初步聚焦')
    ax.fill(theta, gain_2x2, color='#d62728', alpha=0.15)

    # 图表细节配置
    ax.set_theta_zero_location('E')  # 0度朝向水平右侧
    ax.set_rlim(config.min_db, 2)
    ax.set_rticks([-10, -20, -30])
    ax.set_yticklabels(['-10dB', '-20dB', '-30dB'], color='gray', fontsize=10)

    title_str = f"垂直天线辐射方向图对比\n($\lambda={config.lam}m, d=0.5\lambda$)"
    ax.set_title(title_str, fontsize=14, fontweight='bold', pad=30)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.7)

    # 物理特性说明标注
    ax.text(np.deg2rad(45), -15, "正方形排布优势:\n垂直主瓣收缩\n能量更集中",
            color='#d62728', fontweight='bold', ha='center')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_vertical_pattern()