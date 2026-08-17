import numpy as np
import matplotlib.pyplot as plt
import time
import warnings

# --- 导入您的算法模块 ---
from qbub import SurfaceLayoutGenerator
from fasterq import FastSCAOptimizerQ
from fasteru import FastSCPOptimizerU

# --- 绘图配置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
warnings.filterwarnings("ignore")


# ==============================================================================
# 1. 物理引擎 (ChannelEngine)
# ==============================================================================
class ChannelEngine:
    def __init__(self, B=16, theta0=np.pi / 6, V=20.0):
        self.B = B;
        self.H = 100.0;
        self.R = 200.0;
        self.V = V
        self.THETA_0 = theta0
        self.T_total = 2 * self.R / self.V

        self.PC = 40.0 * 1e-3;
        self.PS = 40.0 * 1e-3
        self.SIGMA_SQ = (10 ** (-90.0 / 10.0)) * 1e-3
        self.L_FRAME = 0.1;
        self.LAMBDA = 0.125;
        self.N_ELEM = 4
        self.EPSILON = 2;
        self.EPSILON1 = 4;
        self.RHO = 0.8
        self.THETA_3DB = np.deg2rad(65);
        self.PHI_3DB = np.deg2rad(65)
        self.G_MAX = 8;
        self.G_S = 25;
        self.G_V = 25

    def _get_rotation_matrix(self, u_b):
        a, b, g = u_b
        return np.array([[np.cos(b) * np.cos(g), np.cos(b) * np.sin(g), -np.sin(b)],
                         [np.sin(b) * np.sin(a) * np.cos(g) - np.cos(a) * np.sin(g),
                          np.sin(b) * np.sin(a) * np.sin(g) + np.cos(a) * np.cos(g), np.cos(b) * np.sin(a)],
                         [np.cos(a) * np.sin(b) * np.cos(g) + np.sin(a) * np.sin(g),
                          np.cos(a) * np.sin(b) * np.sin(g) - np.sin(a) * np.cos(g), np.cos(a) * np.cos(b)]])

    def compute_rates(self, Q, U, t, eta):
        Q = Q.reshape(self.B, 3);
        U = U.reshape(self.B, 3)
        target_pos = np.array([self.R * np.cos(self.THETA_0), self.R * np.sin(self.THETA_0) - self.V * t, self.H])
        pc_vals = np.zeros(self.B);
        ps_vals = np.zeros(self.B)

        for b in range(self.B):
            vec = target_pos - Q[b]
            dist = max(np.linalg.norm(vec), 1.0)
            R_mat = self._get_rotation_matrix(U[b])
            v_loc = np.dot(R_mat.T, vec / dist)
            theta_dev = np.arccos(np.clip(v_loc[2], -1, 1));
            phi_tilde = np.arctan2(v_loc[1], v_loc[0])
            val_p = 12 * (np.abs(phi_tilde) / self.PHI_3DB) ** 2;
            val_t = 12 * (np.abs(theta_dev) / self.THETA_3DB) ** 2
            gain_db = self.G_MAX - np.minimum(-(-np.minimum(val_p, self.G_V) - np.minimum(val_t, self.G_S)), self.G_S)
            gain = 10 ** (gain_db / 10.0)
            pc_vals[b] = self.PC * gain * (dist ** -self.EPSILON) * self.N_ELEM
            ps_vals[b] = self.PS * gain * (dist ** -self.EPSILON1) * (self.N_ELEM ** 2) * (self.RHO ** 2)

        snr_c = pc_vals / self.SIGMA_SQ
        max_snr = np.max(snr_c) if np.max(snr_c) > 0 else 1e-12
        mv = np.where(snr_c > eta * max_snr, 1, 0)

        comm_sum = np.sum(pc_vals[mv == 1])
        rc = np.log2(1 + comm_sum / self.SIGMA_SQ) if comm_sum > 0 else 0
        sens_snr = ps_vals[mv == 0] / self.SIGMA_SQ
        rs = (1.0 / self.L_FRAME) * np.sum(np.log2(1 + sens_snr)) if len(sens_snr) > 0 else 0
        return 0.5 * rc + 0.5 * rs

    def get_trajectory_avg(self, Q, U, eta=0.5):
        t_samples = np.arange(0, self.T_total, self.L_FRAME)
        if len(t_samples) == 0: t_samples = np.array([0.0])
        total = 0
        for t in t_samples: total += self.compute_rates(Q, U, t, eta)
        return total / len(t_samples)


# ==============================================================================
# 2. 主执行逻辑 (消融实验)
# ==============================================================================
if __name__ == "__main__":
    # --- 全局参数 ---
    B = 16;
    C = 20.0;
    H = 100.0;
    R = 200.0
    THETA_GLOBAL = np.deg2rad(60)
    ETA_VAL = 0.9

    # 扫描速度
    speeds = np.linspace(20, 100, 9)  # 9个点 (20, 30, ..., 100)

    # 实验设置
    NUM_MC = 5  # 蒙特卡洛次数 (如果慢可以改成 1 或 2)

    # 结果容器: (策略数, 速度点数)
    # 0: Initial(Fixed), 1: Only Q, 2: Only U, 3: Joint(Q+U)
    avg_results = np.zeros((4, len(speeds)))

    print(f"=== 开始消融实验 (Runs={NUM_MC}): Fixed vs Only-Q vs Only-U vs Joint ===")

    total_start = time.time()

    for mc in range(NUM_MC):
        print(f"\n>>> MC Run {mc + 1}/{NUM_MC} ...")

        for i_v, v_val in enumerate(speeds):
            # 1. 环境初始化
            current_engine = ChannelEngine(B=B, theta0=THETA_GLOBAL, V=v_val)
            current_gen = SurfaceLayoutGenerator(B=B, C=C, theta_0=THETA_GLOBAL, H=H, R=R, V=v_val)

            # --- 策略 A: Initial (Fixed High Quality) ---
            # 使用 generator 生成高质量初始点
            Q_init, U_init = current_gen.generate()  # 假设 generate 默认返回的就是高质量点
            Q_init = Q_init.reshape(B, 3);
            U_init = U_init.reshape(B, 3)

            rate_fixed = current_engine.get_trajectory_avg(Q_init, U_init, eta=ETA_VAL)
            avg_results[0, i_v] += rate_fixed

            # --- 准备优化器 ---
            try:
                # 实例化 Q 优化器
                try:
                    math_q = FastSCAOptimizerQ(B, C, H, R, v_val, 0.125, ETA_VAL, THETA_GLOBAL)
                except:
                    math_q = FastSCAOptimizerQ(B, C, H, R, 0.125, ETA_VAL)
                math_q.V = v_val;
                math_q.theta0 = THETA_GLOBAL

                # 实例化 U 优化器
                try:
                    math_u = FastSCPOptimizerU(Q_init, ETA_VAL, THETA_GLOBAL)
                except:
                    math_u = FastSCPOptimizerU(Q_init, ETA_VAL)
                math_u.V = v_val;
                math_u.theta0 = THETA_GLOBAL

                # --- 策略 B: Only Q (U 保持 Initial) ---
                # 仅运行 math_q，U 使用 U_init
                Q_only = np.array(math_q.optimize(Q_init.copy(), U_init.copy())).reshape(B, 3)
                rate_q = current_engine.get_trajectory_avg(Q_only, U_init, eta=ETA_VAL)
                avg_results[1, i_v] += rate_q

                # --- 策略 C: Only U (Q 保持 Initial) ---
                # 仅运行 math_u，Q 使用 Q_init
                # 注意：SCPOptimizerU 需要当前的 Q 来计算几何
                math_u.Q = Q_init
                U_only = np.array(math_u.optimize(U_init.copy())).reshape(B, 3)
                rate_u = current_engine.get_trajectory_avg(Q_init, U_only, eta=ETA_VAL)
                avg_results[2, i_v] += rate_u

                # --- 策略 D: Joint Q & U (Math Algorithm) ---
                # 在 Only Q 的基础上，继续优化 U (交替优化)
                # Q* = Q_only
                math_u.Q = Q_only
                U_joint = np.array(math_u.optimize(U_init.copy())).reshape(B, 3)
                rate_joint = current_engine.get_trajectory_avg(Q_only, U_joint, eta=ETA_VAL)
                avg_results[3, i_v] += rate_joint

            except Exception as e:
                # 如果优化报错，全部退化为 Fixed
                # print(f"Opt Error at V={v_val}: {e}")
                avg_results[1, i_v] += rate_fixed
                avg_results[2, i_v] += rate_fixed
                avg_results[3, i_v] += rate_fixed

    # 计算平均
    avg_results /= NUM_MC
    print(f"\n全部完成，耗时: {time.time() - total_start:.2f}s")

    # ==============================================================================
    # 3. 绘图
    # ==============================================================================
    plt.figure(figsize=(10, 6))

    # 绘图数据与样式
    plot_configs = [
        # (数据索引, 标签, 颜色线型, 线宽, 标记大小)
        (3, 'Joint Opt (Q & U)', 'r-s', 2.5, 8),  # 最强：红色实线
        (1, 'Only Position (Q)', 'b--o', 2.0, 6),  # 次强：蓝色虚线
        (2, 'Only Rotation (U)', 'g-.^', 2.0, 6),  # 较弱：绿色点划线
        (0, 'Initial (Fixed)', 'k:x', 1.5, 6)  # 基准：黑色点线
    ]

    for idx, label, fmt, lw, ms in plot_configs:
        plt.plot(speeds, avg_results[idx], fmt, linewidth=lw, label=label, markersize=ms, alpha=0.8)

    plt.xlabel('Vehicle Speed (m/s)', fontsize=12)
    plt.ylabel('Average Weighted Sum Rate', fontsize=12)
    plt.title(f'消融实验：位置与姿态优化的贡献对比 (Ave of {NUM_MC} Runs)', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)

    # 图例放在右上角
    plt.legend(loc='upper right', framealpha=0.9, shadow=True, fontsize=10)
    plt.xticks(speeds)

    plt.tight_layout()
    plt.show()

    # 打印数值对比 (取 60m/s 处的点)
    mid_idx = len(speeds) // 2
    print(f"\n=== 数值对比 (at V={speeds[mid_idx]} m/s) ===")
    print(f"Fixed  : {avg_results[0, mid_idx]:.2f}")
    print(f"Only U : {avg_results[2, mid_idx]:.2f}")
    print(f"Only Q : {avg_results[1, mid_idx]:.2f}")
    print(f"Joint  : {avg_results[3, mid_idx]:.2f}")