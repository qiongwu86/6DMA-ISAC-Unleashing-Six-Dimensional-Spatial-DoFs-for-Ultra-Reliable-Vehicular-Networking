import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
import matplotlib.ticker as ticker  # 用于精细控制坐标轴

from qbub import SurfaceLayoutGenerator
from uq import SCPOptimizerU
from Qcurr import SCAOptimizerQ
from ga1_ import GeneticOptimizer
from Pos import FullPSOOptimizer
from fasteru import FastSCPOptimizerU
    # from fasterq import FastSCAOptimizerQ # 不再需要，因为位置固定

# --- 绘图配置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
warnings.filterwarnings("ignore")


# ==============================================================================
# 1. 物理引擎 (ChannelEngine)
#    - 修正了 V=0 问题
#    - 修正了 Rs 量纲问题
# ==============================================================================
class ChannelEngine:
    def __init__(self, B=16, theta0=np.pi / 6, V=20.0):
        self.B = B
        self.H = 100.0
        self.R = 200.0

        # --- 修正点 1: V=0 保护 ---
        if V < 1e-3:
            self.V = 0.0
            self.T_total = 10.0  # 静止时仿真 10秒用于取平均
        else:
            self.V = V
            self.T_total = 2 * self.R / self.V

        self.THETA_0 = theta0
        self.PC = 40.0 * 1e-3
        self.PS = 40.0 * 1e-3
        self.SIGMA_SQ = (10 ** (-90.0 / 10.0)) * 1e-3
        self.L_FRAME = 0.1
        self.LAMBDA = 0.125
        self.N_ELEM = 4
        self.EPSILON = 2
        self.EPSILON1 = 4
        self.RHO = 0.8
        self.THETA_3DB = np.deg2rad(65)
        self.PHI_3DB = np.deg2rad(65)
        self.G_MAX = 8
        self.G_S = 25
        self.G_V = 25

    def _get_rotation_matrix(self, u_b):
        a, b, g = u_b
        return np.array([[np.cos(b) * np.cos(g), np.cos(b) * np.sin(g), -np.sin(b)],
                         [np.sin(b) * np.sin(a) * np.cos(g) - np.cos(a) * np.sin(g),
                          np.sin(b) * np.sin(a) * np.sin(g) + np.cos(a) * np.cos(g), np.cos(b) * np.sin(a)],
                         [np.cos(a) * np.sin(b) * np.cos(g) + np.sin(a) * np.sin(g),
                          np.cos(a) * np.sin(b) * np.sin(g) - np.sin(a) * np.cos(g), np.cos(a) * np.cos(b)]])

    def compute_rates(self, Q, U, t, eta):
        Q = Q.reshape(self.B, 3)
        U = U.reshape(self.B, 3)
        target_pos = np.array([self.R * np.cos(self.THETA_0), self.R * np.sin(self.THETA_0) - self.V * t, self.H])
        pc_vals = np.zeros(self.B)
        ps_vals = np.zeros(self.B)

        for b in range(self.B):
            vec = target_pos - Q[b]
            dist = max(np.linalg.norm(vec), 1.0)
            R_mat = self._get_rotation_matrix(U[b])
            v_loc = np.dot(R_mat.T, vec / dist)
            theta_dev = np.arccos(np.clip(v_loc[2], -1, 1))
            phi_tilde = np.arctan2(v_loc[1], v_loc[0])
            val_p = 12 * (np.abs(phi_tilde) / self.PHI_3DB) ** 2
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
        # --- 修正点 2: 去掉 1/L_FRAME，使其成为 bps/Hz 量级 ---
        rs = np.sum(np.log2(1 + sens_snr)) if len(sens_snr) > 0 else 0

        return 0.5 * rc + 0.5 * rs

    def get_trajectory_average_rate(self, Q, U, eta=0.5):
        t_samples = np.arange(0, self.T_total, self.L_FRAME)
        if len(t_samples) == 0: t_samples = np.array([0.0])
        total = 0
        for t in t_samples: total += self.compute_rates(Q, U, t, eta)
        return total / len(t_samples)


# ==============================================================================
# 2. 新增优化类: CircularRailSCPOptimizer
#    - 替代原有的 Math 算法
#    - 实现固定 Q + 优化 U + 修正下倾角
# ==============================================================================
class CircularRailSCPOptimizer:
    def __init__(self, B, R, H, V, theta0_global, eta, beta_deg=15.0):
        self.B = B
        self.R = R
        self.H = H
        self.V = V
        self.theta0 = theta0_global
        self.eta = eta
        self.beta_rad = np.radians(beta_deg)

    def _generate_circular_q(self):
        """生成严格的圆形分布位置"""
        phase_offsets = np.linspace(0, 2 * np.pi, self.B, endpoint=False)
        psi_angles = phase_offsets + self.theta0
        x = self.R * np.cos(psi_angles)
        y = self.R * np.sin(psi_angles)
        z = np.full(self.B, self.H)
        return np.stack([x, y, z], axis=1)

    def _enforce_constraints(self, U_opt):
        """修正下倾角"""
        U_corrected = np.zeros_like(U_opt)
        r_xy = np.cos(self.beta_rad)
        z_fixed = -np.sin(self.beta_rad)

        for b in range(self.B):
            u_vec = U_opt[b]
            phi = np.arctan2(u_vec[1], u_vec[0])
            U_corrected[b, 0] = r_xy * np.cos(phi)
            U_corrected[b, 1] = r_xy * np.sin(phi)
            U_corrected[b, 2] = z_fixed
        return U_corrected

    def optimize(self, U_init):
        # 1. 生成固定的 Q
        Q_fixed = self._generate_circular_q()

        try:
            # 2. 调用 FastSCPOptimizerU
            try:
                optimizer_u = FastSCPOptimizerU(Q_fixed, self.eta, self.theta0)
            except:
                optimizer_u = FastSCPOptimizerU(Q_fixed, self.eta)
                optimizer_u.theta0 = self.theta0

            # 注入参数
            optimizer_u.Q = Q_fixed
            optimizer_u.V = self.V

            # 优化
            U_raw = np.array(optimizer_u.optimize(U_init.copy())).reshape(self.B, 3)

            # 3. 修正下倾角
            U_final = self._enforce_constraints(U_raw)

            return Q_fixed, U_final

        except Exception as e:
            # print(f"SCP Error: {e}")
            return Q_fixed, U_init


# ==============================================================================
# 3. 主程序 (速率扫描)
# ==============================================================================
if __name__ == "__main__":
    # --- 参数 ---
    B = 16
    C = 20.0
    H = 100.0
    R = 200.0
    THETA_GLOBAL = np.deg2rad(60)
    ETA_VAL = 0.9

    # --- 修改点 3: 扫描范围 0-30，步长 5 ---
    speeds = np.arange(0, 31, 5)

    NUM_MC = 10
    avg_results = np.zeros((4, len(speeds)))

    print(f"=== 开始车速扫描 (V=0~30, Step=5) x {NUM_MC} 次蒙特卡洛 ===")
    print("算法: 1.Initial, 2.Math(Circ+SCP), 3.PSO, 4.GA")

    total_start = time.time()

    for mc in range(NUM_MC):
        if (mc + 1) % 2 == 0 or mc == 0:
            print(f">>> MC Run {mc + 1} / {NUM_MC} ...")

        for i_v, v_val in enumerate(speeds):

            # 初始化引擎
            current_engine = ChannelEngine(B=B, theta0=THETA_GLOBAL, V=v_val)
            current_gen = SurfaceLayoutGenerator(B=B, C=C, theta_0=THETA_GLOBAL, H=H, R=R, V=v_val)

            # --- A. Initial ---
            Q_init, U_init = current_gen.generate()
            Q_init = Q_init.reshape(B, 3)
            U_init = U_init.reshape(B, 3)

            rate_init = current_engine.get_trajectory_average_rate(Q_init, U_init, eta=ETA_VAL)
            if np.isnan(rate_init): rate_init = 0
            avg_results[0, i_v] += rate_init

            # --- B. Math (Circular Rail + SCP U) ---
            # 使用新写的混合优化器
            try:
                circ_opt = CircularRailSCPOptimizer(
                    B, R, H, v_val, THETA_GLOBAL, ETA_VAL, beta_deg=15.0
                )
                # 不需要 Q_init, 它会自己生成圆形 Q
                Q_math, U_math = circ_opt.optimize(U_init.copy())

                rate_math = current_engine.get_trajectory_average_rate(Q_math, U_math, eta=ETA_VAL)
                avg_results[1, i_v] += rate_math
            except Exception as e:
                # print(f"Math Error: {e}")
                avg_results[1, i_v] += rate_init

            # --- C. PSO ---
            try:
                try:
                    pso_opt = FullPSOOptimizer(eta=ETA_VAL, theta0=THETA_GLOBAL)
                except:
                    pso_opt = FullPSOOptimizer(eta=ETA_VAL)

                sim_v = v_val if v_val > 1e-3 else 0.001
                pso_opt.theta0 = THETA_GLOBAL
                pso_opt.V = sim_v
                pso_opt.T_total = 2 * R / sim_v
                if hasattr(pso_opt, 'time_points_opt'):
                    pso_opt.time_points_opt = np.linspace(0, pso_opt.T_total, 10)

                Q_pso, U_pso, _ = pso_opt.optimize(Q_init=Q_init, U_init=U_init)
                rate_pso = current_engine.get_trajectory_average_rate(Q_pso, U_pso, eta=ETA_VAL)
                avg_results[2, i_v] += rate_pso
            except Exception as e:
                avg_results[2, i_v] += rate_init

            # --- D. GA ---
            try:
                try:
                    ga_opt = GeneticOptimizer(B, C, H, R, V=v_val, theta0=THETA_GLOBAL, eta=ETA_VAL)
                except:
                    ga_opt = GeneticOptimizer(B, C, H, R, V=20.0, theta0=THETA_GLOBAL, eta=ETA_VAL)

                sim_v = v_val if v_val > 1e-3 else 0.001
                ga_opt.V = sim_v
                ga_opt.theta0 = THETA_GLOBAL
                if hasattr(ga_opt, 'T_total'): ga_opt.T_total = 2 * R / sim_v

                Q_ga, U_ga, _ = ga_opt.optimize(Q_init=Q_init, U_init=U_init)
                rate_ga = current_engine.get_trajectory_average_rate(Q_ga, U_ga, eta=ETA_VAL)
                avg_results[3, i_v] += rate_ga
            except Exception as e:
                avg_results[3, i_v] += rate_init

    # 取平均
    avg_results /= NUM_MC
    print(f"\n全部实验完成，耗时: {time.time() - total_start:.2f}s")

    # ==============================================================================
    # 4. 绘图
    # ==============================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    algo_names = ['Initial', 'Circular+SCP', 'PSO', 'GA']
    styles = ['k:o', 'r-s', 'b--^', 'g-.d']

    for i in range(4):
        ax.plot(speeds, avg_results[i], styles[i], linewidth=2.5, markersize=6, label=algo_names[i], alpha=0.85)

    ax.set_xlabel('Velocity (km/h)', fontsize=12)
    ax.set_ylabel('Average Rate (bps/Hz)', fontsize=12)

    # --- 修改点 4: 强制 X 轴从 0 开始，设置分度 ---
    ax.set_xlim(0, 30)
    ax.set_xticks(np.arange(0, 31, 5))

    # 设置 Y 轴自动范围，保留一点边距
    y_min = np.min(avg_results)
    y_max = np.max(avg_results)
    ax.set_ylim(y_min * 0.9, y_max * 1.1)

    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=12, loc='best', framealpha=0.9, shadow=True)

    plt.tight_layout()
    plt.show()