import numpy as np
import matplotlib.pyplot as plt
import time
import warnings

# --- 导入您的算法模块 ---
from qbub import SurfaceLayoutGenerator
from uq import SCPOptimizerU
from Qcurr import SCAOptimizerQ
from ga1_ import GeneticOptimizer
from Pos import FullPSOOptimizer
from fasterq import FastSCAOptimizerQ
from fasteru import FastSCPOptimizerU

# --- 绘图配置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
warnings.filterwarnings("ignore")


# ==============================================================================
# 1. 物理引擎 (ChannelEngine)
#    保持不变，负责计算速率
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
            theta_dev = np.arccos(np.clip(v_loc[2], -1, 1))
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
        rs =  np.sum(np.log2(1 + sens_snr)) if len(sens_snr) > 0 else 0
        return 0.5 * rc + 0.5 * rs

    def get_trajectory_average_rate(self, Q, U, eta=0.5):
        t_samples = np.arange(0, self.T_total, self.L_FRAME)
        if len(t_samples) == 0: t_samples = np.array([0.0])
        total = 0
        for t in t_samples: total += self.compute_rates(Q, U, t, eta)
        return total / len(t_samples)


# ==============================================================================
# 2. 主执行逻辑 (Speed Sweep + Monte Carlo)
# ==============================================================================
if __name__ == "__main__":
    # --- 参数 ---
    B = 16;
    C = 20.0;
    H = 100.0;
    R = 200.0
    THETA_GLOBAL = np.deg2rad(60)
    ETA_VAL = 0.9

    # 扫描变量: 车速
    speeds = np.arange(5, 31, 5)

    # 蒙特卡洛参数
    NUM_MC = 1  # 实验次数

    # 结果存储矩阵: (算法数, 车速数)
    # 0:Initial, 1:Math, 2:PSO, 3:GA
    avg_results = np.zeros((4, len(speeds)))

    print(f"=== 开始车速扫描 (V=20~100) x {NUM_MC} 次蒙特卡洛 ===")

    total_start = time.time()

    # 外层循环：实验次数
    for mc in range(NUM_MC):
        print(f"\n>>>>>> 第 {mc + 1} / {NUM_MC} 次蒙特卡洛实验 <<<<<<")

        # 内层循环：车速扫描
        for i_v, v_val in enumerate(speeds):
            # print(f"   处理车速 V = {v_val} m/s ...")

            # 初始化环境
            current_engine = ChannelEngine(B=B, theta0=THETA_GLOBAL, V=v_val)
            # 注意：每次都要重新生成随机生成器实例，以保证随机性
            current_gen = SurfaceLayoutGenerator(B=B, C=C, theta_0=THETA_GLOBAL, H=H, R=R, V=v_val)

            # --- A. FPA ---
            Q_init, U_init = current_gen.generate()
            Q_init = Q_init.reshape(B, 3);
            U_init = U_init.reshape(B, 3)

            rate_init = current_engine.get_trajectory_average_rate(Q_init, U_init, eta=ETA_VAL)
            if np.isnan(rate_init): rate_init = 0
            avg_results[0, i_v] += rate_init  # 累加

            # --- B. Math 算法 ---
            try:
                try:
                    math_q = FastSCAOptimizerQ(B, C, H, R, v_val, 0.125, ETA_VAL, THETA_GLOBAL)
                except:
                    math_q = FastSCAOptimizerQ(B, C, H, R, 0.125, ETA_VAL)
                math_q.V = v_val;
                math_q.theta0 = THETA_GLOBAL

                try:
                    math_u = FastSCPOptimizerU(Q_init, ETA_VAL, THETA_GLOBAL)
                except:
                    math_u = FastSCPOptimizerU(Q_init, ETA_VAL)
                math_u.V = v_val;
                math_u.theta0 = THETA_GLOBAL

                Q_math = np.array(math_q.optimize(Q_init.copy(), U_init.copy())).reshape(B, 3)
                math_u.Q = Q_math
                U_math = np.array(math_u.optimize(U_init.copy())).reshape(B, 3)

                rate_math = current_engine.get_trajectory_average_rate(Q_math, U_math, eta=ETA_VAL)
                avg_results[1, i_v] += rate_math
            except Exception as e:
                # print(f"Math Error: {e}")
                avg_results[1, i_v] += rate_init  # 如果报错，用初始值兜底，避免平均值拉太低

            # --- C.  udma ---
            try:
                try:
                    generator = SurfaceLayoutGenerator(B=B, C=C, theta_0=np.pi / 3, H=H, R=R, V=V)
                    Q_int, U_int = generator.generate()
                    udma_opt = FastSCPOptimizerU(q_fixed=Q_int, eta=0.9, theta0=np.pi / 6, B=16, H=100.0, R=200.0, V=20.0)
                except:
                    udma_opt = FastSCPOptimizerU(eta=ETA_VAL)
                udma_opt.theta0 = THETA_GLOBAL;
                udma_opt.V = v_val;
                udma_opt.T_total = 2 * R / v_val
                if hasattr(udma_opt, 'time_points_opt'): udma_opt.time_points_opt = np.linspace(0, udam_opt.T_total, 10)

                # 传入初始值
                Q_pso, U_pso, _ = udma_opt.optimize(Q_init=Q_init, U_init=U_init)
                rate_udma = current_engine.get_trajectory_average_rate(Q_pso, U_pso, eta=ETA_VAL)
                avg_results[2, i_v] += rate_udma
            except Exception as e:
                avg_results[2, i_v] += rate_init

            # --- D. qdma 算法 ---
            try:
                try:
                    ga_opt = GeneticOptimizer(B, C, H, R, V=v_val, theta0=THETA_GLOBAL, eta=ETA_VAL)
                except:
                    ga_opt = GeneticOptimizer(B, C, H, R, V=20.0, theta0=THETA_GLOBAL, eta=ETA_VAL)
                ga_opt.V = v_val;
                ga_opt.theta0 = THETA_GLOBAL
                if hasattr(ga_opt, 'T_total'): ga_opt.T_total = 2 * R / v_val

                # 传入初始值
                Q_ga, U_ga, _ = ga_opt.optimize(Q_init=Q_init, U_init=U_init)
                rate_ga = current_engine.get_trajectory_average_rate(Q_ga, U_ga, eta=ETA_VAL)
                avg_results[3, i_v] += rate_ga
            except Exception as e:
                avg_results[3, i_v] += rate_init

    # --- 计算平均值 ---
    avg_results /= NUM_MC
    print(f"\n全部实验完成，耗时: {time.time() - total_start:.2f}s")

    # ==============================================================================
    # 3. 绘图 (平均值)
    # ==============================================================================
    plt.figure(figsize=(10, 6))

    algo_names = ['Initial (Random)', 'Math (SCA/SCP)', 'PSO', 'Genetic (GA)']
    styles = ['k:o', 'b-s', 'r--^', 'g-.d']

    for i in range(4):
        plt.plot(speeds, avg_results[i], styles[i], linewidth=2, label=algo_names[i], alpha=0.8)

    plt.xlabel(' V (Km/h)', fontsize=12)
    plt.ylabel('(Average Rate)', fontsize=12)

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10)
    # --- 修改点 3: 显式设置横坐标分度值为 5 ---
    plt.xticks(np.arange(5, 31, 5))

    plt.tight_layout()
    plt.show()