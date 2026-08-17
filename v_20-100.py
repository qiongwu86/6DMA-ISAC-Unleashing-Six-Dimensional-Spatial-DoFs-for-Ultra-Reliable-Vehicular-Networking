import numpy as np
import matplotlib.pyplot as plt
import time
import warnings

# --- 导入您的算法模块 ---
# 请确保这些文件在当前目录下
from qbub import SurfaceLayoutGenerator
from uq import SCPOptimizerU
from Qcurr import SCAOptimizerQ
from ga1_ import GeneticOptimizer
from  Pos import FullPSOOptimizer
from fasterq import FastSCAOptimizerQ
from fasteru import FastSCPOptimizerU

# --- 绘图配置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
warnings.filterwarnings("ignore")


# ==============================================================================
# 1. 物理引擎 (ChannelEngine)
#    配置：N=4, L=0.1, 感知公式=累加型
# ==============================================================================
class ChannelEngine:
    def __init__(self, B=16, theta0=np.pi / 6, V=20.0):
        self.B = B
        self.H = 100.0;
        self.R = 200.0;
        self.V = V  # 【关键】车速是变量
        self.THETA_0 = theta0

        # 总时间 = 路程 / 速度 (假设走完 2R 的水平投影或根据具体几何)
        # 这里维持原逻辑：T = 2R/V
        self.T_total = 2 * self.R / self.V

        # 物理参数
        self.PC = 40.0 * 1e-3
        self.PS = 40.0 * 1e-3
        self.SIGMA_SQ = (10 ** (-90.0 / 10.0)) * 1e-3

        self.L_FRAME = 0.1  # 感知帧长
        self.LAMBDA = 0.125
        self.N_ELEM = 4  # 【关键】天线数量保持为 4

        self.EPSILON = 2;
        self.EPSILON1 = 4
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
            val_p = 12 * (np.abs(phi_tilde) / self.PHI_3DB) ** 2
            val_t = 12 * (np.abs(theta_dev) / self.THETA_3DB) ** 2
            gain_db = self.G_MAX - np.minimum(-(-np.minimum(val_p, self.G_V) - np.minimum(val_t, self.G_S)), self.G_S)
            gain = 10 ** (gain_db / 10.0)

            pc_vals[b] = self.PC * gain * (dist ** -self.EPSILON) * self.N_ELEM
            ps_vals[b] = self.PS * gain * (dist ** -self.EPSILON1) * (self.N_ELEM ** 2) * (self.RHO ** 2)

        snr_c = pc_vals / self.SIGMA_SQ
        max_snr = np.max(snr_c) if np.max(snr_c) > 0 else 1e-12
        mv = np.where(snr_c > eta * max_snr, 1, 0)

        # 1. 通信 (Log Sum P)
        comm_sum = np.sum(pc_vals[mv == 1])
        rc = np.log2(1 + comm_sum / self.SIGMA_SQ) if comm_sum > 0 else 0

        # 2. 感知 (Sum Log P) - 空间分集累加
        sens_snr = ps_vals[mv == 0] / self.SIGMA_SQ
        if len(sens_snr) > 0:
            rs = (1.0 / self.L_FRAME) * np.sum(np.log2(1 + sens_snr))
        else:
            rs = 0

        # 简单加权
        return 0.5 * rc + 0.5 * rs

    def get_trajectory_average_rate(self, Q, U, eta=0.5):
        # 使用物理帧长采样
        t_samples = np.arange(0, self.T_total, self.L_FRAME)
        if len(t_samples) == 0: t_samples = np.array([0.0])

        total = 0
        for t in t_samples:
            total += self.compute_rates(Q, U, t, eta)
        return total / len(t_samples)


# ==============================================================================
# 2. 主执行逻辑 (Speed Sweep)
# ==============================================================================
if __name__ == "__main__":
    # --- 固定参数 ---
    B = 16
    C = 20.0
    H = 100.0;
    R = 200.0
    THETA_GLOBAL = np.deg2rad(60)  # 初始角度 60度
    ETA_VAL = 0.9  # 模式选择阈值

    # --- 变量：车速 (20 ~ 100 m/s) ---
    speeds = np.arange(20, 100, 10)
    # [20, 30, 40, ..., 100]

    # 结果存储
    results_db = {
        'Initial (Unoptimized)': [],
        'Math (SCA/SCP)': [],
        'PSO': [],
        'Genetic (GA)': []
    }

    print(f"=== 开始车速扫描分析 (V=20~100) ===")
    print(f"固定参数: Angle=60°, Eta={ETA_VAL}, N_ELEM=4")

    total_start = time.time()

    # ... (前面的 ChannelEngine 和 import 保持不变) ...

    # ==============================================================================
    # 2. 主执行逻辑 (Speed Sweep) - [修复版]
    # ==============================================================================
    print(f"=== 开始车速扫描分析 (V=20~100) ===")

    for v_val in speeds:
        print(f"\n>>> 处理车速 V = {v_val} m/s ...")

        # 1. 初始化环境
        current_engine = ChannelEngine(B=B, theta0=THETA_GLOBAL, V=v_val)
        current_gen = SurfaceLayoutGenerator(B=B, C=C, theta_0=THETA_GLOBAL, H=H, R=R, V=v_val)

        # ==========================================
        # A. 初始值 (Initial) - [排查 0 的原因]
        # ==========================================
        Q_init, U_init = current_gen.generate()

        # 强制检查形状
        Q_init = Q_init.reshape(B, 3)
        U_init = U_init.reshape(B, 3)

        rate_init = current_engine.get_trajectory_average_rate(Q_init, U_init, eta=ETA_VAL)

        # 检查是否为 NaN
        if np.isnan(rate_init):
            print("   [Init] 警告: 初始速率计算结果为 NaN (可能是距离过近导致除零)")
            rate_init = 0

        results_db['Initial (Unoptimized)'].append(rate_init)
        print(f"   [Init] Rate: {rate_init:.4f}")

        # ==========================================
        # B. Math 算法 (SCA/SCP) - [兼容性修复]
        # ==========================================
        try:
            # 1. 实例化 (只传核心参数，防止 TypeError)
            # 假设您的类定义至少需要 B, C, H, R, V, lambda, eta
            # 如果您的类定义不需要 V，请从下面删掉 V
            try:
                math_q = FastSCAOptimizerQ(B,  H, R, v_val, theta0=np.pi / 6, eta=ETA_VAL)
            except TypeError:
                # 备用：如果不接受 V，就不传 V
                math_q = FastSCAOptimizerQ(B,  H, R, v_val, theta0=np.pi / 6, eta=ETA_VAL)  # V用默认占位

            # 【关键】手动更新参数，确保算法知道当前的车速和角度
            math_q.V = v_val
            math_q.theta0 = THETA_GLOBAL

            # 同理处理 U 优化器
            try:
                math_u = FastSCPOptimizerU(Q_init, eta=ETA_VAL, theta0=np.pi / 6)
            except TypeError:
                # 如果参数列表不同，请根据您的 fasteru.py 修改这里
                math_u = FastSCPOptimizerU(Q_init, eta=ETA_VAL, theta0=np.pi / 6)

            # 手动更新参数
            math_u.V = v_val
            math_u.theta0 = THETA_GLOBAL

            # 2. 执行优化
            Q_math_raw = math_q.optimize(Q_init.copy(), U_init.copy())
            Q_math = np.array(Q_math_raw).reshape(B, 3)

            math_u.Q = Q_math
            U_math_raw = math_u.optimize(U_init.copy())
            U_math = np.array(U_math_raw).reshape(B, 3)

            # 3. 计算速率
            rate_math = current_engine.get_trajectory_average_rate(Q_math, U_math, eta=ETA_VAL)
            results_db['Math (SCA/SCP)'].append(rate_math)
            print(f"   [Math] Rate: {rate_math:.4f}")

        except Exception as e:
            print(f"   [Math] 报错: {e}")
            # 打开下面这行可以看到具体的报错行数
            # import traceback; traceback.print_exc()
            results_db['Math (SCA/SCP)'].append(0)

        # ==========================================
        # C. PSO 算法
        # ==========================================
        try:
            try:
                pso_opt = FullPSOOptimizer(eta=ETA_VAL, theta0=THETA_GLOBAL)
            except TypeError:
                pso_opt = FullPSOOptimizer(eta=ETA_VAL)

            # 【关键】手动更新参数
            pso_opt.theta0 = THETA_GLOBAL
            pso_opt.V = v_val
            pso_opt.T_total = 2 * R / v_val  # 重新计算总时间
            # 更新时间采样点 (重要！否则 PSO 还在按旧时间跑)
            if hasattr(pso_opt, 'time_points_opt'):
                pso_opt.time_points_opt = np.linspace(0, pso_opt.T_total, 10)  # 稀疏采样

            # 传入初始值加速收敛
            Q_pso, U_pso, _ = pso_opt.optimize(Q_init=Q_init, U_init=U_init)

            rate_pso = current_engine.get_trajectory_average_rate(Q_pso, U_pso, eta=ETA_VAL)
            results_db['PSO'].append(rate_pso)
            print(f"   [PSO ] Rate: {rate_pso:.4f}")
        except Exception as e:
            print(f"   [PSO ] 报错: {e}")
            results_db['PSO'].append(0)

        # ==========================================
        # D. GA 算法
        # ==========================================
        try:
            try:
                ga_opt = GeneticOptimizer(B, C, H, R, V=v_val, theta0=THETA_GLOBAL, eta=ETA_VAL)
            except TypeError:
                ga_opt = GeneticOptimizer(B, C, H, R, V=20.0, theta0=THETA_GLOBAL, eta=ETA_VAL)

            # 手动更新
            ga_opt.V = v_val
            ga_opt.theta0 = THETA_GLOBAL
            # 如果 GA 内部预计算了时间点，也需要更新
            if hasattr(ga_opt, 'T_total'):
                ga_opt.T_total = 2 * R / v_val
                ga_opt.time_points = np.linspace(0, ga_opt.T_total, 10)

            Q_ga, U_ga, _ = ga_opt.optimize()
            rate_ga = current_engine.get_trajectory_average_rate(Q_ga, U_ga, eta=ETA_VAL)
            results_db['Genetic (GA)'].append(rate_ga)
            print(f"   [GA  ] Rate: {rate_ga:.4f}")
        except Exception as e:
            print(f"   [GA  ] 报错: {e}")
            results_db['Genetic (GA)'].append(0)

    # ... (后面的绘图代码保持不变) ...

    # ==============================================================================
    # 3. 绘图
    # ==============================================================================
    print("\n>>> 绘制对比图...")
    plt.figure(figsize=(10, 6))

    styles = {
        'Initial (Unoptimized)': {'fmt': 'k:o', 'lw': 1.5},
        'Math (SCA/SCP)': {'fmt': 'b-s', 'lw': 2.5},
        'PSO': {'fmt': 'r--^', 'lw': 2},
        'Genetic (GA)': {'fmt': 'g-.d', 'lw': 2}
    }

    for name, rates in results_db.items():
        s = styles.get(name)
        plt.plot(speeds, rates, s['fmt'], linewidth=s['lw'], label=name, alpha=0.8)

    plt.xlabel('车速 V (m/s)', fontsize=12)
    plt.ylabel('综合加权速率 (Weighted Rate)', fontsize=12)
    plt.title(f'不同车速下的算法性能对比 ($\eta=0.9, \\theta_0=60^\circ$)', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10)
    plt.xticks(speeds)

    plt.tight_layout()
    plt.show()