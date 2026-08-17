import numpy as np
import matplotlib.pyplot as plt
import time
import warnings

# --- 导入您的自定义算法模块 ---
# 请确保这些文件在当前目录下
from qbub import SurfaceLayoutGenerator
from uq import SCPOptimizerU
from Qcurr import SCAOptimizerQ
from GA import GeneticOptimizer
from pos_1 import FullPSOOptimizer
from fasterq import FastSCAOptimizerQ
from fasteru import FastSCPOptimizerU

# --- 绘图配置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
plt.rcParams['figure.dpi'] = 100
warnings.filterwarnings("ignore")


# ==============================================================================
# 1. 物理引擎 (ChannelEngine) - 负责计算速率
# ==============================================================================
class ChannelEngine:
    def __init__(self, B=16, theta0=np.pi / 6):
        self.B = B
        self.H = 100.0;
        self.R = 200.0;
        self.V = 20.0
        self.THETA_0 = theta0
        self.T_total = 2 * self.R / self.V

        # 物理参数
        self.PC = 40.0 * 1e-3
        self.PS = 40.0 * 1e-3
        self.SIGMA_SQ = (10 ** (-90.0 / 10.0)) * 1e-3
        self.L_FRAME = 0.1  # 【关键】感知帧长设为 0.1s
        self.LAMBDA = 0.125
        self.N_ELEM = 4
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
        """计算 t 时刻的加权总速率"""
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

        # --- 模式选择 (Eta 策略) ---
        snr_c = pc_vals / self.SIGMA_SQ
        max_snr = np.max(snr_c) if np.max(snr_c) > 0 else 1e-12
        mv = np.where(snr_c > eta * max_snr, 1, 0)

        # 通信速率
        comm_sum = np.sum(pc_vals[mv == 1])
        rc = np.log2(1 + comm_sum / self.SIGMA_SQ) if comm_sum > 0 else 0

        # 感知速率 (空间分集累加模型)
        sens_snr = ps_vals[mv == 0] / self.SIGMA_SQ
        rs = (1.0 / self.L_FRAME) * np.sum(np.log2(1 + sens_snr)) if len(sens_snr) > 0 else 0

        # 简单加权总分 (可根据需要修改权重)
        rt = 0.5 * rc + 0.5 * rs
        return rt

    def get_trajectory_average_rate(self, Q, U, eta=0.5):
        """计算全轨迹平均速率"""
        # 使用物理帧长进行采样，确保精准
        t_samples = np.arange(0, self.T_total, self.L_FRAME)
        if len(t_samples) == 0: t_samples = np.array([0.0])

        total_rate = 0
        for t in t_samples:
            rt = self.compute_rates(Q, U, t, eta)
            total_rate += rt
        return total_rate / len(t_samples)


# ==============================================================================
# 2. 主执行逻辑 (Angle Sweep 0-90)
# ==============================================================================
if __name__ == "__main__":
    # --- 全局参数 ---
    B = 16
    C = 20.0
    H = 100.0;
    R = 200.0;
    V = 20.0
    ETA = 0.5  # 固定模式选择阈值

    # 【关键】定义 0 到 90 度，步长 15 度
    angles_deg = np.arange(0, 91, 15)
    # 结果: [0, 15, 30, 45, 60, 75, 90]

    # 存储结果字典
    results_db = {
        'Initial (Random)': [],
        'Math (SCA/SCP)': [],
        'PSO': [],
        'Genetic (GA)': []
    }

    print(f"=== 开始角度扫描分析 (Eta={ETA}) ===")
    print(f"扫描角度: {angles_deg}")

    total_start_time = time.time()

    for deg in angles_deg:
        rad = np.deg2rad(deg)
        print(f"\n>>> 正在处理初始角度 Theta0 = {deg}° ...")

        # 1. 针对当前角度，重新初始化环境
        # 必须这样做，因为 theta0 决定了车辆轨迹，也就是优化的目标函数
        current_engine = ChannelEngine(B=B, theta0=rad)
        current_gen = SurfaceLayoutGenerator(B=B, C=C, theta_0=rad, H=H, R=R, V=V)

        # ==========================================
        # A. 初始随机点位 (Baseline)
        # ==========================================
        # 每次都生成一个新的随机分布作为起点
        Q_init, U_init = current_gen.generate()
        rate_init = current_engine.get_trajectory_average_rate(Q_init, U_init, eta=ETA)
        results_db['Initial (Random)'].append(rate_init)

        # ==========================================
        # B. Math 算法 (FastSCA/SCP)
        # ==========================================
        try:
            # 尝试传入 theta0，如果您的类还没修改支持 theta0，这里会捕获异常并处理
            try:
                math_q = FastSCAOptimizerQ(B,  H, R, V, theta0=rad, eta=ETA)
            except TypeError:
                # 兼容旧代码：如果不接受 theta0，就不传
                math_q = FastSCAOptimizerQ(B,  H, R, V, theta0=rad, eta=ETA)

            try:
                math_u = FastSCPOptimizerU(Q_init, eta=ETA, theta0=rad)
            except TypeError:
                math_u = FastSCPOptimizerU(Q_init, eta=ETA)

            # 运行优化 (传入副本)
            Q_math = math_q.optimize(Q_init.copy(), U_init.copy()).reshape(B, 3)
            math_u.Q = Q_math
            U_math = math_u.optimize(U_init.copy()).reshape(B, 3)

            rate_math = current_engine.get_trajectory_average_rate(Q_math, U_math, eta=ETA)
            results_db['Math (SCA/SCP)'].append(rate_math)
            print(f"   [Math] Rate: {rate_math:.4f}")

        except Exception as e:
            print(f"   [Math] Error: {e}")
            results_db['Math (SCA/SCP)'].append(0)

        # ==========================================
        # C. PSO 算法
        # ==========================================
        try:
            # 同样尝试传入 theta0
            try:
                pso_opt = FullPSOOptimizer(eta=ETA, theta0=rad)
            except TypeError:
                pso_opt = FullPSOOptimizer(eta=ETA)
                pso_opt.theta0 = rad  # 强制赋值

            Q_pso, U_pso, _ = pso_opt.optimize()
            rate_pso = current_engine.get_trajectory_average_rate(Q_pso, U_pso, eta=ETA)
            results_db['PSO'].append(rate_pso)
            print(f"   [PSO ] Rate: {rate_pso:.4f}")
        except Exception as e:
            print(f"   [PSO ] Error: {e}")
            results_db['PSO'].append(0)

        # ==========================================
        # D. GA 算法
        # ==========================================
        try:
            try:
                ga_opt = GeneticOptimizer(B, C, H, R, V, theta0=rad, eta=ETA)
            except TypeError:
                # GA 参数较多，如果不匹配可能需要手动对齐，这里假设参数顺序一致
                ga_opt = GeneticOptimizer(B, C, H, R, V, rad, eta=ETA)

            Q_ga, U_ga, _ = ga_opt.optimize()
            rate_ga = current_engine.get_trajectory_average_rate(Q_ga, U_ga, eta=ETA)
            results_db['Genetic (GA)'].append(rate_ga)
            print(f"   [GA  ] Rate: {rate_ga:.4f}")
        except Exception as e:
            print(f"   [GA  ] Error: {e}")
            results_db['Genetic (GA)'].append(0)

    print(f"\n全部计算完成。总耗时: {time.time() - total_start_time:.2f}s")

    # ==============================================================================
    # 3. 绘图 (折线图)
    # ==============================================================================
    plt.figure(figsize=(10, 6))

    # 样式配置
    styles = {
        'Initial (Random)': {'color': 'black', 'marker': 'o', 'linestyle': ':', 'lw': 1.5},
        'Math (SCA/SCP)': {'color': 'blue', 'marker': 's', 'linestyle': '-', 'lw': 2.5},
        'PSO': {'color': 'red', 'marker': '^', 'linestyle': '--', 'lw': 2},
        'Genetic (GA)': {'color': 'green', 'marker': 'd', 'linestyle': '-.', 'lw': 2}
    }

    for name, rates in results_db.items():
        # 如果某个算法在某个角度出错(为0)，可以选择不画或者画出来
        s = styles.get(name)
        plt.plot(angles_deg, rates, label=name, **s)

        # 简单标注数值
        # for x, y in zip(angles_deg, rates):
        #     if y > 0: plt.text(x, y, f"{y:.1f}", fontsize=8, ha='center', va='bottom')

    plt.xlabel('车辆初始角度 $\\theta_0$ (Degree)', fontsize=12)
    plt.ylabel('轨迹平均综合速率 (Average Rate)', fontsize=12)
    plt.title(f'不同初始角度下的优化算法性能对比 (0-90°)', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10)
    plt.xticks(angles_deg)  # 强制显示所有刻度

    plt.tight_layout()
    plt.show()