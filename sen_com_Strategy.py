import numpy as np
import matplotlib.pyplot as plt
import time
import warnings

# --- 导入模块 ---
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

    def compute_rates(self, Q, U, t, strategy_mode='proposed'):
        """
        :param strategy_mode:
            'proposed': Eta=0.9 (混合)
            'all_comm': 强制全通信
            'all_sens': 强制全感知
        """
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

        # --- 策略控制逻辑 ---
        if strategy_mode == 'all_comm':
            # 强制 mv 全为 1
            mv = np.ones(self.B, dtype=int)
        elif strategy_mode == 'all_sens':
            # 强制 mv 全为 0
            mv = np.zeros(self.B, dtype=int)
        else:  # 'proposed'
            # 正常 Eta = 0.9 筛选
            snr_c = pc_vals / self.SIGMA_SQ
            max_snr = np.max(snr_c) if np.max(snr_c) > 0 else 1e-12
            mv = np.where(snr_c > 0.9 * max_snr, 1, 0)

        # 1. 通信 (Log Sum)
        comm_sum = np.sum(pc_vals[mv == 1])
        rc = np.log2(1 + comm_sum / self.SIGMA_SQ) if comm_sum > 0 else 0

        # 2. 感知 (Sum Log)
        sens_snr = ps_vals[mv == 0] / self.SIGMA_SQ
        if len(sens_snr) > 0:
            rs = (1.0 / self.L_FRAME) * np.sum(np.log2(1 + sens_snr))
        else:
            rs = 0

        # 简单平均权重
        return 0.5 * rc + 0.5 * rs

    def get_trajectory_avg(self, Q, U, strategy_mode='proposed'):
        t_samples = np.arange(0, self.T_total, self.L_FRAME)
        if len(t_samples) == 0: t_samples = np.array([0.0])
        total = 0
        for t in t_samples: total += self.compute_rates(Q, U, t, strategy_mode)
        return total / len(t_samples)


# ==============================================================================
# 2. 主程序 (Strategies Comparison)
# ==============================================================================
if __name__ == "__main__":
    # 参数
    B = 16;
    C = 20.0;
    H = 100.0;
    R = 200.0
    THETA_GLOBAL = np.deg2rad(60)  # 固定初始角度
    ETA_VAL = 0.9

    # 扫描车速
    speeds = np.arange(20, 101, 10)

    # 存储结果
    results = {
        'Proposed (Dynamic)': [],
        'Full Communication': [],
        'Full Sensing': []
    }

    print(f"=== 开始策略对比实验 (Proposed vs Full Comm vs Full Sens) ===")

    for v_val in speeds:
        print(f"Processing V = {v_val} m/s ...")

        # 1. 初始化
        current_engine = ChannelEngine(B=B, theta0=THETA_GLOBAL, V=v_val)
        current_gen = SurfaceLayoutGenerator(B=B, C=C, theta_0=THETA_GLOBAL, H=H, R=R, V=v_val)
        Q_init, U_init = current_gen.generate()

        # 2. 运行数学优化算法 (寻找最优布局)
        # 我们使用 Proposed 策略 (Eta=0.9) 来寻找最优的物理位置
        # 因为我们的目的是证明：即使在这个最优位置下，如果强制全通信或全感知，效果也不好
        try:
            try:
                math_q = FastSCAOptimizerQ(B, C, H, R, v_val, 0.125, ETA_VAL)
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

            # 优化得到最优布局 Q*, U*
            Q_opt = np.array(math_q.optimize(Q_init.copy(), U_init.copy())).reshape(B, 3)
            math_u.Q = Q_opt
            U_opt = np.array(math_u.optimize(U_init.copy())).reshape(B, 3)

        except Exception as e:
            print(f"Optimization failed: {e}, using random init.")
            Q_opt, U_opt = Q_init, U_init

        # 3. 在同一套 Q*, U* 下，测试三种分配策略的性能

        # A. Proposed (Dynamic Eta=0.9)
        r_prop = current_engine.get_trajectory_avg(Q_opt, U_opt, strategy_mode='proposed')
        results['Proposed (Dynamic)'].append(r_prop)

        # B. Full Communication (强制所有做通信)
        r_comm = current_engine.get_trajectory_avg(Q_opt, U_opt, strategy_mode='all_comm')
        results['Full Communication'].append(r_comm)

        # C. Full Sensing (强制所有做感知)
        r_sens = current_engine.get_trajectory_avg(Q_opt, U_opt, strategy_mode='all_sens')
        results['Full Sensing'].append(r_sens)

    # ==============================================================================
    # 3. 绘图
    # ==============================================================================
    plt.figure(figsize=(10, 6))

    # 样式
    plt.plot(speeds, results['Proposed (Dynamic)'], 'r-o', linewidth=3, markersize=8,
             label='Proposed Allocation (Dynamic)')
    plt.plot(speeds, results['Full Communication'], 'b--s', linewidth=2, label='Baseline 1: All Communication')
    plt.plot(speeds, results['Full Sensing'], 'g-.^', linewidth=2, label='Baseline 2: All Sensing')

    # 填充颜色展示优势区域
    plt.fill_between(speeds, results['Proposed (Dynamic)'], results['Full Sensing'], color='red', alpha=0.1,
                     label='Performance Gain')

    plt.xlabel('车辆移动速度 (m/s)', fontsize=12)
    plt.ylabel('综合加权速率 (Weighted Sum Rate)', fontsize=12)
    plt.title('不同通感资源分配策略的性能对比', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10)
    plt.xticks(speeds)

    # 标注平均值
    avg_prop = np.mean(results['Proposed (Dynamic)'])
    avg_comm = np.mean(results['Full Communication'])
    plt.text(60, avg_prop + 5, f"Proposed Avg: {avg_prop:.1f}", color='red', fontweight='bold')
    plt.text(60, avg_comm + 5, f"All Comm Avg: {avg_comm:.1f}", color='blue')

    plt.tight_layout()
    plt.show()