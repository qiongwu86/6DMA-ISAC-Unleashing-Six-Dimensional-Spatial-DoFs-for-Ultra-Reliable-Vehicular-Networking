import numpy as np
import matplotlib.pyplot as plt
import warnings
import time

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

        # 物理参数
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

    def get_instantaneous_rates(self, Q, U, t, eta):
        """
        计算 t 时刻的瞬时速率 (不求平均)
        返回: (rc, rs, rt)
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
            theta_dev = np.arccos(np.clip(v_loc[2], -1, 1))
            phi_tilde = np.arctan2(v_loc[1], v_loc[0])
            val_p = 12 * (np.abs(phi_tilde) / self.PHI_3DB) ** 2;
            val_t = 12 * (np.abs(theta_dev) / self.THETA_3DB) ** 2
            gain_db = self.G_MAX - np.minimum(-(-np.minimum(val_p, self.G_V) - np.minimum(val_t, self.G_S)), self.G_S)
            gain = 10 ** (gain_db / 10.0)
            pc_vals[b] = self.PC * gain * (dist ** -self.EPSILON) * self.N_ELEM
            ps_vals[b] = self.PS * gain * (dist ** -self.EPSILON1) * (self.N_ELEM ** 2) * (self.RHO ** 2)

        # 模式选择
        snr_c = pc_vals / self.SIGMA_SQ
        max_snr = np.max(snr_c) if np.max(snr_c) > 0 else 1e-12
        mv = np.where(snr_c > eta * max_snr, 1, 0)

        # 1. 通信速率 (Log Sum)
        comm_sum = np.sum(pc_vals[mv == 1])
        rc = np.log2(1 + comm_sum / self.SIGMA_SQ) if comm_sum > 0 else 0

        # 2. 感知速率 (Sum Log) - 空间分集累加
        sens_snr = ps_vals[mv == 0] / self.SIGMA_SQ
        if len(sens_snr) > 0:
            rs = (1.0 / self.L_FRAME) * np.sum(np.log2(1 + sens_snr))
        else:
            rs = 0

        # 3. 综合速率
        rt = 0.5 * rc + 0.5 * rs

        return rc, rs, rt


# ==============================================================================
# 2. 主程序 (Monte Carlo Time Domain Analysis)
# ==============================================================================
if __name__ == "__main__":
    # --- 参数设置 ---
    B = 16
    C = 20.0
    H = 100.0;
    R = 200.0;
    V = 20.0
    THETA_GLOBAL = np.deg2rad(60)  # 初始角度 60度
    ETA_VAL = 0.9

    # --- 实验配置 ---
    NUM_MC = 10  # 蒙特卡洛实验次数
    TIME_STEPS = 100  # 时间轴采样点数

    print(f"=== 开始瞬时速率分析 ({NUM_MC} 次蒙特卡洛平均) ===")

    # 1. 初始化引擎和生成器
    engine = ChannelEngine(B=B, theta0=THETA_GLOBAL, V=V)
    gen = SurfaceLayoutGenerator(B=B, C=C, theta_0=THETA_GLOBAL, H=H, R=R, V=V)

    # 时间轴
    time_axis = np.linspace(0, engine.T_total, TIME_STEPS)

    # 累加器 (用于存储多次实验的结果)
    accum_rc = np.zeros(TIME_STEPS)
    accum_rs = np.zeros(TIME_STEPS)
    accum_rt = np.zeros(TIME_STEPS)

    total_start = time.time()

    # --- 蒙特卡洛循环 ---
    for i_mc in range(NUM_MC):
        print(f">>> 正在进行第 {i_mc + 1}/{NUM_MC} 次实验...")

        # A. 生成新的随机初始点 (关键：每次位置都要不一样)
        Q_init, U_init = gen.generate()

        # B. 运行数学优化 (SCA)
        try:
            # 实例化并手动传参
            try:
                math_q = FastSCAOptimizerQ(B, C, H, R, V, 0.125, ETA_VAL, THETA_GLOBAL)
            except:
                math_q = FastSCAOptimizerQ(B, C, H, R, 0.125, ETA_VAL)
            math_q.V = V;
            math_q.theta0 = THETA_GLOBAL

            try:
                math_u = FastSCPOptimizerU(Q_init, ETA_VAL, THETA_GLOBAL)
            except:
                math_u = FastSCPOptimizerU(Q_init, ETA_VAL)
            math_u.V = V;
            math_u.theta0 = THETA_GLOBAL

            # 执行优化
            Q_opt = np.array(math_q.optimize(Q_init.copy(), U_init.copy())).reshape(B, 3)
            math_u.Q = Q_opt
            U_opt = np.array(math_u.optimize(U_init.copy())).reshape(B, 3)

        except Exception as e:
            print(f"   优化失败，使用初始值: {e}")
            Q_opt, U_opt = Q_init, U_init

        # C. 时域扫描 (0 -> T_total)
        # 临时列表存储单次实验的轨迹
        trace_rc = []
        trace_rs = []
        trace_rt = []

        for t in time_axis:
            rc, rs, rt = engine.get_instantaneous_rates(Q_opt, U_opt, t, eta=ETA_VAL)
            trace_rc.append(rc)
            trace_rs.append(rs)
            trace_rt.append(rt)

        # D. 累加结果
        accum_rc += np.array(trace_rc)
        accum_rs += np.array(trace_rs)
        accum_rt += np.array(trace_rt)

    # --- 计算平均值 ---
    mean_rc = accum_rc / NUM_MC
    mean_rs = accum_rs / NUM_MC
    mean_rt = accum_rt / NUM_MC

    print(f"\n全部实验完成，总耗时: {time.time() - total_start:.2f}s")

    # ==============================================================================
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- 左轴：综合速率 & 通信速率 ---
    color_total = 'tab:red'
    color_comm = 'tab:blue'

    # 1. 绘制综合速率
    ln1 = ax1.plot(time_axis, mean_rt, color=color_total, linewidth=3,
                   label=f'Avg Weighted Total Rate')

    # 2. 绘制通信速率
    ln2 = ax1.plot(time_axis, mean_rc, color=color_comm, linestyle='--', linewidth=2,
                   label=f'Avg Comm Rate (Rc)')

    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('sum rate  (bits/s/Hz)', fontsize=12)
    ax1.tick_params(axis='y')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- 右轴：感知速率 ---
    ax2 = ax1.twinx()
    color_sens = 'tab:green'

    # 3. 绘制感知速率
    ln3 = ax2.plot(time_axis, mean_rs, color=color_sens, linestyle='-.', linewidth=2,
                   label=f'Avg Sensing Rate (Rs)')

    ax2.set_ylabel(' Sensing Rate (bits/s/Hz)', color=color_sens, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color_sens)

    # --- 合并图例 ---
    lines = ln1 + ln2 + ln3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', framealpha=0.9, shadow=True)

    # --- 【新增】添加平均值文本框 ---
    # 计算全过程的平均值 (Scalar)
    avg_val_rc = np.mean(mean_rc)
    avg_val_rs = np.mean(mean_rs)
    avg_val_rt = np.mean(mean_rt)

    textstr = '\n'.join((
        f'Overall Averages:',
        f'  Comm Rate: {avg_val_rc:.2f}',
        f'  Sens Rate: {avg_val_rs:.2f}',
        f'  Total Rate: {avg_val_rt:.2f}'
    ))



    plt.tight_layout()
    plt.show()

    # 打印最终平均统计值
    print(f"平均通信速率: {avg_val_rc:.4f}")
    print(f"平均感知速率: {avg_val_rs:.4f}")
    print(f"平均综合速率: {avg_val_rt:.4f}")