import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds
import warnings
import time

# --- 绘图配置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
warnings.filterwarnings("ignore")


# ==============================================================================
# 0. 辅助类: SurfaceLayoutGenerator (确保代码独立运行)
# ==============================================================================
class SurfaceLayoutGenerator:
    def __init__(self, B, C, theta_0, H, R, V):
        self.B = B
        self.C = C
        self.theta_0 = theta_0
        self.H = H
        self.R = R
        self.V = V

    def generate(self):
        # 在 [-C, C] 范围内随机生成初始位置
        # Z 轴固定或者也在一定范围内，这里简化为 XY 平面随机，Z=0
        Q_init = np.random.uniform(-self.C, self.C, (self.B, 3))
        # 简单的姿态初始化
        U_init = np.random.uniform(0, 2 * np.pi, (self.B, 3))
        return Q_init, U_init


# ==============================================================================
# 1. 物理引擎 (ChannelEngine)
# ==============================================================================
class ChannelEngine:
    def __init__(self, B, N_elem, theta0=np.pi / 6, V=20.0):
        self.B = B
        self.N_ELEM = N_elem  # 阵元数量

        self.H = 100.0;
        self.R = 200.0;
        self.V = V;
        self.THETA_0 = theta0
        self.T_total = 2 * self.R / self.V
        self.PC = 40e-3;
        self.PS = 40e-3;
        self.SIGMA_SQ = 10 ** (-9) * 1e-3
        self.L_FRAME = 0.1
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

    def get_trajectory_avg(self, Q, U, eta=0.6):
        t_samples = np.arange(0, self.T_total, self.L_FRAME)
        if len(t_samples) == 0: return 0
        total = 0
        Q = Q.reshape(self.B, 3);
        U = U.reshape(self.B, 3)

        for t in t_samples:
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
                gain_db = self.G_MAX - np.minimum(-(-np.minimum(val_p, self.G_V) - np.minimum(val_t, self.G_S)),
                                                  self.G_S)
                gain = 10 ** (gain_db / 10.0)

                # 物理增益修正
                # 通信正比于 N
                pc_vals[b] = self.PC * gain * (dist ** -self.EPSILON) * self.N_ELEM
                # 感知正比于 N^2
                ps_vals[b] = self.PS * gain * (dist ** -self.EPSILON1) * (self.N_ELEM ** 2) * (self.RHO ** 2)

            snr_c = pc_vals / self.SIGMA_SQ
            mv = np.where(snr_c > eta * (np.max(snr_c) if np.max(snr_c) > 0 else 1e-12), 1, 0)

            comm_sum = np.sum(pc_vals[mv == 1])
            rc = np.log2(1 + comm_sum / self.SIGMA_SQ) if comm_sum > 0 else 0

            sens_snr = ps_vals[mv == 0] / self.SIGMA_SQ
            rs =  np.sum(np.log2(1 + sens_snr)) if len(sens_snr) > 0 else 0

            total += 0.5 * rc + 0.5 * rs

        return total / len(t_samples)


# ==============================================================================
# 2. 标准 SCA 优化器
# ==============================================================================
class StandardSCAOptimizer:
    def __init__(self, B, N_elem, engine, eta):
        self.B = B;
        self.N_ELEM = N_elem;
        self.engine = engine;
        self.eta = eta;
        self.ln2 = np.log(2)

    def _get_sca_weights(self, Q, U, t):
        target_pos = np.array([self.engine.R * np.cos(self.engine.THETA_0),
                               self.engine.R * np.sin(self.engine.THETA_0) - self.engine.V * t, self.engine.H])
        pc = np.zeros(self.B);
        ps = np.zeros(self.B)

        for b in range(self.B):
            vec = target_pos - Q[b]
            dist = max(np.linalg.norm(vec), 1.0)
            gain = 1.0
            pc[b] = self.engine.PC * gain * (dist ** -self.engine.EPSILON) * self.N_ELEM
            ps[b] = self.engine.PS * gain * (dist ** -self.engine.EPSILON1) * (self.N_ELEM ** 2) * (
                        self.engine.RHO ** 2)

        snr = pc / self.engine.SIGMA_SQ
        mv = np.where(snr > self.eta * (np.max(snr) + 1e-12), 1, 0)

        wc = np.zeros(self.B);
        ws = np.zeros(self.B)
        sum_pc = np.sum(pc[mv == 1])
        denom_c = self.ln2 * (self.engine.SIGMA_SQ + sum_pc)
        wc[mv == 1] = 1.0 / denom_c

        denom_s = self.engine.L_FRAME * self.ln2 * (self.engine.SIGMA_SQ + ps[mv == 0])
        ws[mv == 0] = 1.0 / denom_s
        return 0.5 * wc + 0.5 * ws, mv

    def _obj_grad(self, q_flat, U_fixed, t_list, params):
        Q = q_flat.reshape(self.B, 3)
        loss = 0.0;
        grad = np.zeros_like(Q)

        for idx, t in enumerate(t_list):
            w, mv = params[idx]
            target_pos = np.array([self.engine.R * np.cos(self.engine.THETA_0),
                                   self.engine.R * np.sin(self.engine.THETA_0) - self.engine.V * t, self.engine.H])

            vecs = target_pos - Q
            dists = np.linalg.norm(vecs, axis=1);
            dists = np.maximum(dists, 1.0)

            g = 1.0
            pc = self.engine.PC * g * (dists ** -self.engine.EPSILON) * self.N_ELEM
            ps = self.engine.PS * g * (dists ** -self.engine.EPSILON1) * (self.N_ELEM ** 2) * (self.engine.RHO ** 2)

            p_curr = np.where(mv == 1, pc, ps)
            loss -= np.sum(w * p_curr)

            eps = np.where(mv == 1, self.engine.EPSILON, self.engine.EPSILON1)
            g_factor = w[:, np.newaxis] * p_curr[:, np.newaxis] * eps[:, np.newaxis] / (dists[:, np.newaxis] ** 2)
            grad -= g_factor * vecs

        return loss / len(t_list), grad.flatten() / len(t_list)


# ==============================================================================
# 3. 仿真主控函数
# ==============================================================================
def run_simulation(B_count, N_count, max_iters, eta=0.9):
    engine = ChannelEngine(B=B_count, N_elem=N_count)
    optimizer = StandardSCAOptimizer(B=B_count, N_elem=N_count, engine=engine, eta=eta)

    # 统一初始化
    gen = SurfaceLayoutGenerator(B=B_count, C=20.0, theta_0=np.pi / 6, H=100.0, R=200.0, V=20.0)
    Q_init, U_init = gen.generate()

    Q_curr = Q_init.copy()
    U_curr = U_init.copy()

    hist = []
    curr_rate = engine.get_trajectory_avg(Q_curr, U_curr, eta)
    hist.append(curr_rate)

    t_list = np.linspace(0, engine.T_total, 5)

    for k in range(max_iters):
        sca_params = []
        for t in t_list:
            w, mv = optimizer._get_sca_weights(Q_curr, U_curr, t)
            sca_params.append((w, mv))

        try:
            res = minimize(
                fun=optimizer._obj_grad,
                x0=Q_curr.flatten(),
                args=(U_curr, t_list, sca_params),
                method='L-BFGS-B', jac=True,
                bounds=Bounds([-200] * B_count * 3, [200] * B_count * 3),
                # 适度放宽限制，保证能爬坡到高点
                options={'maxiter': 5, 'ftol': 1e-4}
            )
            Q_temp = res.x.reshape(B_count, 3)
        except:
            Q_temp = Q_curr

        # 单调更新
        new_rate = engine.get_trajectory_avg(Q_temp, U_curr, eta)
        if new_rate > curr_rate:
            curr_rate = new_rate
            Q_curr = Q_temp

        hist.append(curr_rate)

    return hist


# ==============================================================================
# 4. 主程序
# ==============================================================================
if __name__ == "__main__":
    MAX_ITERS = 25
    NUM_MC = 10  # 10次实验

    print(f"=== 对比实验: 总阵元 64 (10次平均) ===")

    # --- 方案 A: 分布式 16x4 ---
    print(">>> 计算方案 A (分布式 16x4)...")
    hist_A_matrix = []
    for i in range(NUM_MC):
        # 简单打印进度
        if i == 0 or (i + 1) % 5 == 0:
            print(f"  Run {i + 1}/{NUM_MC} ...")
        h = run_simulation(16, 4, MAX_ITERS)
        hist_A_matrix.append(h)
    hist_A = np.mean(hist_A_matrix, axis=0)
    print(f"方案 A 最终平均速率: {hist_A[-1]:.2f}")

    # --- 方案 B: 集中式 4x16 ---
    print(">>> 计算方案 B (集中式 4x16)...")
    hist_B_matrix = []
    for i in range(NUM_MC):
        if i == 0 or (i + 1) % 5 == 0:
            print(f"  Run {i + 1}/{NUM_MC} ...")
        h = run_simulation(4, 16, MAX_ITERS)
        hist_B_matrix.append(h)
    hist_B = np.mean(hist_B_matrix, axis=0)
    print(f"方案 B 最终平均速率: {hist_B[-1]:.2f}")

    # --- 绘图 ---
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(MAX_ITERS + 1)

    ax.plot(x, hist_A, 'r--o', linewidth=2.5, markersize=5, label='Distributed: B=16, N=4')
    ax.plot(x, hist_B, 'b--s', linewidth=2.5, markersize=5, label='Centralized: B=4, N=16')

    ax.set_xlabel('Number of Iterations', fontsize=12)
    ax.set_ylabel('Achievable rate(bps/Hz)', fontsize=12)
    ax.set_xlim(0, MAX_ITERS)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=12, loc='lower right', framealpha=0.9, shadow=True)

    # 动态调整 Y 轴
    y_min = min(np.min(hist_A), np.min(hist_B))
    y_max = max(np.max(hist_A), np.max(hist_B))
    ax.set_ylim(y_min * 0.8, y_max * 1.1)

    plt.tight_layout()
    plt.show()