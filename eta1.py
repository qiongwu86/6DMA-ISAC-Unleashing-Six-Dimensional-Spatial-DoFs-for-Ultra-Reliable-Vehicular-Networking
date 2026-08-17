import numpy as np
import matplotlib.pyplot as plt
from qbub import SurfaceLayoutGenerator

# --- 绘图设置 (防止中文乱码) ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from uq import SCPOptimizerU
from Qcurr import SCAOptimizerQ


# ==============================================================================
# 1. 布局生成器 (SurfaceLayoutGenerator)
#    负责生成 Q 和 U
# ==============================================================================
class SurfaceLayoutGenerator:
    def __init__(self, B, C, theta_0, H, R, V, d_min=None):
        self.B = B
        self.C = C
        self.theta_0 = theta_0
        self.H = H;
        self.R = R;
        self.V = V
        # 如果未指定 d_min，则根据波长计算 (lambda=0.125)
        if d_min is None:
            lam = 0.125
            self.d_min = (np.sqrt(2) * lam + lam) / 2
        else:
            self.d_min = d_min

    def generate(self):
        """生成随机但合规的初始 Q 和 U"""
        print(f"--- SurfaceLayoutGenerator: 生成 {self.B} 个初始点 ---")

        # 1. 随机生成 Q (带防碰撞检测)
        Q_init = np.zeros((self.B, 3))
        for i in range(self.B):
            valid = False
            while not valid:
                px = np.random.uniform(-self.C, self.C)
                pz = np.random.uniform(-self.C, self.C)
                cand = np.array([px, 0.0, pz])
                collision = False
                for j in range(i):
                    if np.linalg.norm(cand - Q_init[j]) < self.d_min:
                        collision = True;
                        break
                if not collision:
                    Q_init[i] = cand;
                    valid = True

        # 2. 随机生成 U (0~2pi)
        U_init = np.random.uniform(0, 2 * np.pi, (self.B, 3))
        return Q_init, U_init


# ==============================================================================
# 2. 物理信道引擎 (ChannelEngine)
#    负责计算精确的 h_c 和 h_s (物理层核心)
# ==============================================================================
class ChannelEngine:
    def __init__(self, B, theta0, H, R, V):
        self.B = B;
        self.THETA_0 = theta0
        self.H = H;
        self.R = R;
        self.V = V

        # 物理常量
        self.PC = 40.0 * 1e-3  # 40mW
        self.PS = self.PC
        self.SIGMA_SQ = (10 ** (-90.0 / 10.0)) * 1e-3
        self.LAMBDA = 0.125
        self.N_ELEM = 4
        self.EPSILON = 2  # 通信路损指数
        self.EPSILON1 = 4  # 感知路损指数 (通常是通信的2倍)
        self.RHO = 0.8  # RCS
        self.L_FRAME = 1.0

        # 增益参数
        self.THETA_3DB = np.deg2rad(65);
        self.PHI_3DB = np.deg2rad(65)
        self.G_MAX = 8;
        self.G_S = 25;
        self.G_V = 25

    def _get_rotation_matrix(self, u_b):
        a, b, g = u_b
        sa, ca = np.sin(a), np.cos(a)
        sb, cb = np.sin(b), np.cos(b)
        sg, cg = np.sin(g), np.cos(g)
        return np.array([[cb * cg, cb * sg, -sb],
                         [sb * sa * cg - ca * sg, sb * sa * sg + ca * cg, cb * sa],
                         [ca * sb * cg + sa * sg, ca * sb * sg - sa * cg, ca * cb]])

    def calculate_channel_powers(self, Q, U, t):
        """
        计算 t 时刻所有表面的接收功率 P_c 和 P_s
        返回: (pc_vals, ps_vals)
        """
        Q = Q.reshape(self.B, 3);
        U = U.reshape(self.B, 3)
        target_pos = np.array([self.R * np.cos(self.THETA_0), self.R * np.sin(self.THETA_0) - self.V * t, self.H])

        pc_vals = np.zeros(self.B, dtype=float)
        ps_vals = np.zeros(self.B, dtype=float)

        for b in range(self.B):
            vec = target_pos - Q[b]
            d_t = np.linalg.norm(vec)
            if d_t < 1.0: d_t = 1.0

            # 局部坐标与增益
            R_mat = self._get_rotation_matrix(U[b])
            v_loc = np.dot(R_mat.T, vec / d_t)
            # 防止数值误差导致 arccos 越界
            theta_dev = np.arccos(np.clip(v_loc[2], -1, 1))
            phi_tilde = np.arctan2(v_loc[1], v_loc[0])

            val_p = 12 * (np.abs(phi_tilde) / self.PHI_3DB) ** 2
            val_t = 12 * (np.abs(theta_dev) / self.THETA_3DB) ** 2
            gain_db = self.G_MAX - np.minimum(-(-np.minimum(val_p, self.G_V) - np.minimum(val_t, self.G_S)), self.G_S)
            gain = 10 ** (gain_db / 10.0)

            # --- 通信功率 P_c (单程) ---
            pl_c = d_t ** (-self.EPSILON)
            # 假设 N 个单元波束赋形增益为 N (功率增益)
            pc_vals[b] = self.PC * gain * pl_c * self.N_ELEM

            # --- 感知功率 P_s (双程) ---
            pl_s = d_t ** (-self.EPSILON1)
            # 假设双程增益，且考虑 RCS (rho)
            ps_vals[b] = self.PS * gain * pl_s * (self.N_ELEM ** 2) * (self.RHO ** 2)
        return pc_vals, ps_vals


# ==============================================================================
# 3. Eta 分析器 (EtaAnalyzer)
#    负责扫描 Eta，归一化，绘图
# ==============================================================================
class EtaAnalyzer:
    def __init__(self, engine):
        self.engine = engine
        self.T_total = 2 * engine.R / engine.V
        self.SIGMA_SQ = engine.SIGMA_SQ
        self.L_FRAME = engine.L_FRAME

    def run_scan(self, Q, U, num_eta=50):
        # =========================================================
        # 1. 物理采样修正：以帧长为单位生成时间轴
        # =========================================================
        # 假设 L_FRAME 在 engine 中已定义 (例如 0.1秒)
        # 使用 arange 生成 [0, 0.1, 0.2, ... T_total]
        t_samples = np.arange(0, self.T_total, self.engine.L_FRAME)

        # 防止 L_FRAME 设置过大导致采样点为空
        if len(t_samples) == 0:
            t_samples = np.array([0.0])

        print(
            f"1. 正在计算物理信道... (总时长={self.T_total:.1f}s, 帧长={self.engine.L_FRAME}s, 总帧数={len(t_samples)})")

        power_cache = []
        for t in t_samples:
            pc, ps = self.engine.calculate_channel_powers(Q, U, t)
            power_cache.append((pc, ps))

        # 2. 计算归一化权重 (Endpoint Normalization)
        print("2. 正在计算归一化权重...")
        rc_max_accum = 0
        for (pc, _) in power_cache:
            rc_max_accum += np.log2(1 + np.sum(pc) / self.SIGMA_SQ)
        rc_max = rc_max_accum / len(t_samples)

        rs_max_accum = 0
        for (_, ps) in power_cache:
            rs_max_accum += np.sum(np.log2(1 + ps / self.SIGMA_SQ))
        rs_max = rs_max_accum / len(t_samples)

        if rc_max < 1e-6: rc_max = 1.0
        if rs_max < 1e-6: rs_max = 1.0

        w_c = 1.0 / rc_max
        w_s = 1.0 / rs_max
        print(f"   Rc_max={rc_max:.4f}, Rs_max={rs_max:.4f}")

        # 3. 扫描 Eta
        print(f"3. 开始扫描 {num_eta} 个 Eta 值...")
        eta_list = np.linspace(0, 1.0, num_eta)
        rc_list = []
        rs_list = []
        rt_list = []
        comm_count_list = []

        for eta in eta_list:
            traj_rc = 0;
            traj_rs = 0
            traj_comm_count = 0

            for (pc, ps) in power_cache:
                # ------------------------------------------------
                # 模式选择逻辑 (这里保持您之前的 Max * Eta 逻辑)
                # ------------------------------------------------
                snr = pc / self.SIGMA_SQ
                max_snr = np.max(snr) if np.max(snr) > 0 else 1e-12

                mv = np.where(snr > eta * max_snr, 1, 0)

                # 统计当前帧参与通信的表面数 (整数)
                traj_comm_count += np.sum(mv)

                # 计算瞬时速率
                p_comm_sum = np.sum(pc[mv == 1])
                r_c = np.log2(1 + p_comm_sum / self.SIGMA_SQ) if p_comm_sum > 0 else 0

                p_sens = ps[mv == 0]
                if len(p_sens) > 0:
                    r_s = (1.0 / self.L_FRAME) * np.sum(np.log2(1 + p_sens / self.SIGMA_SQ))
                else:
                    r_s = 0

                traj_rc += r_c
                traj_rs += r_s

            # 4. 计算全轨迹平均值 (Time-Averaged Rate)
            # 除以的是【总帧数】
            avg_rc = traj_rc / len(t_samples)
            avg_rs = traj_rs / len(t_samples)

            # 平均通信数量 = 总计活跃次数 / 总帧数
            # 物理意义：平均每帧有多少个表面在工作
            avg_comm_count = traj_comm_count / len(t_samples)

            avg_rt = w_c * avg_rc + w_s * avg_rs

            rc_list.append(avg_rc)
            rs_list.append(avg_rs)
            rt_list.append(avg_rt)
            comm_count_list.append(avg_comm_count)

        return eta_list, rc_list, rs_list, rt_list, (w_c, w_s), comm_count_list

    def plot(self, eta_x, rc, rs, rt, weights, comm_counts):  # 接收 comm_counts
        w_c, w_s = weights
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # 左轴：加权总速率 (Trade-off 目标)
        color = 'tab:red'
        ax1.set_xlabel('模式选择阈值因子 $\eta$ (Eta)', fontsize=12)
        ax1.set_ylabel('原始物理速率 (bits/s/Hz)', color='tab:blue', fontsize=12)
        l1, = ax1.plot(eta_x, rt, color=color, linewidth=3, label='加权总速率 (Objective)')

        ax1.grid(True, linestyle=':', alpha=0.6)



        # 右轴：原始物理速率
        ax2 = ax1.twinx()


        l2, = ax2.plot(eta_x, rc, 'b--', label=f'通信速率 (原始)', alpha=0.6, linewidth=2)
        l3, = ax2.plot(eta_x, rs, 'g-.', label=f'感知速率 (原始)', alpha=0.6, linewidth=2)
        ax1.tick_params(axis='y', labelcolor='tab:blue')



        # 图例 (合并 ax1 和 ax2 的图例)
        lines = [l1,l2,l3]
        ax1.legend(lines, [l.get_label() for l in lines], loc='center right')

        plt.tight_layout()
        plt.show()


# ==============================================================================
# 主程序入口
# ==============================================================================
if __name__ == "__main__":
    # 参数设置
    B = 16
    C = 20.0
    H = 100;
    R = 200;
    V = 20
    THETA_GLOBAL = np.pi / 3
    t_val = 1.5
    # 1. 生成布局 (Q, U)
    # 使用 generator 生成随机合规的初始点
    generator = SurfaceLayoutGenerator(B=B, C=C, theta_0=THETA_GLOBAL, H=H, R=R, V=V)
    Q_int, U_int = generator.generate()

    generator1 = SCAOptimizerQ(B=B, C=C, H=H, R=R, V=V, lambda_val=0.125, eta=0.5, theta0=np.pi / 6)
    Q_fixed = generator1.optimize(q_init=Q_int, u_fixed=U_int, t_input=t_val)

    generator2 = SCPOptimizerU(q_fixed=Q_fixed, eta=0.5, theta0=np.pi / 3)
    U_fixed = generator2.optimize(U_int)
    # 2. 准备物理引擎 (用于计算真实速率)
    # 必须有这个，因为 generator 只负责产生坐标，不负责算信道
    phys_engine = ChannelEngine(B=B, theta0=THETA_GLOBAL, H=H, R=R, V=V)

    # 3. 准备分析器
    analyzer = EtaAnalyzer(phys_engine)

    # 4. 执行 Eta 扫描
    print("\n>>> 开始 Eta 扫描分析...")
    # 这里直接传入生成的 Q_opt, U_opt
    # run_scan 现在返回 comm_count_list
    eta_vals, rc_vals, rs_vals, rt_vals, ws, comm_counts = analyzer.run_scan(Q_int, U_int, num_eta=100)

    # 5. 绘图
    # 传入 comm_counts 用于绘制顶部 X 轴
    analyzer.plot(eta_vals, rc_vals, rs_vals, rt_vals, ws, comm_counts)