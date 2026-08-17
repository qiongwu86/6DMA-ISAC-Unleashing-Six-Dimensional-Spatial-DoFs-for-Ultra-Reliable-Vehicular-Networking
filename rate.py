import numpy as np
from matplotlib import pyplot as plt

from qbub import SurfaceLayoutGenerator
from Qcurr import SCAOptimizerQ
from  uq import  SCPOptimizerU
H = 100.0;
R = 200.0;
THETA_0 = np.pi / 6;
V = 20.0
B = 16  # 表面数量
C = 1.0  # 空间范围
LAMBDA = 0.125
N_ELEM = 4
D_NT = LAMBDA / 2
L_FRAME = 0.1
rho=0.8
# 物理参数 (Watt)
PC = 0.04;
PS = 0.04

SIGMA= -50
noise_mw = 10 ** (SIGMA / 10.0)
snr_linear = PS / noise_mw

EPSILON = 2;
EPSILON1 = 4;
RHO = 0.8

# 增益参数
THETA_3DB = np.deg2rad(65);
PHI_3DB = np.deg2rad(65)
G_MAX = 8;
G_S = 25;
G_V = 25
class SCARateCalculator:
    def __init__(self, B=16, theta0=np.pi / 6):
        """
        初始化计算器，设置物理常量
        """
        # --- 1. 系统几何与物理参数 ---
        self.B = B
        self.H = 100.0
        self.R = 200.0
        self.V = 20.0
        self.theta0 = theta0

        # --- 2. 功率与噪声 ---
        # 40mW -> 0.04W
        self.PC = 40.0 * 1e-3
        self.PS = self.PC
        # -90dBm -> 1e-12 W
        self.SIGMA_SQ = (10 ** (-90.0 / 10.0)) * 1e-3
        self.ln2 = np.log(2)

        # --- 3. 路径损耗与增益参数 ---
        self.EPSILON = 2  # 通信路损指数
        self.EPSILON1 = 4  # 感知路损指数
        self.RHO = 0.8  # 雷达截面积相关系数
        self.N_ELEM = 4  # 阵列单元数
        self.L_FRAME = 1.0  # 帧长

        # 天线增益参数
        self.THETA_3DB = np.deg2rad(65)
        self.PHI_3DB = np.deg2rad(65)
        self.G_MAX = 8;
        self.G_S = 25;
        self.G_V = 25

    # ==============================================================
    #  内部工具：几何与物理计算
    # ==============================================================
    def _get_rotation_matrix(self, u_b):
        """计算旋转矩阵"""
        a, b, g = u_b
        sa, ca = np.sin(a), np.cos(a)
        sb, cb = np.sin(b), np.cos(b)
        sg, cg = np.sin(g), np.cos(g)
        return np.array([[cb * cg, cb * sg, -sb],
                         [sb * sa * cg - ca * sg, sb * sa * sg + ca * cg, cb * sa],
                         [ca * sb * cg + sa * sg, ca * sb * sg - sa * cg, ca * cb]])

    def _calculate_channel_power(self, q, u, t):
        """
        计算每个表面的接收信号功率 P_rx (Communication & Sensing)
        """
        target_pos = np.array([
            self.R * np.cos(self.theta0),
            self.R * np.sin(self.theta0) - self.V * t,
            self.H
        ])

        pc_list = np.zeros(self.B)
        ps_list = np.zeros(self.B)

        # 将输入数组 reshape 为 (B, 3) 以防万一
        Q = q.reshape(self.B, 3)
        U = u.reshape(self.B, 3)

        for b in range(self.B):
            vec = target_pos - Q[b]
            dist = np.linalg.norm(vec)
            if dist < 1.0: dist = 1.0

            # 计算局部坐标系下的角度
            R_mat = self._get_rotation_matrix(U[b])
            v_loc = np.dot(R_mat.T, vec / dist)
            xb, yb, zb = v_loc

            theta_dev = np.arccos(np.clip(zb, -1, 1))
            phi_tilde = np.arctan2(yb, xb)

            # 计算增益 (Sinc 变体近似)
            val_p = 12 * (np.abs(phi_tilde) / self.PHI_3DB) ** 2
            val_t = 12 * (np.abs(theta_dev) / self.THETA_3DB) ** 2
            gain_db = self.G_MAX - np.minimum(-(-np.minimum(val_p, self.G_V) - np.minimum(val_t, self.G_S)), self.G_S)
            gain = 10 ** (gain_db / 10.0)

            # 路径损耗
            pl_c = dist ** (-self.EPSILON)
            pl_s = dist ** (-self.EPSILON1)

            # 计算能量 P
            pc_list[b] = self.PC * gain * pl_c * self.N_ELEM
            ps_list[b] = self.PS * gain * pl_s * (self.N_ELEM ** 2) * (self.RHO ** 2)

        return pc_list, ps_list

    def _get_sca_coeffs(self, p_val):
        """
        计算 SCA 系数 alpha 和 beta
        公式: log2(1 + P/sigma^2) >= alpha * P + beta
        """
        snr = p_val / self.SIGMA_SQ
        # alpha = 1 / (sigma^2 * ln2 * (1+snr))
        alpha = (1.0 / self.SIGMA_SQ) / (self.ln2 * (1 + snr))
        # beta = log2(1+snr) - alpha * P
        beta = np.log2(1 + snr) - alpha * p_val
        return alpha, beta

    # ==============================================================
    #  核心接口：计算速率
    # ==============================================================

    def compute_rate_trajectory(self, q, u, eta, num_samples=50, weight_c=0.5):
        # 确保 self.T_total 存在
        if not hasattr(self, 'T_total'):
            self.T_total = 2 * self.R / self.V

        t_axis = np.linspace(0, self.T_total, num_samples)

        rc_list = []
        rs_list = []
        rt_list = []

        for t in t_axis:
            # 1. 计算功率
            pc_arr, ps_arr = self._calculate_channel_power(q, u, t)

            # 2. 计算 SCA 系数
            ac, bc = self._get_sca_coeffs(pc_arr)
            as_, bs = self._get_sca_coeffs(ps_arr)

            # 3. 模式选择
            snr_c = pc_arr / self.SIGMA_SQ
            max_snr = np.max(snr_c) if np.max(snr_c) > 0 else 1e-12
            mv = np.where(snr_c > eta * max_snr, 1, 0)

            # 4. 速率计算
            # 通信
            rc_elements = ac * (PC/noise_mw)*(pc_arr)**2 + bc
            r_c_inst = np.sum(rc_elements * mv)

            # 感知
            rs_elements = (1.0 / self.L_FRAME) * (as_ * (PS / noise_mw) * (ps_arr) ** 2 + bs)
            r_s_inst = np.sum(rs_elements * (1 - mv))

            # 总速率
            r_total = weight_c * r_c_inst + (1 - weight_c) * r_s_inst

            rc_list.append(r_c_inst)
            rs_list.append(r_s_inst)
            rt_list.append(r_total)

        return t_axis, np.array(rc_list), np.array(rs_list), np.array(rt_list)

    # ==============================================================
    #  绘图工具
    # ==============================================================
    def plot_rates_over_time(self, q, u, eta):
        t, rc, rs, rt = self.compute_rate_trajectory(q, u, eta)

        plt.figure(figsize=(10, 6))
        plt.plot(t, rc, label='通信速率 (Communication)', linestyle='--', alpha=0.7)
        plt.plot(t, rs, label='感知速率 (Sensing)', linestyle='-.', alpha=0.7)
        plt.plot(t, rt, label='加权总速率 (Total)', linewidth=2.5, color='red')

        plt.title(f'SCA 凸近似速率随时间变化曲线 (T={self.T_total:.1f}s)')
        plt.xlabel('时间 t (s)')
        plt.ylabel('速率 Rate (bits/s/Hz)')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()
# ==============================================================
#  使用示例
# ==============================================================
if __name__ == "__main__":
    # 1. 实例化计算类\

    generator = SurfaceLayoutGenerator(B=B, C=20.0, theta_0=np.pi / 3, H=H, R=R, V=V)
    Q_int, U_int = generator.generate()

    # 3. 备份初始值 (关键步骤！防止被修改)
    Q_initial_backup = Q_int.copy()
    U_initial_backup = U_int.copy()

    # 4. 优化 Q (传入副本)
    generator1 = SCAOptimizerQ(u_fixed=U_initial_backup, eta=0.5, theta0=np.pi / 3)
    Q_fixed = generator1.optimize(Q_initial_backup.copy())

    # 5. 优化 U (传入副本)
    generator2 = SCPOptimizerU(q_fixed=Q_fixed, eta=0.5, theta0=np.pi / 3)
    U_fixed = generator2.optimize(U_initial_backup.copy())

    # 6. 计算速率曲线 (只计算数据，不弹窗)
    calculator = SCARateCalculator(B=16, theta0=np.pi / 3)  # 注意 theta0 要一致

    # 计算初始状态的速率
    t_axis, _, _, rt_init = calculator.compute_rate_trajectory(Q_initial_backup, U_initial_backup, eta=0.5)

    # 计算优化后状态的速率
    _, _, _, rt_opt = calculator.compute_rate_trajectory(Q_fixed, U_fixed, eta=0.5)

    # 7. 统一绘图
    plt.figure(figsize=(10, 6))
    plt.plot(t_axis, rt_init, 'b--o', label='Initial (Before Optimization)', alpha=0.7)
    plt.plot(t_axis, rt_opt, 'r-s', label='Optimized (After Q+U)', linewidth=2)

    plt.title('Comparison of Total Rate Before and After Optimization')
    plt.xlabel('Time (s)')
    plt.ylabel('Rate (bits/s/Hz)')
    plt.legend()
    plt.grid(True)
    plt.show()