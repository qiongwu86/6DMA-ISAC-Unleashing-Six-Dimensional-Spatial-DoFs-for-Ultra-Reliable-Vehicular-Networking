import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from itertools import combinations
import warnings

warnings.filterwarnings("ignore")

# ====================================================================
# 1. 全局常量 (保持一致)
# ====================================================================
H = 100.0;
R = 200.0;
THETA_0 = np.pi / 6;
V = 20.0
B = 16
C = 1.0
LAMBDA = 0.125
N_ELEM = 4;
D_NT = LAMBDA / 2
L_FRAME = 1.0

PC = 0.04;
PS = 0.04
SIGMA= -50
EPSILON = 2;
EPSILON1 = 4;
RHO = 0.8

THETA_3DB = np.deg2rad(65);
PHI_3DB = np.deg2rad(65)
G_MAX = 8;
G_S = 25;
G_V = 25



def dbm_to_watt(dbm_value):
    """
    将 dBm 转换为 Watt
    公式: W = 10^(dBm / 10) / 1000
    """
    # 1. 先算出毫瓦 (mW)
    mw = 10 ** (dbm_value / 10.0)

    # 2. 将毫瓦转换为瓦特 (W)
    watt = mw / 1000.0

    return watt

SIGMA_SQ=dbm_to_watt(SIGMA)





class PenaltyMethodOptimizerU:
    def __init__(self, q_fixed, eta, theta0):
        self.Q = q_fixed
        self.eta = eta
        self.theta0 = theta0
        self.ln2 = np.log(2)

        # 优化参数
        self.sca_loops = 10  # SCA 外层迭代
        self.penalty_factor = 1000.0  # 惩罚系数 mu (初始值)
        self.penalty_growth = 2.0  # 惩罚系数增长倍率
        self.time_points = np.linspace(0, 2 * R / V, 5)

    # ------------------------------------------------------
    # 基础几何工具
    # ------------------------------------------------------
    def _get_rotation_matrix(self, u_b):
        a, b, g = u_b
        sa, ca = np.sin(a), np.cos(a)
        sb, cb = np.sin(b), np.cos(b)
        sg, cg = np.sin(g), np.cos(g)
        return np.array([[cb * cg, cb * sg, -sb],
                         [sb * sa * cg - ca * sg, sb * sa * sg + ca * cg, cb * sa],
                         [ca * sb * cg + sa * sg, ca * sb * sg - sa * cg, ca * cb]])

    def _get_normal_vector(self, u_b):
        return self._get_rotation_matrix(u_b)[:, 2]

    # ------------------------------------------------------
    # 方法 B 核心：线性化法向量计算
    # ------------------------------------------------------
    def _calc_linearized_normal(self, delta_u_b, n_ref):
        """
        基于方法 B (image_b168f8.png) 计算线性化后的法向量
        n_new = R(delta_u) * n_ref

        R(delta_u) 近似为:
        [[ 1,       dg,     -db ],
         [ -dg,     1,       da ],
         [ db,      -da,     1  ]]
        """
        da, db, dg = delta_u_b

        # 构造小角度旋转矩阵
        R_delta = np.array([
            [1.0, dg, -db],
            [-dg, 1.0, da],
            [db, -da, 1.0]
        ])

        # 线性化后的新法向量
        return np.dot(R_delta, n_ref)

    # ------------------------------------------------------
    # 物理引擎
    # ------------------------------------------------------
    def _calculate_channel_energy(self, U, t):
        target_pos = np.array([R * np.cos(self.theta0), R * np.sin(self.theta0) - V * t, H])
        hc_e, hs_e = np.zeros(B), np.zeros(B)

        for b in range(B):
            vec = target_pos - self.Q[b]
            dist = np.linalg.norm(vec)
            if dist < 1.0: dist = 1.0

            R_mat = self._get_rotation_matrix(U[b])
            v_loc = np.dot(R_mat.T, vec / dist)
            xb, yb, zb = v_loc

            theta_dev = np.arccos(np.clip(zb, -1, 1))
            phi_tilde = np.arctan2(yb, xb)

            val_p = 12 * (np.abs(phi_tilde) / PHI_3DB) ** 2
            val_t = 12 * (np.abs(theta_dev) / THETA_3DB) ** 2
            gain_db = G_MAX - np.minimum(-(-np.minimum(val_p, G_V) - np.minimum(val_t, G_S)), G_S)
            gain = 10 ** (gain_db / 10.0)

            pl_c = dist ** (-EPSILON);
            pl_s = dist ** (-EPSILON1)
            hc_e[b] = PC * gain * pl_c * N_ELEM
            hs_e[b] = PS * gain * pl_s * (N_ELEM ** 2) * (RHO ** 2)

        return hc_e, hs_e

    # ------------------------------------------------------
    # SCA 与 惩罚函数构建
    # ------------------------------------------------------
    def _get_sca_coefficients(self, U_ref, t):
        pc, ps = self._calculate_channel_energy(U_ref, t)

        # 通信系数
        snr_c = PC / SIGMA_SQ
        alpha_c = (1.0 / ((1 + snr_c * pc ** 2) * (self.ln2))) * snr_c
        beta_c = np.log2(1 + snr_c * pc ** 2) - alpha_c * (pc ** 2)

        # 感知系数
        snr_s = PS / SIGMA_SQ
        alpha_s = (1.0 / ((1 + snr_s * ps ** 2) * (self.ln2))) * snr_s
        beta_s = np.log2(1 + snr_s * ps ** 2) - alpha_s * (ps ** 2)

        return alpha_c, beta_c, alpha_s, beta_s

    def _loss_function(self, delta_u_flat, U_ref, sca_params_list, n_refs, mu, weight_a=0.5):
        """
        总目标函数 = -凸近似速率 + 惩罚项
        Optimization Variable: delta_u (增量)
        """
        delta_U = delta_u_flat.reshape(B, 3)
        U_curr = U_ref + delta_U  # 更新后的真实角度

        # 1. 计算负的凸近似速率 (Minimization target)
        rate_obj = 0.0
        for idx, t in enumerate(self.time_points):
            ac, bc, as_, bs = sca_params_list[idx]
            pc, ps = self._calculate_channel_energy(U_curr, t)

            snr = pc / SIGMA_SQ
            # Mode selection based on current U to be accurate,
            # or fix it based on U_ref for stricter SCA. Here we use current.
            mv = np.where(snr > self.eta * np.max(snr), 1, 0)

            rc_hat = ac * pc + bc
            rs_hat = (1.0 / L_FRAME) * (as_ * ps + bs)

            val = np.sum(np.where(mv == 1, weight_a * rc_hat, (1 - weight_a) * rs_hat))
            rate_obj += val

        avg_rate = rate_obj / len(self.time_points)

        # 2. 计算线性化约束的惩罚项 (Method B)
        penalty = 0.0

        for b in range(B):
            # 获取线性化法向量 n_lin = R(delta) * n_ref
            n_lin = self._calc_linearized_normal(delta_U[b], n_refs[b])

            # A. 朝向约束: n^T q >= 0  =>  -n^T q <= 0
            # 违反量 = max(0, -n^T q)
            g2 = -np.dot(n_lin, self.Q[b])
            penalty += np.maximum(0, g2) ** 2

            # B. 互不遮挡
            for j in range(B):
                if b == j: continue
                # n_b^T (q_j - q_b) <= 0
                # 违反量 = max(0, n_b^T (q_j - q_b))
                diff = self.Q[j] - self.Q[b]
                g1 = np.dot(n_lin, diff)
                penalty += np.maximum(0, g1) ** 2

        # Total Loss (Unconstrained)
        return -avg_rate + mu * penalty

    # ------------------------------------------------------
    # 主优化流程
    # ------------------------------------------------------
    def optimize(self, u_init):
        print(f"--- 启动惩罚函数法优化 U (Method B Linearization) ---")

        U_curr = u_init.copy()
        mu = self.penalty_factor




        # SCA 循环
        for k in range(self.sca_loops):
            # 1. 准备 SCA 参数
            sca_params = [self._get_sca_coefficients(U_curr, t) for t in self.time_points]
            n_refs = np.array([self._get_normal_vector(U_curr[b]) for b in range(B)])

            # 2. 定义内层优化变量: delta_u
            delta_u_0 = np.zeros(B * 3)

            # 3. [关键步骤] 计算 Delta U 的动态边界
            # 使得 0 <= U_curr + delta <= 2pi
            # 即 -U_curr <= delta <= 2pi - U_curr
            u_flat = U_curr.flatten()
            delta_bounds = []
            for val in u_flat:
                # 下界: 0 - val
                # 上界: 2pi - val
                lb = 0.0 - val
                ub = 2 * np.pi - val
                delta_bounds.append((lb, ub))

            # 4. 使用 L-BFGS-B (支持边界的拟牛顿法)
            res = minimize(
                fun=self._loss_function,
                x0=delta_u_0,
                args=(U_curr, sca_params, n_refs, mu),
                method='L-BFGS-B',  # 切换为支持边界的求解器
                bounds=delta_bounds,  # 传入动态边界
                options={'ftol': 1e-4, 'disp': False}
            )

            # 5. 更新 U
            delta_u_opt = res.x.reshape(B, 3)
            U_curr = U_curr + delta_u_opt

            # 增大惩罚
            mu *= self.penalty_growth

            norm_delta = np.linalg.norm(delta_u_opt)
            if k % 2 == 0:
                print(f"   Iter {k + 1}: Loss={res.fun:.4f}, |Delta U|={norm_delta:.4f}")

            if norm_delta < 1e-3:
                print("   >>> 算法收敛")
                break

        return U_curr


# ====================================================================
# 调用示例
# ====================================================================
if __name__ == "__main__":
    # 模拟数据
    np.random.seed(42)
    Q_fixed = np.random.uniform(-10, 10, (B, 3))
    # 确保 Q 满足最小距离 (简化处理)
    for b in range(B): Q_fixed[b, 2] = np.abs(Q_fixed[b, 2])  # z>0

    U_init = np.random.uniform(0, 2 * np.pi, (B, 3))

    # 实例化并运行
    optimizer = PenaltyMethodOptimizerU(q_fixed=Q_fixed, eta=0.5, theta0=np.pi / 6)
    U_opt = optimizer.optimize(U_init)

    print("\n优化后的 U (前2个):")
    print(U_opt[:16])