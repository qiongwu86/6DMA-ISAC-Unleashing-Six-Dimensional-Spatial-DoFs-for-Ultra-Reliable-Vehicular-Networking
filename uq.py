import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds
from itertools import combinations
import warnings

# 忽略数值警告
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from qbub import SurfaceLayoutGenerator
from Qcurr import SCAOptimizerQ
from Pos import FullPSOOptimizer

# ====================================================================
# 1. 全局物理常量
# ====================================================================
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

EPSILON = 2;
EPSILON1 = 4;
RHO = 0.8

# 增益参数
THETA_3DB = np.deg2rad(65);
PHI_3DB = np.deg2rad(65)
G_MAX = 8;
G_S = 25;
G_V = 25

def reshape_vars(X):
    """将一维优化向量还原为 U 和 Q"""
    U = X[:B * 3].reshape(B, 3)
    Q = X[B * 3:].reshape(B, 3)
    return U, Q

def calculate_xyz_tilde(u_b, t):
    f_t = calculate_f_t(t)
    R_matrix = get_rotation_matrix(u_b)
    return -np.dot(R_matrix.T, f_t)

def calculate_theta_tilde(u_b, t):
    z_tilde = calculate_xyz_tilde(u_b, t)[2]
    return np.pi / 2 - np.arccos(np.clip(z_tilde, -1, 1))

def get_rotation_matrix(u_b):
    alpha, beta, gamma = u_b
    sa, ca = np.sin(alpha), np.cos(alpha)
    sb, cb = np.sin(beta), np.cos(beta)
    sg, cg = np.sin(gamma), np.cos(gamma)
    return np.array([
        [cb * cg, cb * sg, -sb],
        [sb * sa * cg - ca * sg, sb * sa * sg + ca * cg, cb * sa],
        [ca * sb * cg + sa * sg, ca * sb * sg - sa * cg, ca * cb]
    ])







def calculate_theta_phi_tilde(u_b, t):
    xb_tilde, yb_tilde, zb_tilde = calculate_xyz_tilde(u_b, t)
    eta_yb = 1 if yb_tilde >= 0 else -1
    denom = np.sqrt(xb_tilde ** 2 + yb_tilde ** 2)

    if denom > 1e-8:
        tilde_phi = np.arccos(np.clip(xb_tilde / denom, -1, 1)) * eta_yb
    else:
        tilde_phi = 0.0
    return tilde_phi

def calculate_psi_0(u_b, t):
    """计算天线间的相差 psi_0(t)"""
    tilde_phi = calculate_theta_phi_tilde(u_b, t)
    return (2 * np.pi * D_NT * np.cos(tilde_phi)) / LAMBDA


def get_normal_vector(u_b):
    return get_rotation_matrix(u_b)[:, 2]

def calculate_mu1_t(u_b, t):
    theta_b_t = calculate_theta_tilde(u_b, t)
    return V / LAMBDA * np.cos(theta_b_t)

def calculate_f_t(t):
    R_cos = R * np.cos(THETA_0)
    R_sin_vt = R * np.sin(THETA_0) - V * t
    theta_t = np.arctan(R_sin_vt / R_cos)
    phi_t = np.arctan(np.sqrt(R_cos ** 2 + R_sin_vt ** 2) / H)
    return np.array([
        np.cos(theta_t) * np.cos(phi_t),
        np.sin(theta_t) * np.cos(phi_t),
        np.sin(phi_t)
    ])


def calculate_angle_metrics(u_b, t):
    f_t = calculate_f_t(t)
    R_mat = get_rotation_matrix(u_b)
    xyz_tilde = -np.dot(R_mat.T, f_t)
    xb, yb, z_tilde = xyz_tilde

    theta_tilde = np.pi / 2 - np.arccos(np.clip(z_tilde, -1, 1))

    denom = np.sqrt(xb ** 2 + yb ** 2)
    eta = 1 if yb >= 0 else -1
    phi_tilde = np.arccos(np.clip(xb / denom, -1, 1)) * eta if denom > 1e-8 else 0.0

    return theta_tilde, phi_tilde


# ====================================================================
# 3. 初始解生成器 (核心修正部分)
# # ====================================================================
def reshape_vars(X):
    U = X[:B * 3].reshape(B, 3)
    Q = X[B * 3:].reshape(B, 3)
    return U, Q

def calculate_violation(X):
    U, Q = reshape_vars(X)
    violation_sum = 0.0
    normals = np.array([get_normal_vector(U[b]) for b in range(B)])

    # 1. 朝向约束
    for b in range(B):
        if np.dot(normals[b], Q[b]) < 0:
            violation_sum += -np.dot(normals[b], Q[b])

    # 2. 成对约束
    for b, j in combinations(range(B), 2):
        diff_jb = Q[j] - Q[b]
        diff_bj = Q[b] - Q[j]
        dist = np.linalg.norm(diff_jb)

        # 最小距离
        if dist < D_MIN:
            violation_sum += (D_MIN - dist) * 10

            # 互不遮挡
        if np.dot(normals[b], diff_jb) > 0:
            violation_sum += np.dot(normals[b], diff_jb)
        if np.dot(normals[j], diff_bj) > 0:
            violation_sum += np.dot(normals[j], diff_bj)

    return violation_sum


# ==========================================
# 核心修正: 随机生成函数
# ==========================================

def generate_valid_configuration():

    u_bounds = [(0, 2 * np.pi)] * (B * 3)
    q_bounds = [(-C, C)] * (B * 3)
    all_bounds = Bounds([b[0] for b in u_bounds + q_bounds], [b[1] for b in u_bounds + q_bounds])


    max_attempts = 100
    for attempt in range(max_attempts):

        # 1. 真正的随机初始化 (基于系统时间熵)
        # 生成完全随机的初始猜测 X0
        X0 = np.concatenate([
            np.random.uniform(0, 2 * np.pi, B * 3),
            np.random.uniform(-C, C, B * 3)
        ])

        # 2. 运行优化器修正位置
        # 使用随机的 X0 作为起点，优化器会收敛到附近的一个可行解
        res = minimize(
            calculate_violation,
            X0,
            method='SLSQP',
            bounds=all_bounds,
            options={'maxiter': 500, 'ftol': 1e-4, 'disp': False}
        )

        final_violation = calculate_violation(res.x)

        # 3. 验证并返回
        if final_violation < 1e-3:
            print(f"✨ 第 {attempt + 1} 次尝试成功生成! (残余误差: {final_violation:.5f})")
            U_opt, Q_opt = reshape_vars(res.x)
            return U_opt, Q_opt

    print("❌ 未能找到解，请检查约束是否过紧。")
    return None, None



# ====================================================================
# 4. 信道与速率计算模块 (修正向量版)
# ====================================================================


def calculate_global_attenuation(t, P):
    R_cos = R * np.cos(THETA_0)
    R_sin_vt = R * np.sin(THETA_0) - V * t
    d_t = np.sqrt(H ** 2 + R_cos ** 2 + R_sin_vt ** 2)

    d_term = d_t ** (-EPSILON)
    phase_term = np.exp(-1j * 2 * np.pi * d_t / LAMBDA)
    return np.sqrt(P * d_term) * phase_term

def calculate_global_attenuation1(t, P):
    R_cos = R * np.cos(THETA_0)
    R_sin_vt = R * np.sin(THETA_0) - V * t
    d_t = np.sqrt(H ** 2 + R_cos ** 2 + R_sin_vt ** 2)

    d_term = d_t ** (-EPSILON1)
    phase_term = np.exp(-1j * 2 * np.pi * d_t / LAMBDA)
    return np.sqrt(P * d_term) * phase_term



def calculate_geometric_phase_ta_vector(q_b, u_b, t):
    """
    计算几何相位差向量 a (qbc,ubc,t)(N=4)
    a = [e^-j*Psi_b, e^-j*(Psi_b + psi_0), ...]^T
    """
    # 1. 计算 n=1 时的参考点 r_b
    rn_ref = calculate_rn_vector(u_b, 1)
    R_mat = get_rotation_matrix(u_b)
    # r_b = q_b + R * r_n
    r_b = q_b + np.dot(R_mat, rn_ref)

    # 2. 计算基准相位 Psi_b
    f_t = calculate_f_t(t)
    Psi_b = np.dot(f_t, r_b)

    # 3. 计算相位差 psi_0
    psi_0_val = calculate_psi_0(u_b, t)

    # 4. 生成向量
    # 相位序列: Psi_b, Psi_b+psi0, Psi_b+2psi0, Psi_b+3psi0
    k_indices = np.arange(N_ELEM)  # [0, 1, 2, 3]
    phases = Psi_b + k_indices * psi_0_val

    t_a_vector = np.exp(-1j * phases)
    return t_a_vector

def calculate_rn_vector(u_b, n_index):
    # 局部参考向量
    l = 0.25 * LAMBDA
    alpha, beta, _ = u_b
    y_shift = (n_index) / 2.0 * D_NT
    if alpha >= 0 and beta >= 0:
        return np.array([l, -y_shift, 0.0])
    else:
        return np.array([l, y_shift, 0.0])


def calculate_ta_vector(q_b, u_b, t):
    # 几何相位差向量
    f_t = calculate_f_t(t)
    R_mat = get_rotation_matrix(u_b)
    rn_ref = calculate_rn_vector(u_b, 0)
    r_b = q_b + np.dot(R_mat, rn_ref)
    Psi_b = np.dot(f_t, r_b)

    _, phi_tilde = calculate_angle_metrics(u_b, t)
    psi_0 = (2 * np.pi * D_NT * np.cos(phi_tilde)) / LAMBDA

    phases = Psi_b + np.arange(N_ELEM) * psi_0
    return np.exp(-1j * phases)


def calculate_gain(u_b, t):
    theta_t, phi_t = calculate_angle_metrics(u_b, t)
    A_H = -np.minimum(12 * (np.abs(phi_t) / PHI_3DB) ** 2, G_V)
    A_V = -np.minimum(12 * (np.abs(theta_t) / THETA_3DB) ** 2, G_S)
    A = G_MAX - np.minimum(-(A_H + A_V), G_S)
    return 10 ** (A / 10.0)


def calculate_h_c_full(Q, U, t):
    # 计算 B*N 维通信信道
    R_vec = R * np.array([np.cos(THETA_0), np.sin(THETA_0)])
    d_t = np.sqrt(H ** 2 + (R_vec[0]) ** 2 + (R_vec[1] - V * t) ** 2)
    path_loss = ((d_t ** (-EPSILON / 2)))
    phase_term=np.exp(-1j * 2 * np.pi * d_t / LAMBDA)

    h_c = np.zeros(B * N_ELEM, dtype=complex)

    for b in range(B):
        u_b, q_b = U[b], Q[b]
        gain = calculate_gain(u_b, t)
        ta_vec = calculate_ta_vector(q_b, u_b, t)

        theta_t, _ = calculate_angle_metrics(u_b, t)
        mu1 = (V / LAMBDA) * np.cos(theta_t)
        doppler = np.exp(1j * 2 * np.pi * mu1 * t)

        block = np.sqrt(PC* path_loss ) * phase_term * np.sqrt(gain) * doppler * ta_vec
        h_c[b * N_ELEM: (b + 1) * N_ELEM] = block
    return h_c


def calculate_h_s_full(Q, U, t):
    """
    计算感知信道 h_s
    输出大小: (B, )，因为 b^H * a 是标量。
    """
    global_factor = calculate_global_attenuation1(t, PS)
    h_s_vector = np.zeros(B, dtype=complex)

    for b in range(B):
        u_b, q_b = U[b], Q[b]
        g_bs = calculate_gain(u_b, t)
        mu_b_t = calculate_mu1_t(u_b, t)

        # a(t) 和 b(t) 实际上就是 t_a 向量 (或者共轭关系)
        # 根据 image_0d35e4.png 和 image_1df81c.png，a(t) 和 t_a 结构一致
        a_vec = calculate_geometric_phase_ta_vector(q_b, u_b, t)
        b_vec = a_vec  # 假设 a=b (收发共用或结构相同)

        surface_gain =g_bs
        # 指数项 e^(j 4pi mu t)
        phase_exponent = np.exp(1j * 4 * np.pi * mu_b_t * t)

        # 内积: b^H * a
        array_cohesion = np.vdot(b_vec, a_vec)  # vdot handles complex conjugate automatically (a^H * b)
        # 注意: numpy vdot 是 x^H * y。公式是 b^H * a。所以参数顺序为 (b_vec, a_vec)。

        h_s_vector[b] = surface_gain * phase_exponent * array_cohesion

    return global_factor * h_s_vector*rho


def get_rates_at_t(U, Q, t, eta ):
    # 1. 计算所有表面的通信 SNR
    hc_full = calculate_h_c_full(Q, U, t)
    hc_reshaped = hc_full.reshape(B, N_ELEM)
    snr_b = np.sum(np.abs(hc_reshaped) ** 2, axis=1)   # 注意 hc 已包含 sqrt(P)

    # 2. 模式切换 logic
    max_snr = np.max(snr_b)
    mode_vec = np.where(snr_b > eta * max_snr, 1, 0)

    # 线性的信噪比 (倍数关系)
    # 先将噪声转回 mW: Noise_mW = 10^(noise_dbm / 10)

    snr_linear = PS / SIGMA_SQ

    # 3. 计算 Rc
    hc_active = hc_full.reshape(B, N_ELEM)[mode_vec == 1].flatten()
    if hc_active.size > 0:
        Rc = np.log2(1 + (snr_linear *(np.sum(np.abs(hc_active))** 2)))
    else:
        Rc = 0.0

    # 4. 计算 Rs
    hs_full = calculate_h_s_full(Q, U, t)
    hs_active = hs_full[mode_vec == 0]
    if hs_active.size > 0:
        # 修正: Rs 公式中 h_s 已经包含 sqrt(P)，直接取能量/噪声
        Rs = (1.0 / L_FRAME) * np.log2(1 + (snr_linear *(np.sum(np.abs(hs_active)) ** 2)))
    else:
        Rs = 0.0

    # 5. 综合速率
    R_total =  Rc +  Rs
    return Rc, Rs, R_total, np.sum(mode_vec)
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

D_MIN = np.sqrt(2) / LAMBDA + LAMBDA / 2
params = {
    "B": B,
    "N_ELEM": N_ELEM,
    "LAMBDA": LAMBDA,
    "V": V,
    "R": R,
    "H": H,
    "THETA_0": THETA_0,
    "D_NT": D_NT,
    "PC": PC,
    "PS": PS,
    "sigma": SIGMA_SQ,
    "EPSILON": EPSILON,
    "EPSILON1": EPSILON1,
    "PHI_3DB": PHI_3DB,
    "THETA_3DB": THETA_3DB,
    "G_MAX": G_MAX,
    "G_S": G_S,
    "G_V": G_V,
    "rho": rho,
    "L_FRAME": L_FRAME,
    "D_MIN": D_MIN,
    "C": C
}


class SCPOptimizerU:
    def __init__(self, q_fixed, eta, theta0, B=16, H=100.0, R=200.0, V=20.0):
        """
        基于罚函数的带信赖域序列凸规划 (SCP) 优化器 - 优化姿态 U
        :param q_fixed: (B, 3) 固定的位置数组
        """
        self.Q = q_fixed
        self.eta = eta
        self.theta0 = theta0
        self.ln2 = np.log(2)

        # 1. 物理参数 (需与环境一致)
        self.B = B
        self.H = H;
        self.R = R;
        self.V = V
        self.PC = 40.0 * 1e-3;
        self.PS = self.PC
        self.SIGMA_SQ = (10 ** (-90.0 / 10.0)) * 1e-3

        # 2. 增益参数
        self.THETA_3DB = np.deg2rad(65);
        self.PHI_3DB = np.deg2rad(65)
        self.G_MAX = 8;
        self.G_S = 25;
        self.G_V = 25
        self.EPSILON = 2;
        self.EPSILON1 = 4;
        self.RHO = 0.8
        self.N_ELEM = 4;
        self.L_FRAME = 1.0

        # 3. SCP 算法超参数



        self.time_points = np.linspace(0, 2 * self.R / self.V, 5)
        # 修改 SCPOptimizerU 类内部的默认值，或者在初始化时覆盖
        self.trust_region = 0.5  # 增大信赖域 (原来是 0.2) -> 允许迈大步
        self.mu_init = 0.1  # 减小初始罚因子 (原来是 10.0) -> 初期不怎么管遮挡，先冲速率
        self.mu_growth = 1.2  # 减缓罚增长 (原来是 2.0) -> 给算法更多喘息时间
        self.sca_loops = 30  # 增加迭代次数 (原来是 15) -> 只要没收敛就接着算
    # ------------------------------------------------------
    # 基础几何运算
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

    def _calc_linearized_normal(self, delta_u, n_ref):
        """
        利用李代数性质计算线性化后的法向量
        n(u + du) ≈ n(u) + cross(delta_u, n(u))
        """
        cross_term = np.cross(delta_u, n_ref)
        return n_ref + cross_term

    # ------------------------------------------------------
    # 物理引擎：计算信道能量
    # ------------------------------------------------------
    def _calculate_channel_energy(self, U, t):
        target_pos = np.array([self.R * np.cos(self.theta0),
                               self.R * np.sin(self.theta0) - self.V * t,
                               self.H])
        hc_e = np.zeros(self.B)
        hs_e = np.zeros(self.B)

        U_reshaped = U.reshape(self.B, 3)

        for b in range(self.B):
            vec = target_pos - self.Q[b]
            dist = np.linalg.norm(vec)
            if dist < 1.0: dist = 1.0

            R_mat = self._get_rotation_matrix(U_reshaped[b])
            v_loc = np.dot(R_mat.T, vec / dist)
            xb, yb, zb = v_loc

            theta_dev = np.arccos(np.clip(zb, -1, 1))
            phi_tilde = np.arctan2(yb, xb)

            val_p = 12 * (np.abs(phi_tilde) / self.PHI_3DB) ** 2
            val_t = 12 * (np.abs(theta_dev) / self.THETA_3DB) ** 2
            gain_db = self.G_MAX - np.minimum(-(-np.minimum(val_p, self.G_V) - np.minimum(val_t, self.G_S)), self.G_S)
            gain = 10 ** (gain_db / 10.0)

            pl_c = dist ** (-self.EPSILON)
            pl_s = dist ** (-self.EPSILON1)

            hc_e[b] = self.PC * gain * pl_c * self.N_ELEM
            hs_e[b] = self.PS * gain * pl_s * (self.N_ELEM ** 2) * (self.RHO ** 2)

        return hc_e, hs_e

    # ------------------------------------------------------
    # SCA 系数计算 (修正版)
    # ------------------------------------------------------
    def _get_sca_coeffs(self, U_ref, t):
        """
        基于 SCA 公式 R(P) >= alpha * P + beta
        """
        # 1. 计算参考点能量
        pc, ps = self._calculate_channel_energy(U_ref, t)

        # 2. 计算系数 (alpha, beta)
        # 通信
        snr_c = pc / self.SIGMA_SQ
        # alpha = 1 / (sigma^2 * ln2 * (1+SNR))
        ac = (1.0 / self.SIGMA_SQ) / (self.ln2 * (1 + snr_c))
        bc = np.log2(1 + snr_c) - ac * pc

        # 感知
        snr_s = ps / self.SIGMA_SQ
        as_ = (1.0 / self.SIGMA_SQ) / (self.ln2 * (1 + snr_s))
        bs = np.log2(1 + snr_s) - as_ * ps

        return ac, bc, as_, bs

    # ------------------------------------------------------
    # 最终目标函数：SCA Rate + 线性化 Penalty
    # ------------------------------------------------------
    def _objective_function(self, delta_u_flat, U_ref, sca_params_list, n_refs, mu, weight_a=0.5):
        delta_U = delta_u_flat.reshape(self.B, 3)
        # 这里只是临时叠加用于计算能量，真正的更新在外部
        U_curr = U_ref + delta_U

        # --- Part 1: SCA 近似速率 ---
        rate_obj = 0.0

        for idx, t in enumerate(self.time_points):
            ac, bc, as_, bs = sca_params_list[idx]

            # 重新计算能量 (因为 U 变了，能量是非凸的，但在信赖域内直接计算)
            pc, ps = self._calculate_channel_energy(U_curr, t)

            snr = pc / self.SIGMA_SQ
            mv = np.where(snr > self.eta * (np.max(snr) + 1e-12), 1, 0)

            # 使用 SCA 线性模型 R_hat = a*P + b
            rc_hat = ac * pc + bc
            rs_hat = (1.0 / self.L_FRAME) * (as_ * ps + bs)

            val = np.sum(np.where(mv == 1, weight_a * rc_hat, (1 - weight_a) * rs_hat))
            rate_obj += val

        avg_rate = rate_obj / len(self.time_points)

        # --- Part 2: 线性化几何惩罚 (Linearized Penalty) ---
        penalty = 0.0
        for b in range(self.B):
            n_lin = self._calc_linearized_normal(delta_U[b], n_refs[b])

            # 约束 A: 朝向 (要求 n^T q >= 0, 即 -n^T q <= 0)
            g_orient = -np.dot(n_lin, self.Q[b])
            if g_orient > 0: penalty += g_orient ** 2

            # 约束 B: 互不遮挡
            for j in range(self.B):
                if b == j: continue
                # 要求 n_b^T (q_j - q_b) <= 0
                diff = self.Q[j] - self.Q[b]
                g_occl = np.dot(n_lin, diff)
                if g_occl > 0: penalty += g_occl ** 2

        # 目标：最大化速率 -> 最小化 (-速率 + 罚项)
        return -avg_rate + mu * penalty

    # ------------------------------------------------------
    # 主优化循环
    # ------------------------------------------------------
    def optimize(self, u_init):
        print(f"--- 启动 SCP-U 优化 (Trust={self.trust_region}, Eta={self.eta}) ---")

        U_curr = u_init.copy()
        mu = self.mu_init

        for k in range(self.sca_loops):
            # 1. 准备参考数据 (Gradient/Reference Info)
            sca_params = [self._get_sca_coeffs(U_curr, t) for t in self.time_points]
            n_refs = np.array([self._get_normal_vector(U_curr[b]) for b in range(self.B)])

            # 2. 信赖域边界处理
            delta_bounds = []
            u_flat = U_curr.flatten()
            for val in u_flat:
                # 物理边界 [0, 2pi] (可选)
                # p_lb = 0.0 - val; p_ub = 2*np.pi - val

                # 主要是信赖域 [-delta, delta]
                lb = -self.trust_region
                ub = self.trust_region

                # 防止空集 (robustness)
                if lb > ub: lb = ub - 1e-6
                delta_bounds.append((lb, ub))

            # 3. 求解子问题 (L-BFGS-B)
            delta_u_0 = np.zeros(self.B * 3)  # 从 0 开始搜

            res = minimize(
                fun=self._objective_function,
                x0=delta_u_0,
                args=(U_curr, sca_params, n_refs, mu),
                method='L-BFGS-B',
                bounds=delta_bounds,
                options={'ftol': 1e-4, 'disp': False, 'maxiter': 50}
            )

            # 4. 更新状态
            delta_opt = res.x.reshape(self.B, 3)
            step_size = np.linalg.norm(delta_opt)

            U_curr = U_curr + delta_opt

            # 5. 更新罚因子
            mu *= self.mu_growth

            if k % 2 == 0:
                # 打印当前真实 Loss (Penalty 是否归零)
                print(f"   Iter {k + 1}/{self.sca_loops}: Loss={res.fun:.4f}, Step={step_size:.4f}, Mu={mu:.1f}")

            if step_size < 1e-3:
                print("   >>> SCP-U 收敛")
                break

        # 结果约束回 [0, 2pi]
        return U_curr % (2 * np.pi)


# ====================================================================
# 辅助函数：计算真实速率用于绘图
# ====================================================================
def calculate_true_rates_over_time(Q, U, eta, theta0):
    """计算真实物理速率 (非近似)"""
    T_total = 2 * R / V
    time_pts = np.linspace(0, T_total, 20)

    rt_list = []

    # 临时的物理计算器
    # 复用 SCPOptimizerU 中的方法，虽然有点冗余但方便
    helper = SCPOptimizerU(Q, eta, theta0)

    for t in time_pts:
        pc, ps = helper._calculate_channel_energy(U, t)
        snr = ps / SIGMA_SQ
        mv = np.where(snr > eta * (np.max(snr) + 1e-12), 1, 0)

        # 真实 log 速率
        ec = np.sum(pc[mv == 1])
        rc = np.log2(1 + ec / SIGMA_SQ) if ec > 0 else 0
        es = np.sum(ps[mv == 0])
        rs = (1.0 / L_FRAME) * np.log2(1 + es / SIGMA_SQ) if es > 0 else 0

        # 假设加权 0.5
        rt_list.append(0.5 * rc + 0.5 * rs)

    return time_pts, rt_list


# ====================================================================
# 主程序
# ====================================================================
if __name__ == "__main__":
    # 1. 初始化
    t_val = 1.5
    generator = SurfaceLayoutGenerator(B=B, C=C, theta_0=np.pi / 3,H=H,R=R,V=V)
    Q_int, U_int = generator.generate()

    generator1 = SCAOptimizerQ(B=B, C=C, H=H, R=R, V=V, lambda_val=0.125, eta=0.5, theta0=np.pi / 6)
    Q_fixed = generator1.optimize(q_init=Q_int, u_fixed=U_int, t_input=t_val)

    generator2 = SCPOptimizerU( q_fixed=Q_fixed, eta=0.5, theta0=np.pi / 6, B=16, H=100.0, R=200.0, V=20.0)
    U_fixed = generator2.optimize(U_int)


    ETA_VAL = 0.5
    THETA_VAL = np.pi / 6

    # 2. 计算优化前速率
    print("计算初始速率...")
    t_axis, r_pre =  calculate_true_rates_over_time(Q_int, U_int, ETA_VAL, THETA_VAL)

    # 3. 执行优化


    # 4. 计算优化后速率
    print("计算优化后速率...")
    _, r_post =  calculate_true_rates_over_time(Q_fixed, U_fixed, ETA_VAL, THETA_VAL)

    # 5. 绘图对比
    plt.figure(figsize=(10, 6))
    plt.plot(t_axis, r_pre, 'b--o', label='优化前 (Initial U)', alpha=0.6)
    plt.plot(t_axis, r_post, 'r-s', label='优化后 (Optimized U)', linewidth=2)

    plt.title(f'优化前后通感总速率对比 (Penalty SCP Algorithm)\nEta={ETA_VAL}, TrustRegion=0.2')
    plt.xlabel('时间 (s)')
    plt.ylabel('加权总速率 (bits/s/Hz)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)

    # 填充提升区域
    plt.fill_between(t_axis, r_pre, r_post, color='red', alpha=0.1, label='Performance Gain')

    plt.tight_layout()
    plt.show()

    print("\n对比完成。")
    print(f"平均速率提升: {np.mean(r_pre):.4f} -> {np.mean(r_post):.4f}")