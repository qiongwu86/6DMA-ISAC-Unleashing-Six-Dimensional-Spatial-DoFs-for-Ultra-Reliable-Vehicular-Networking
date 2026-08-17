import numpy as np
from scipy.optimize import minimize, Bounds
from itertools import combinations
import warnings
from qbub import SurfaceLayoutGenerator
# 忽略数值计算中的 RuntimeWarning (如 log(0))
warnings.filterwarnings("ignore")


class SCAOptimizerQ:
    def __init__(self, B, C, H, R, V, lambda_val=0.125, eta=0.5, theta0=np.pi / 6):
        """
        初始化 SCA 优化器 (针对位置 Q)
        :param B: 表面数量
        :param C: 空间边界
        :param H, R, V: 场景几何参数
        :param lambda_val: 波长
        :param eta: 模式选择阈值
        :param theta0: 车辆初始角度
        """
        # 1. 系统参数
        self.B = B
        self.C = C
        self.H = H
        self.R = R
        self.V = V
        self.eta = eta
        self.theta0 = theta0

        # 2. 物理常量
        self.PC = 40.0 * 1e-3  # 40mW
        self.PS = self.PC
        self.SIGMA_SQ = (10 ** (-90.0 / 10.0)) * 1e-3
        self.ln2 = np.log(2)

        self.EPSILON = 2
        self.EPSILON1 = 4
        self.RHO = 0.8
        self.N_ELEM = 4
        self.L_FRAME = 1.0

        # 3. 几何约束参数
        self.d_min = (np.sqrt(2) * lambda_val + lambda_val) / 2
        self.THETA_3DB = np.deg2rad(65)
        self.PHI_3DB = np.deg2rad(65)
        self.G_MAX = 8;
        self.G_S = 25;
        self.G_V = 25

        # 4. SCA 迭代参数
        self.sca_loops = 10  # 外层线性化迭代次数

    # ==============================================================
    #  基础物理计算
    # ==============================================================
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

    def _calculate_channel_energy(self, Q, U, t):
        """计算物理接收能量 (真实值)"""
        target_pos = np.array([
            self.R * np.cos(self.theta0),
            self.R * np.sin(self.theta0) - self.V * t,
            self.H
        ])

        hc_e = np.zeros(self.B)
        hs_e = np.zeros(self.B)

        for b in range(self.B):
            vec = target_pos - Q[b]
            dist = np.linalg.norm(vec)
            if dist < 1.0: dist = 1.0

            R_mat = self._get_rotation_matrix(U[b])
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

    def _get_sca_coeffs(self, Q_ref, U_fixed, t):
        """计算 SCA 线性化系数 alpha, beta"""
        pc, ps = self._calculate_channel_energy(Q_ref, U_fixed, t)

        snr_c = pc / self.SIGMA_SQ
        ac = (1.0 / self.SIGMA_SQ) / (self.ln2 * (1 + snr_c))
        bc = np.log2(1 + snr_c) - ac * pc

        snr_s = ps / self.SIGMA_SQ
        as_ = (1.0 / self.SIGMA_SQ) / (self.ln2 * (1 + snr_s))
        bs = np.log2(1 + snr_s) - as_ * ps
        return ac, bc, as_, bs

    # ==============================================================
    #  优化目标与约束 (SCA)
    # ==============================================================
    def _convex_objective(self, q_flat, U_fixed, t_list, sca_params_list, weight_a=0.5):
        """
        目标函数：最小化 (-总速率)
        使用 SCA 近似公式 R = a*P + b
        """
        Q_curr = q_flat.reshape(self.B, 3)
        rate_obj = 0.0

        # 遍历时间点 (可能是单个时间 t，也可能是列表)
        # 如果输入是单个 t，t_list 应该是一个 [t]
        for idx, t in enumerate(t_list):
            ac, bc, as_, bs = sca_params_list[idx]
            # 计算当前 Q 下的物理能量 P
            pc, ps = self._calculate_channel_energy(Q_curr, U_fixed, t)

            snr = pc / self.SIGMA_SQ
            mv = np.where(snr > self.eta * (np.max(snr) + 1e-12), 1, 0)

            # 近似速率
            rc = ac * pc + bc
            rs = (1.0 / self.L_FRAME) * (as_ * ps + bs)
            val = np.sum(np.where(mv == 1, weight_a * rc, (1 - weight_a) * rs))
            rate_obj += val

        return -rate_obj / len(t_list)

    def _linearized_constraints(self, q_flat, Q_ref, U_fixed):
        """
        构建线性化约束 (Constraint Linearization)
        包括：朝向、遮挡、最小距离(SCA切平面)
        返回值为数组，要求 >= 0
        """
        Q_val = q_flat.reshape(self.B, 3)
        cons_vals = []

        # 获取法向量
        normals = np.array([self._get_normal_vector(U_fixed[b]) for b in range(self.B)])

        # 1. 朝向约束: n^T * q >= 0
        for b in range(self.B):
            cons_vals.append(np.dot(normals[b], Q_val[b]))

        # 2. 互不遮挡与最小距离
        for b, j in combinations(range(self.B), 2):
            # A. 最小距离约束 (非凸 -> 凸近似)
            # 原始: ||q_b - q_j|| >= d_min
            # 线性化: v_ref^T * ( (q_b - q_j) - (q_b_ref - q_j_ref) ) + dist_ref >= d_min
            # 简化后: v_ref^T * (q_b - q_j) >= d_min
            diff_ref = Q_ref[b] - Q_ref[j]
            dist_ref = np.linalg.norm(diff_ref)

            if dist_ref > 1e-6:
                v_ref = diff_ref / dist_ref  # 梯度方向
                # 约束: v^T * (q_b - q_j) - d_min >= 0
                linear_dist_constr = np.dot(v_ref, Q_val[b] - Q_val[j]) - self.d_min
                cons_vals.append(linear_dist_constr)

            # B. 遮挡约束 (已经是线性的)
            # n_b^T (q_j - q_b) <= 0  =>  -n_b^T (q_j - q_b) >= 0
            cons_vals.append(-np.dot(normals[b], Q_val[j] - Q_val[b]))
            cons_vals.append(-np.dot(normals[j], Q_val[b] - Q_val[j]))

        return np.array(cons_vals)

    # ==============================================================
    #  主优化函数
    # ==============================================================
    def optimize(self, q_init, u_fixed, t_input):
        """
        执行优化
        :param q_init: (B, 3) 初始位置
        :param u_fixed: (B, 3) 固定姿态
        :param t_input: 可以是单个时间浮点数 t，也可以是时间列表 [t1, t2...]
        :return: 优化后的 Q
        """
        # 处理时间输入，统一转为列表
        if isinstance(t_input, (float, int)):
            t_list = [float(t_input)]
        else:
            t_list = t_input

        print(f"--- 启动 SCA-Q 优化 (Time points: {len(t_list)}) ---")

        Q_curr = q_init.copy()
        Q_ref = q_init.copy()

        # 边界约束: |q| <= C
        bounds = Bounds([-self.C] * (self.B * 3), [self.C] * (self.B * 3))

        # 外层 SCA 循环
        for k in range(self.sca_loops):
            # 1. 计算当前参考点下的 SCA 系数
            sca_params_list = [self._get_sca_coeffs(Q_ref, u_fixed, t) for t in t_list]

            # 2. 定义约束 (传入当前的 Q_ref 进行线性化)
            cons = {'type': 'ineq',
                    'fun': lambda x: self._linearized_constraints(x, Q_ref, u_fixed)}

            # 3. 内层求解: 梯度下降 (SLSQP)
            res = minimize(
                fun=self._convex_objective,
                x0=Q_curr.flatten(),
                args=(u_fixed, t_list, sca_params_list),
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'ftol': 1e-4, 'disp': False, 'maxiter': 50}
            )

            # 4. 更新状态
            Q_next = res.x.reshape(self.B, 3)
            diff = np.linalg.norm(Q_next - Q_curr)

            Q_curr = Q_next.copy()
            Q_ref = Q_next.copy()

            if k % 2 == 0:
                print(f"   Iter {k + 1}/{self.sca_loops}: Loss={res.fun:.4f}, Step={diff:.4f}")

            if diff < 1e-3:
                print("   >>> Q 优化收敛")
                break

        return Q_curr


import numpy as np
import matplotlib.pyplot as plt

# 设置绘图风格，防止中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class RateCalculator:
    def __init__(self, optimizer_instance):
        """
        初始化计算器，直接复用优化器中的物理参数，防止参数不一致
        :param optimizer_instance: 之前实例化好的 SCAOptimizerQ 或 JointOptimizer 对象
        """
        # 从优化器实例中提取物理参数
        self.B = optimizer_instance.B
        self.H = optimizer_instance.H
        self.R = optimizer_instance.R
        self.V = optimizer_instance.V
        self.theta0 = optimizer_instance.theta0
        self.PC = optimizer_instance.PC
        self.PS = optimizer_instance.PS
        self.SIGMA_SQ = optimizer_instance.SIGMA_SQ
        self.ln2 = optimizer_instance.ln2
        self.L_FRAME = optimizer_instance.L_FRAME

        # 物理计算辅助函数 (复用优化器的方法)
        self._get_rotation_matrix = optimizer_instance._get_rotation_matrix
        self._calculate_channel_energy = optimizer_instance._calculate_channel_energy
        self._get_sca_coeffs = optimizer_instance._get_sca_coeffs

    def compute_instant_rates(self, Q, U, eta, t):
        """
        计算某一时刻 t 的瞬时速率 (通信, 感知, 总和)
        使用 SCA 凸近似公式 R = alpha * P + beta
        """
        # 1. 计算物理能量
        pc, ps = self._calculate_channel_energy(Q, U, t)

        # 2. 计算 SCA 系数
        ac, bc, as_, bs = self._get_sca_coeffs(Q, U, t)

        # 3. 模式选择 (Mode Selection)
        snr_c = pc / self.SIGMA_SQ
        max_snr = np.max(snr_c) if np.max(snr_c) > 0 else 1e-12
        mv = np.where(snr_c > eta * max_snr, 1, 0)

        # 4. 计算速率 (SCA 公式)
        # 通信速率 (只算 mv=1)
        rc_elem = ac * (pc)**2 + bc
        rate_comm = np.sum(rc_elem * mv)

        # 感知速率 (只算 mv=0)
        rs_elem = (1.0 / self.L_FRAME) * (as_ * (ps)**2 + bs)
        rate_sens = np.sum(rs_elem * (1 - mv))

        # 总速率 (加权 0.5)
        rate_total = 0.5 * rate_comm + 0.5 * rate_sens

        return rate_comm, rate_sens, rate_total

    def plot_trajectory_comparison(self, Q_init, U_init, Q_opt, U_opt, eta, num_points=50):
        """
        生成优化前后随时间变化的速率对比图
        """
        # 1. 生成时间轴 [0, 2R/V]
        T_total = 2 * self.R / self.V
        t_axis = np.linspace(0, T_total, num_points)

        # 2. 存储数据
        res_init = {'comm': [], 'sens': [], 'total': []}
        res_opt = {'comm': [], 'sens': [], 'total': []}

        print(f"--- 正在计算轨迹速率 (T=0~{T_total:.1f}s, Points={num_points}) ---")

        for t in t_axis:
            # 计算优化前
            rc1, rs1, rt1 = self.compute_instant_rates(Q_init, U_init, eta, t)
            res_init['comm'].append(rc1)
            res_init['sens'].append(rs1)
            res_init['total'].append(rt1)

            # 计算优化后
            rc2, rs2, rt2 = self.compute_instant_rates(Q_opt, U_opt, eta, t)
            res_opt['comm'].append(rc2)
            res_opt['sens'].append(rs2)
            res_opt['total'].append(rt2)

        # 3. 绘图
        plt.figure(figsize=(12, 10))

        # 子图 1: 总速率对比
        plt.subplot(3, 1, 1)
        plt.plot(t_axis, res_init['total'], 'b--o', label='优化前 (Initial)', alpha=0.6, markersize=4)
        plt.plot(t_axis, res_opt['total'], 'r-s', label='优化后 (Optimized)', linewidth=2, markersize=4)
        plt.title(f'总速率对比 (Total Sum Rate) | Eta={eta}')
        plt.ylabel('Rate (bits/s/Hz)')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()

        # 子图 2: 通信速率对比
        plt.subplot(3, 1, 2)
        plt.plot(t_axis, res_init['comm'], 'b--', label='优化前 Comm')
        plt.plot(t_axis, res_opt['comm'], 'r-', label='优化后 Comm')
        plt.ylabel('Comm Rate')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()

        # 子图 3: 感知速率对比
        plt.subplot(3, 1, 3)
        plt.plot(t_axis, res_init['sens'], 'g--', label='优化前 Sensing')
        plt.plot(t_axis, res_opt['sens'], 'm-', label='优化后 Sensing')
        plt.xlabel('Time (s)')
        plt.ylabel('Sensing Rate')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()

        plt.tight_layout()
        plt.show()

        # 打印平均值提升
        avg_init = np.mean(res_init['total'])
        avg_opt = np.mean(res_opt['total'])
        gain = (avg_opt - avg_init) / avg_init * 100 if avg_init > 0 else 0
        print(f"\n平均总速率: {avg_init:.4f} -> {avg_opt:.4f} (提升 {gain:.2f}%)")

# ==========================================
# 调用示例
# ==========================================
if __name__ == "__main__":
    # 1. 模拟数据
    B = 16
    C = 20.0
    H = 100;
    R = 200;
    V = 20

    # 假设已有合规的 Q, U
    generator = SurfaceLayoutGenerator(B=B, C=C, theta_0=np.pi / 3, H=H, R=R, V=V)
    Q_int, U_int = generator.generate()

    # 2. 实例化优化器
    optimizer = SCAOptimizerQ(B, C, H, R, V, eta=0.5)

    # 3. 执行优化 (输入 Q, U, eta已在init中, t=1.5s)
    t_val = 1.5
    Q_optimized = optimizer.optimize(q_init=Q_int, u_fixed=U_int, t_input=t_val)

    print("\n优化完成。")
    print("Q_start shape:", Q_int.shape)
    print("Q_opt shape:  ", Q_optimized.shape)


    rate_calc = RateCalculator(optimizer)



    U_init_plot = U_int
    U_opt_plot = U_int



        # 3. 生成对比图
    rate_calc.plot_trajectory_comparison(
            Q_init=Q_int,  # 初始位置 (注意要用 copy 备份过的原始值)
            U_init=U_init_plot,
            Q_opt=Q_optimized,  # 优化后位置
            U_opt=U_opt_plot,
            eta=0.5
        )