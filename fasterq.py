# 文件名: fasterq.py
import numpy as np
from scipy.optimize import minimize, Bounds


class FastSCAOptimizerQ:
    def __init__(self, B, H, R, V, theta0, eta=0.5):

        self.B = B
        self.H = H
        self.R = R
        self.V = V
        self.theta0 = theta0
        self.eta = eta

        # 物理常量
        self.PC = 40.0 * 1e-3
        self.PS = 40.0 * 1e-3
        self.SIGMA_SQ = (10 ** (-90.0 / 10.0)) * 1e-3
        self.EPSILON = 2
        self.EPSILON1 = 4
        self.N_ELEM = 4
        self.L_FRAME = 0.1  # 关键参数
        self.ln2 = np.log(2)

        # 增益参数
        self.THETA_3DB = np.deg2rad(65)
        self.PHI_3DB = np.deg2rad(65)
        self.G_MAX = 8;
        self.G_S = 25;
        self.G_V = 25
        self.RHO = 0.8

    def _get_rotation_matrix(self, u_b):
        a, b, g = u_b
        return np.array([[np.cos(b) * np.cos(g), np.cos(b) * np.sin(g), -np.sin(b)],
                         [np.sin(b) * np.sin(a) * np.cos(g) - np.cos(a) * np.sin(g),
                          np.sin(b) * np.sin(a) * np.sin(g) + np.cos(a) * np.cos(g), np.cos(b) * np.sin(a)],
                         [np.cos(a) * np.sin(b) * np.cos(g) + np.sin(a) * np.sin(g),
                          np.cos(a) * np.sin(b) * np.sin(g) - np.sin(a) * np.cos(g), np.cos(a) * np.cos(b)]])

    def _calculate_physics(self, Q, U, t):
        """辅助函数：计算距离、增益、功率"""
        target_pos = np.array([self.R * np.cos(self.theta0), self.R * np.sin(self.theta0) - self.V * t, self.H])

        pc_list = np.zeros(self.B)
        ps_list = np.zeros(self.B)
        dists = np.zeros(self.B)
        vecs = np.zeros((self.B, 3))
        gains = np.zeros(self.B)

        for b in range(self.B):
            vec = target_pos - Q[b]
            dist = max(np.linalg.norm(vec), 1.0)

            dists[b] = dist
            vecs[b] = vec

            # 计算增益
            R_mat = self._get_rotation_matrix(U[b])
            v_loc = np.dot(R_mat.T, vec / dist)
            theta_dev = np.arccos(np.clip(v_loc[2], -1, 1))
            phi_tilde = np.arctan2(v_loc[1], v_loc[0])
            val_p = 12 * (np.abs(phi_tilde) / self.PHI_3DB) ** 2
            val_t = 12 * (np.abs(theta_dev) / self.THETA_3DB) ** 2
            gain_db = self.G_MAX - np.minimum(-(-np.minimum(val_p, self.G_V) - np.minimum(val_t, self.G_S)), self.G_S)
            gains[b] = 10 ** (gain_db / 10.0)

            # 功率
            pl_c = dist ** (-self.EPSILON)
            pl_s = dist ** (-self.EPSILON1)

            pc_list[b] = self.PC * gains[b] * pl_c * self.N_ELEM
            ps_list[b] = self.PS * gains[b] * pl_s * (self.N_ELEM ** 2) * (self.RHO ** 2)

        return pc_list, ps_list, dists, vecs, gains

    # =========================================================
    #  这就是报错说找不到的函数，必须加上！
    # =========================================================
    def _get_sca_weights(self, Q, U, t):
        """
        计算 SCA 的一阶泰勒展开系数 (Weights)
        这是绘图脚本 run_math_step_by_step 需要调用的核心函数
        """
        pc, ps, _, _, _ = self._calculate_physics(Q, U, t)

        # 1. 模式选择
        snr = pc / self.SIGMA_SQ
        max_snr = np.max(snr) if np.max(snr) > 0 else 1e-12
        mv = np.where(snr > self.eta * max_snr, 1, 0)  # 1=Comm, 0=Sens

        weights_c = np.zeros(self.B)
        weights_s = np.zeros(self.B)

        # 2. 通信梯度权重 (标量分母)
        sum_pc = np.sum(pc[mv == 1])
        denom_c = self.ln2 * (self.SIGMA_SQ + sum_pc)
        weights_c[mv == 1] = 1.0 / denom_c

        # 3. 感知梯度权重 (向量分母) - 关键修正
        # 必须是 (sigma + P_i)，而不是 Sum(P)
        denom_s = self.L_FRAME * self.ln2 * (self.SIGMA_SQ + ps[mv == 0])
        weights_s[mv == 0] = 1.0 / denom_s

        # 综合权重
        total_weights = 0.5 * weights_c + 0.5 * weights_s

        return total_weights, mv

    def _objective_and_gradient(self, q_flat, U_fixed, t_list, sca_params_list):
        Q = q_flat.reshape(self.B, 3)
        total_loss = 0.0
        total_grad = np.zeros_like(Q)

        for idx, t in enumerate(t_list):
            weights, mv = sca_params_list[idx]

            # 重新计算物理量
            pc, ps, dists, vecs, gains = self._calculate_physics(Q, U_fixed, t)

            # --- 1. Loss ---
            current_p = np.where(mv == 1, pc, ps)
            # 最小化负效用 (Maximize Linear Approx)
            total_loss -= np.sum(weights * current_p)

            # --- 2. Gradient ---
            eps_vec = np.where(mv == 1, self.EPSILON, self.EPSILON1)

            # dP/dQ = P * eps * vec / dist^2
            grad_per_surface = current_p[:, np.newaxis] * eps_vec[:, np.newaxis] * (vecs / (dists[:, np.newaxis] ** 2))

            # dLoss/dQ = - weight * dP/dQ
            total_grad -= weights[:, np.newaxis] * grad_per_surface

        return total_loss / len(t_list), total_grad.flatten() / len(t_list)

    def optimize(self, Q_init, U_fixed):
        Q_curr = Q_init.copy()
        t_list = np.linspace(0, 2 * self.R / self.V, 5)

        for k in range(10):  # 建议跑 10 轮
            sca_params_list = []
            for t in t_list:
                weights, mv = self._get_sca_weights(Q_curr, U_fixed, t)
                sca_params_list.append((weights, mv))

            res = minimize(
                fun=self._objective_and_gradient,
                x0=Q_curr.flatten(),
                args=(U_fixed, t_list, sca_params_list),
                method='L-BFGS-B',
                jac=True,
                bounds=Bounds([-self.R] * self.B * 3, [self.R] * self.B * 3),
                options={'ftol': 1e-4, 'maxiter': 20, 'disp': False}
            )
            Q_curr = res.x.reshape(self.B, 3)

        return Q_curr