import numpy as np
from scipy.optimize import minimize, Bounds


class FastSCPOptimizerU:
    def __init__(self, q_fixed, eta, theta0, B=16, H=100.0, R=200.0, V=20.0):
        self.Q = q_fixed
        self.eta = eta
        self.theta0 = theta0

        # 物理参数
        self.B = B;
        self.H = H;
        self.R = R;
        self.V = V
        self.PC = 40.0 * 1e-3
        self.SIGMA_SQ = (10 ** (-90.0 / 10.0)) * 1e-3
        self.N_ELEM = 4
        self.L_FRAME = 1.0
        self.ln2 = np.log(2)

        # 增益参数
        self.THETA_3DB = np.deg2rad(65)
        self.PHI_3DB = np.deg2rad(65)
        self.G_MAX = 8;
        self.G_S = 25;
        self.G_V = 25
        self.EPSILON = 2

        # 优化参数
        self.mu_init = 1.0
        self.mu_growth = 1.5
        self.sca_loops = 15
        self.trust_region = 0.5

        # ------------------------------------------------------------------

    # 核心数学：旋转矩阵及其导数
    # ------------------------------------------------------------------
    def _calc_R_and_Derivs(self, u_b):
        """
        同时计算 R 及其对 alpha, beta, gamma 的偏导数矩阵
        返回: R, dR_da, dR_db, dR_dg
        """
        a, b, g = u_b
        sa, ca = np.sin(a), np.cos(a)
        sb, cb = np.sin(b), np.cos(b)
        sg, cg = np.sin(g), np.cos(g)

        # 原始旋转矩阵 R
        # R = [ [cb*cg,           cb*sg,           -sb],
        #       [sb*sa*cg-ca*sg,  sb*sa*sg+ca*cg,  cb*sa],
        #       [ca*sb*cg+sa*sg,  ca*sb*sg-sa*cg,  ca*cb] ]
        R = np.array([
            [cb * cg, cb * sg, -sb],
            [sb * sa * cg - ca * sg, sb * sa * sg + ca * cg, cb * sa],
            [ca * sb * cg + sa * sg, ca * sb * sg - sa * cg, ca * cb]
        ])

        # 1. 对 Alpha (a) 求导
        # R 的第 0 行没有 a，导数为 0
        dR_da = np.array([
            [0, 0, 0],
            [sb * ca * cg + sa * sg, sb * ca * sg - sa * cg, cb * ca],
            [-sa * sb * cg + ca * sg, -sa * sb * sg - ca * cg, -sa * cb]
        ])

        # 2. 对 Beta (b) 求导
        dR_db = np.array([
            [-sb * cg, -sb * sg, -cb],
            [cb * sa * cg, cb * sa * sg, -sb * sa],
            [ca * cb * cg, ca * cb * sg, -ca * sb]
        ])

        # 3. 对 Gamma (g) 求导
        dR_dg = np.array([
            [-cb * sg, cb * cg, 0],
            [-sb * sa * sg - ca * cg, sb * sa * cg - ca * sg, 0],
            [-ca * sb * sg + sa * cg, ca * sb * cg + sa * sg, 0]
        ])

        return R, dR_da, dR_db, dR_dg

    # ------------------------------------------------------------------
    # SCA 系数计算 (保持简单能量求和)
    # ------------------------------------------------------------------
    def _get_sca_coeffs(self, U, t):
        # 简化计算用于获取系数
        target_pos = np.array([self.R * np.cos(self.theta0), self.R * np.sin(self.theta0) - self.V * t, self.H])
        pc_list = np.zeros(self.B)
        U_mesh = U.reshape(self.B, 3)

        for b in range(self.B):
            vec = target_pos - self.Q[b]
            dist = max(np.linalg.norm(vec), 1.0)

            # 简化的增益计算 (用于系数估算)
            R_mat, _, _, _ = self._calc_R_and_Derivs(U_mesh[b])
            v_loc = np.dot(R_mat.T, vec / dist)
            theta = np.arccos(np.clip(v_loc[2], -1, 1))
            phi = np.arctan2(v_loc[1], v_loc[0])

            # Gain dB
            val_p = 12 * (np.abs(phi) / self.PHI_3DB) ** 2
            val_t = 12 * (np.abs(theta) / self.THETA_3DB) ** 2
            G_db = self.G_MAX - min(val_p + val_t, self.G_S)  # 简化逻辑
            Gain = 10 ** (G_db / 10.0)

            pc_list[b] = self.PC * Gain * (dist ** -self.EPSILON) * self.N_ELEM

        snr = pc_list / self.SIGMA_SQ
        ac = (1.0 / self.SIGMA_SQ) / (self.ln2 * (1 + snr))
        bc = np.log2(1 + snr) - ac * pc_list
        return ac, bc

    # ------------------------------------------------------------------
    # 目标函数与梯度 (核心加速部分)
    # ------------------------------------------------------------------
    def _objective_and_gradient(self, u_flat, U_ref, sca_params, mu):
        """
        计算 Loss 和 Gradient (利用链式法则)
        """
        U_curr = U_ref + u_flat.reshape(self.B, 3)  # delta_u
        grad_u = np.zeros((self.B, 3))  # (alpha, beta, gamma)
        total_loss = 0.0

        # 预计算时间点
        t_list = np.linspace(0, 2 * self.R / self.V, 5)

        for idx, t in enumerate(t_list):
            ac_list, bc_list = sca_params[idx]
            target_pos = np.array([self.R * np.cos(self.theta0), self.R * np.sin(self.theta0) - self.V * t, self.H])

            for b in range(self.B):
                # 1. 几何准备
                vec_global = target_pos - self.Q[b]
                dist = np.linalg.norm(vec_global)
                vec_norm = vec_global / max(dist, 1.0)  # 全局方向向量

                # 获取 R 及其导数
                R, dR_da, dR_db, dR_dg = self._calc_R_and_Derivs(U_curr[b])

                # 转局部坐标: v' = R^T * v
                v_loc = np.dot(R.T, vec_norm)  # (x', y', z')
                x_p, y_p, z_p = v_loc

                # 2. 计算增益 G 和 dG/d_angles
                # 角度
                # clip z_p 防止梯度爆炸
                z_p_safe = np.clip(z_p, -0.9999, 0.9999)
                theta = np.arccos(z_p_safe)
                phi = np.arctan2(y_p, x_p)

                # Gain dB 部分的导数
                # G_db = G_max - 12(phi/Phi_3db)^2 - 12(theta/Theta_3db)^2
                # d(G_db)/d(theta) = -24 * theta / Theta_3db^2
                dGdb_dtheta = -24 * theta / (self.THETA_3DB ** 2)
                dGdb_dphi = -24 * phi / (self.PHI_3DB ** 2)

                # 链式: dG/d... = G * ln10/10 * dGdb/d...
                G_db = self.G_MAX - (12 * (theta / self.THETA_3DB) ** 2 + 12 * (phi / self.PHI_3DB) ** 2)
                G = 10 ** (G_db / 10.0)
                G_coef = G * np.log(10) / 10.0

                dG_dtheta = G_coef * dGdb_dtheta
                dG_dphi = G_coef * dGdb_dphi

                # 3. 角度对局部坐标的导数
                # d(theta)/d(z') = -1 / sqrt(1 - z'^2)
                dtheta_dzp = -1.0 / np.sqrt(1 - z_p_safe ** 2)

                # d(phi)/d(x') = -y' / (x'^2 + y'^2)
                # d(phi)/d(y') = x' / (x'^2 + y'^2)
                r_xy_sq = x_p ** 2 + y_p ** 2
                if r_xy_sq < 1e-6: r_xy_sq = 1e-6
                dphi_dxp = -y_p / r_xy_sq
                dphi_dyp = x_p / r_xy_sq

                # 4. 局部坐标对欧拉角的导数
                # v' = R^T * v_global
                # dv'/da = d(R^T)/da * v_global = (dR_da)^T * v_global
                dv_da = np.dot(dR_da.T, vec_norm)
                dv_db = np.dot(dR_db.T, vec_norm)
                dv_dg = np.dot(dR_dg.T, vec_norm)

                # 5. 总链式法则 (G 对 alpha 的导数)
                # dG/da = (dG/dtheta * dtheta/dz' * dz'/da) + (dG/dphi * (dphi/dx'*dx'/da + dphi/dy'*dy'/da))

                # Alpha
                dG_da = (dG_dtheta * dtheta_dzp * dv_da[2]) + \
                        (dG_dphi * (dphi_dxp * dv_da[0] + dphi_dyp * dv_da[1]))

                # Beta
                dG_db = (dG_dtheta * dtheta_dzp * dv_db[2]) + \
                        (dG_dphi * (dphi_dxp * dv_db[0] + dphi_dyp * dv_db[1]))

                # Gamma
                dG_dg = (dG_dtheta * dtheta_dzp * dv_dg[2]) + \
                        (dG_dphi * (dphi_dxp * dv_dg[0] + dphi_dyp * dv_dg[1]))

                # 6. 最终 Power 的梯度
                # P = Const * G
                # dP/du = Const * dG/du
                const_P = self.PC * (dist ** -self.EPSILON) * self.N_ELEM
                dP_du = const_P * np.array([dG_da, dG_db, dG_dg])

                # 7. 累加到 SCA Loss 梯度
                # Loss = - sum(ac * P + bc)
                # Grad = - ac * dP/du
                current_ac = ac_list[b]
                grad_u[b] += -current_ac * dP_du

                # 累加 Loss
                total_loss -= (current_ac * const_P * G + bc_list[b])

        # 平均化速率 Loss 和 Grad
        total_loss /= len(t_list)
        grad_u /= len(t_list)

        # --- Part 2: 罚函数梯度 (朝向约束) ---
        # 约束: -n^T q <= 0
        # n 是 R 的第三列 R[:, 2]
        # n = [ -sb, cb*sa, ca*cb ]^T
        for b in range(self.B):
            R, dR_da, dR_db, dR_dg = self._calc_R_and_Derivs(U_curr[b])
            n_curr = R[:, 2]

            # 违规值 g(x)
            g_val = -np.dot(n_curr, self.Q[b])

            if g_val > 0:  # 有违规
                # Penalty = mu * (g_val)^2
                total_loss += mu * (g_val ** 2)

                # Grad = 2 * mu * g_val * d(g_val)/du
                # d(g_val)/du = - d(n^T)/du * q = - q^T * dn/du
                dn_da = dR_da[:, 2]
                dn_db = dR_db[:, 2]
                dn_dg = dR_dg[:, 2]

                dg_da = -np.dot(dn_da, self.Q[b])
                dg_db = -np.dot(dn_db, self.Q[b])
                dg_dg = -np.dot(dn_dg, self.Q[b])

                penalty_grad = 2 * mu * g_val * np.array([dg_da, dg_db, dg_dg])
                grad_u[b] += penalty_grad

        return total_loss, grad_u.flatten()

    def optimize(self, u_init):
        print(f"--- 启动极速 SCP-U 优化 (Analytical Gradient) ---")
        U_curr = u_init.copy()
        mu = self.mu_init

        for k in range(self.sca_loops):
            # SCA 参数
            sca_params = [self._get_sca_coeffs(U_curr, t) for t in np.linspace(0, 2 * self.R / self.V, 5)]

            # 信赖域
            delta_u_0 = np.zeros(self.B * 3)
            bounds = Bounds([-self.trust_region] * self.B * 3, [self.trust_region] * self.B * 3)

            res = minimize(
                fun=self._objective_and_gradient,
                x0=delta_u_0,
                args=(U_curr, sca_params, mu),
                method='L-BFGS-B',
                jac=True,  # <--- 开启解析梯度
                bounds=bounds,
                options={'ftol': 1e-4, 'disp': False}
            )

            delta = res.x.reshape(self.B, 3)
            U_curr += delta
            mu *= self.mu_growth

            step = np.linalg.norm(delta)
            if k % 5 == 0:
                print(f"   Iter {k}: Loss={res.fun:.4f}, Step={step:.4f}")

            if step < 1e-3: break

        return U_curr % (2 * np.pi)