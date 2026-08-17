import numpy as np
from scipy.optimize import minimize, Bounds
from itertools import combinations
H = 100.0;
R = 200.0;
V = 20.0
B = 16
C = 1  # 空间范围
LAMBDA = 0.125
N_ELEM = 4
D_NT = LAMBDA / 2
L_FRAME = 1
D_MIN = (np.sqrt(2) * LAMBDA + LAMBDA) / 2
THETA_0 = np.pi / 6;
# 功率与噪声
PC = 0.04;
PS = 0.04
SIGMA = -50
EPSILON = 2;
EPSILON1 = 4;
RHO = 0.8

# 增益参数
THETA_3DB = np.deg2rad(65);
PHI_3DB = np.deg2rad(65)
G_MAX = 8
G_S = 25;
G_V = 25


class SurfaceLayoutGenerator:
    def __init__(self, B, C, theta_0, H,R,V,lambda_val=0.125):
        """
        初始化生成器
        :param B: 表面数量 (int)
        :param C: 空间坐标限制 [-C, C] (float)
        :param theta_0: 车辆初始角度 (float, 虽然不直接约束表面位置，但作为环境参数传入)
        :param lambda_val: 波长，用于计算最小距离 d_min (默认 0.125)
        """
        self.B = B
        self.C = C
        self.theta_0 = theta_0
        self.H=H
        self.R=R
        self.V=V
        self.lambda_val = lambda_val

        # 根据公式计算最小距离 d_min
        # d_min = sqrt(2)/lambda + lambda/2
        self.d_min =( np.sqrt(2) * self.lambda_val + self.lambda_val) / 2

        # 优化器配置
        self.max_attempts = 20  # 最大随机重启次数
        self.opt_max_iter = 500  # 单次优化的最大迭代步数

    def _get_rotation_matrix(self, u_b):
        """计算旋转矩阵 R(u)"""
        alpha, beta, gamma = u_b
        sa, ca = np.sin(alpha), np.cos(alpha)
        sb, cb = np.sin(beta), np.cos(beta)
        sg, cg = np.sin(gamma), np.cos(gamma)
        return np.array([
            [cb * cg, cb * sg, -sb],
            [sb * sa * cg - ca * sg, sb * sa * sg + ca * cg, cb * sa],
            [ca * sb * cg + sa * sg, ca * sb * sg - sa * cg, ca * cb]
        ])

    def _get_normal_vector(self, u_b):
        """获取法向量 n(u)"""
        return self._get_rotation_matrix(u_b)[:, 2]

    def _reshape_vars(self, x):
        """将一维优化向量重塑为 U 和 Q"""
        # x 的前 B*3 是 U (角度)，后 B*3 是 Q (位置)
        U = x[:self.B * 3].reshape(self.B, 3)
        Q = x[self.B * 3:].reshape(self.B, 3)
        return U, Q

    def _calculate_violation(self, x):
        """
        计算约束违反程度（损失函数）。
        目标是将此值优化为 0。
        """
        U, Q = self._reshape_vars(x)
        violation = 0.0

        # 预计算所有法向量
        normals = np.array([self._get_normal_vector(U[b]) for b in range(self.B)])

        # 1. 朝向约束: n_b^T * q_b >= 0
        # 如果 n^T * q < 0，则违反，加入惩罚
        for b in range(self.B):
            val = np.dot(normals[b], Q[b])
            if val < 0:
                violation += -val

        # 2. 边界约束: |q| <= C
        # 如果超出边界，加入惩罚
        if np.any(np.abs(Q) > self.C):
            violation += np.sum(np.maximum(0, np.abs(Q) - self.C))

        # 3. 成对约束 (最小距离 & 互不遮挡)
        for b, j in combinations(range(self.B), 2):
            diff_jb = Q[j] - Q[b]
            dist = np.linalg.norm(diff_jb)

            # (A) 最小距离约束: ||q_b - q_j|| >= d_min
            if dist < self.d_min:
                violation += (self.d_min - dist) * 10.0  # 加大权重优先满足距离

            # (B) 互不遮挡约束 1: n(u_b)^T (q_j - q_b) <= 0
            val1 = np.dot(normals[b], diff_jb)
            if val1 > 0:
                violation += val1

            # (C) 互不遮挡约束 2: n(u_j)^T (q_b - q_j) <= 0
            val2 = np.dot(normals[j], -diff_jb)
            if val2 > 0:
                violation += val2

        return violation

    def generate(self):
        """
        【修改版】生成完全随机但合规的初始 Q 和 U
        用于验证算法鲁棒性：每次运行结果都不同
        """
        print(f"--- SurfaceLayoutGenerator: 生成 {self.B} 个随机初始点 ---")

        # ==========================================
        # 1. 随机生成 Q (必须满足 d_min 约束)
        # ==========================================
        Q_init = np.zeros((self.B, 3))

        # 这是一个 "试错法" (Rejection Sampling)
        for i in range(self.B):
            valid_point_found = False
            attempts = 0

            # 尝试撒点，直到找到一个不拥挤的位置
            while not valid_point_found and attempts < 1000:
                # 在空间 [-C, C] 内随机生成 X 和 Z 坐标 (假设 y=0)
                px = np.random.uniform(-self.C, self.C)
                pz = np.random.uniform(-self.C, self.C)

                # 候选点
                candidate_q = np.array([px, 0.0, pz])

                # 检查与之前已生成的 i 个点的距离
                collision = False
                for j in range(i):
                    # 计算距离
                    dist = np.linalg.norm(candidate_q - Q_init[j])

                    # 如果距离小于 d_min，说明撞了，这位置不行
                    if dist < self.d_min:
                        collision = True
                        break

                # 如果没撞，就采用这个点
                if not collision:
                    Q_init[i] = candidate_q
                    valid_point_found = True

                attempts += 1

            # 如果尝试了1000次都没地方放 (空间太小或者运气太差)
            if not valid_point_found:
                print(f"警告: 第 {i} 个点无法找到随机空位，强制放置在备用位置。")
                # 备用策略：按顺序排开，防止程序崩溃
                Q_init[i] = np.array([-self.C + i * 0.1, 0, -self.C])

        # ==========================================
        # 2. 随机生成 U (完全随机朝向)
        # ==========================================
        # 不再计算几何对准，而是生成 0 到 2pi 的随机角度
        # 这样初始速率会很低，正好用来测试优化器能不能把它救回来
        U_init = np.random.uniform(0, 2 * np.pi, (self.B, 3))

        return Q_init, U_init
# ==========================================
# 调用示例
# ==========================================

if __name__ == "__main__":
    # 定义输入参数
    B_in = 16
    C_in = 1.0  # 建议保持较大空间以容纳 d_min=11.37m
    Theta0_in = np.pi / 6

    # 1. 实例化生成器
    generator = SurfaceLayoutGenerator(B=B, C=C, theta_0=np.pi / 3,H=H,R=R,V=V)

    # 2. 生成数据
    q_result, u_result = generator.generate()

    # 3. 输出结果验证
    if q_result is not None:
        print("\n生成结果示例:")
        print(f"Q (位置) shape: {q_result.shape}")
        print(f"U (姿态) shape: {u_result.shape}")
        print("\n前 3 个表面的位置 Q:")
        print(q_result[:16])
        print("\n前 3 个表面的旋转姿态 U (Euler Angles):")
        print(u_result[:16])