
import numpy as np
import matplotlib.pyplot as plt
from math import pi, sqrt, log2, exp, cos, sin, atan, acos
# -------------------------- 固定参数 --------------------------
P_C = 40e-3  # 通信发射功率（W）
P_S = 40e-3  # 感知发射功率（W）
σ2 = 10 ** (-50 / 10) * 1e-3  # 噪声方差（W）
λ = 0.125  # 波长（m）
ε = 2  # 距离衰减系数
β = 0.8  # 反射系数
T = 20  # 总时间（s）
L = 10e-3  # 感知帧长度（10ms）
M = 5  # 限制条件数量
μ_k = 100  # 惩罚因子
c = 1  # 坐标限制（||q|| ≤ c/2）
d_min = 0.5  # 表面最小间距（m）
v = 20  # 目标速度（m/s）
R = 200

# -------------------------- 初始状态生成 --------------------------
def init_q():
    """生成满足限制条件的初始位置q（3维）"""
    while True:
        q = np.random.uniform(-c / 4, c / 4, 3)  # 确保初始在可行域
        if np.linalg.norm(q) <= c / 2:
            return q


def init_u():
    """生成初始旋转姿态u（欧拉角，弧度）"""
    return np.radians(np.random.uniform(-90, 90, 3))  # α, β, γ ∈ [-90°,90°]


def init_s():
    """生成初始状态s（0=通信，1=感知）"""
    return np.random.choice([0, 1])


# -------------------------- 基础函数（角度、距离、信道） --------------------------
def θ_t(t):
    """计算θ(t)（角度，弧度）"""

    θ0 = pi / 6
    return atan((R * sin(θ0) - v * t) / (R * cos(θ0)))


def φ_t(t):
    """计算φ(t)（角度，弧度）"""

    θ0 = pi / 6
    term = sqrt((R * cos(θ0)) ** 2 + (R * sin(θ0) - v * t) ** 2)
    return atan(term / 100)  # h=100m


def d_t(t, q):
    """计算t时刻距离d(t)"""

    θ0 = pi / 6
    term = sqrt((R * cos(θ0)) ** 2 + (R * sin(θ0) - v * t) ** 2)
    return sqrt(q[2] ** 2 + term ** 2)  # q[2]为高度


def h_c(t, q, u):
    """通信信道h_c(q,u,t)（修正复数问题）"""
    α, β_u, γ = u  # 旋转角（欧拉角）
    # 旋转矩阵（确保实数运算）
    sα, cα = sin(α), cos(α)
    sβ, cβ = sin(β_u), cos(β_u)
    sγ, cγ = sin(γ), cos(γ)
    R_ub = np.array([
        [cβ * cγ, sβ * sα * cγ - cα * sγ, -sβ],
        [sβ * sα * cγ + cα * sγ, sβ * sα * sγ + cα * cγ, cβ * sα],
        [cα * sβ * cγ - sα * sγ, cα * sβ * sγ + sα * cγ, cα * cβ]
    ])
    # f(t)向量（确保实数）
    θ = θ_t(t)
    φ = φ_t(t)
    f = np.array([
        cos(θ) * cos(φ),
        sin(θ) * cos(φ),
        sin(φ)
    ])
    # 载体坐标系分量（取实部，避免数值误差导致的复数）
    xyz = -np.dot(R_ub.T, f)
    xyz = np.real(xyz)  # 关键修复：强制转为实数

    # 增益（8dBi）
    g = 10 ** (8 / 10)
    # 距离衰减
    d = d_t(t, q)
    # 相位项（修正复数运算）

    # 计算复数的相位角（arctan2(虚部/实部)）
    phase_complex = np.exp(-1j * 2 * np.pi * d / λ)
    phase1 = np.angle(phase_complex)  # 得到实数相位角（范围：-π~π）
    # 多普勒相位（确保角度为实数）
    angle = pi / 2 - acos(np.clip(xyz[2], -1, 1))  # 限制输入范围，避免数值错误
    phase_complex = np.exp(1j * 2 * np.pi * (v / λ) * np.cos(angle) * t)
    phase2 = np.angle(phase_complex)  # 结果为实数（范围：-π ~ π）
    phase = phase1 * phase2

    # 信道最终值（取实部避免复数累积）
    h = sqrt(P_C) * (d ** (-ε)) * phase * sqrt(g) * xyz
    return np.real(h)  # 关键修复：返回实数


import numpy as np  # 确保导入numpy

def h_s(t, q, u):
    """感知信道h_s(q,u,t)"""
    h_c_val = h_c(t, q, u)
    d = d_t(t, q)
    # 1. 先计算复数相位（保留复数运算）
    phase_complex = np.exp(-1j * 2 * np.pi * d / λ) * np.exp(1j * 2 * np.pi * (v / λ) * t)
    # 2. 后续运算使用复数，最后转实数
    h = np.sqrt(P_S) * (d ** (-eJ)) * B * phase_complex * np.sqrt(10 ** (8 / 10)) * h_c_val
    return np.real(h)  # 最终转为实数


# -------------------------- 速率计算 --------------------------
def rate_c(t, q, u):
    """通信速率R_c(t)"""
    h = h_c(t, q, u)
    snr = (P_C * np.abs(h) ** 2) / σ2
    return log2(1 + snr) if snr > 0 else 0


def rate_s(t, q, u):
    """感知速率R_s(t)"""
    h = h_s(t, q, u)
    snr = (P_S * np.abs(h) ** 2) / σ2
    return log2(1 + snr) if snr > 0 else 0


# -------------------------- 限制条件 --------------------------
def gm(q, B):
    """综合限制条件g_m(q)（简化实现）"""
    g1 = 1 if all(np.dot(np.array([1, 0, 0]), (j - q)) > 0 for j in B if not np.array_equal(j, q)) else 0
    g2 = -1 if all(np.dot(np.array([0, 1, 0]), j) < 0 for j in B) else 0
    g3 = 1 if all(np.dot(np.array([0, 0, 1]), (q - j)) > 0 for j in B if not np.array_equal(j, q)) else 0
    g4 = 1 if np.linalg.norm(q) > c / 2 else 0
    g5 = 1 if all(np.linalg.norm(q - j) > d_min for j in B if not np.array_equal(j, q)) else 0
    return [g1, g2, g3, g4, g5]


# -------------------------- 损失函数 --------------------------
def f_tqs(t, q, u, s):
    """t时刻损失函数（负速率，越小越好）"""
    if s == 0:
        return -rate_c(t, q, u)
    else:
        return -rate_s(t, q, u)


def loss_function(q, u, s):
    """总损失函数（时间平均）"""
    t = np.linspace(0, T, 100)
    loss = 0.0
    for ti in t:
        loss += f_tqs(ti, q, u, s)
    return loss / len(t)


# -------------------------- 梯度计算 --------------------------
def grad_q(q, u, s):
    """位置q的梯度（数值梯度）"""
    eps = 1e-6
    grad = np.zeros_like(q, dtype=np.float64)
    for i in range(3):
        q_plus = q.copy()
        q_plus[i] += eps
        q_minus = q.copy()
        q_minus[i] -= eps
        # 计算损失函数差（确保实数）
        loss_plus = loss_function(q_plus, u, s)
        loss_minus = loss_function(q_minus, u, s)
        grad[i] = (loss_plus - loss_minus) / (2 * eps)
    return grad


def grad_u(u, q, s):
    """旋转姿态u的梯度（数值梯度）"""
    eps = 1e-6
    grad = np.zeros_like(u, dtype=np.float64)
    for i in range(3):
        u_plus = u.copy()
        u_plus[i] += eps
        u_minus = u.copy()
        u_minus[i] -= eps
        grad[i] = (loss_function(q, u_plus, s) - loss_function(q, u_minus, s)) / (2 * eps)
    return grad


# -------------------------- 优化函数 --------------------------
def optimize_q(u, s, max_iter=50, lr=0.5):
    """第一层优化：位置q"""
    q = init_q()
    B = [q + np.random.uniform(-0.1, 0.1, 3) for _ in range(2)]  # 其他表面
    for iter in range(max_iter):
        # 计算梯度
        g = grad_q(q, u, s)
        # 惩罚项
        gm_vals = gm(q, B)
        grad_penalty = μ_k * np.sum([gv for gv in gm_vals if gv > 0])  # 仅惩罚正的约束违反
        g += grad_penalty
        # 更新q
        q_new = q - lr * g
        # 强制约束（确保在可行域）
        if np.linalg.norm(q_new) <= c / 2:
            q = q_new
        # 学习率衰减
        if iter % 10 == 0:
            lr *= 0.9
    return q


def optimize_u(q, s, max_iter=50, lr=0.1):
    """第二层优化：旋转姿态u"""
    u = init_u()
    for iter in range(max_iter):
        g = grad_u(u, q, s)
        u_new = u - lr * g
        u = u_new
        if iter % 10 == 0:
            lr *= 0.9
    return u


def switch_s(q, u, t):
    """根据信噪比切换状态s"""
    snr_c = (P_C * np.abs(h_c(t, q, u)) ** 2) / σ2
    snr_s = (P_S * np.abs(h_s(t, q, u)) ** 2) / σ2
    return 0 if snr_c > snr_s else 1


# -------------------------- 主程序 --------------------------
if __name__ == "__main__":
    # 初始状态
    u_init = init_u()
    s_init = init_s()

    # 第一层优化：位置q
    q_opt = optimize_q(u_init, s_init)
    print("最佳位置q（第一层优化结果）：")
    print(f"q = [{q_opt[0]:.4f}, {q_opt[1]:.4f}, {q_opt[2]:.4f}]")

    # 第二层优化：旋转姿态u
    u_opt = optimize_u(q_opt, s_init)
    print("\n最佳旋转姿态u（弧度）：")
    print(f"u = [{u_opt[0]:.4f}, {u_opt[1]:.4f}, {u_opt[2]:.4f}]")

    # 切换状态s（t=10s时）
    s_opt = switch_s(q_opt, u_opt, t=10)
    print("\n最佳状态s（t=10s）：")
    print(f"s = {s_opt}（0=通信，1=感知）")

    # 计算最佳位置下的通感速率
    t_range = np.linspace(0, T, 200)
    rates = []
    for ti in t_range:
        s_t = switch_s(q_opt, u_opt, ti)
        if s_t == 0:
            rates.append(rate_c(ti, q_opt, u_opt))
        else:
            rates.append(rate_s(ti, q_opt, u_opt))

    # 绘制速率曲线
    plt.rcParams["font.family"] = ["SimHei", "Arial Unicode MS"]
    plt.figure(figsize=(10, 6))
    plt.plot(t_range, rates, color="crimson", linewidth=2)
    plt.xlabel("时间 t (秒)")
    plt.ylabel("通感速率 (bps/Hz)")
    plt.title("最佳位置q下的通感速率曲线")
    plt.grid(alpha=0.3)
    plt.show()

    # 平均速率
    print(f"\n平均通感速率：{np.mean(rates):.4f} bps/Hz")
