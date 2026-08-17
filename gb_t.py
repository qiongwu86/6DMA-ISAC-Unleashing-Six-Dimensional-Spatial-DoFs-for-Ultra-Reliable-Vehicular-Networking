import numpy as np
import matplotlib.pyplot as plt
from math import atan, sqrt, cos, sin, acos, pi

# -------------------------- 固定参数（按题目给定） --------------------------
h = 100       # 高度（m）
R = 200       # 初始距离（m）
θ0 = pi / 6   # 初始角度（弧度，30°）
v = 20        # 速度（m/s）

# 旋转矩阵参数（αb=βb=γb=30°，转换为弧度）
αb = βb = γb = pi / 6
# 计算正余弦值（s=sin，c=cos；角标b省略简化）
sα, cα = sin(αb), cos(αb)  # sin(30°)=0.5, cos(30°)=√3/2≈0.8660
sβ, cβ = sin(βb), cos(βb)
sγ, cγ = sin(γb), cos(γb)

# 增益参数
G_max = 8       # dBi
G_s = 25        # dB
G_v = 25        # dB
θ3db = 65 * pi / 180  # 65°转弧度
φ3db = 65 * pi / 180  # 65°转弧度


# -------------------------- 核心函数（按公式实现） --------------------------
def θ_t(t):
    """计算θ(t)：随时间变化的角度（弧度）"""
    numerator = R * sin(θ0) - v * t
    denominator = R * cos(θ0)
    return atan(numerator / denominator)

def φ_t(t):
    """计算φ(t)：随时间变化的角度（弧度）"""
    term = sqrt((R * cos(θ0))**2 + (R * sin(θ0) - v * t)** 2)
    return atan(term / h)

def R_ub():
    """构建旋转矩阵R(ub)（严格按题目给定的矩阵元素）"""
    return np.array([
        [cβ * cγ,
         sβ * sα * cγ - cα * sγ,
         -sβ],
        [sβ * sα * cγ + cα * sγ,  # 修正第二行第二列（原公式第二行第二列应为该值）
         sβ * sα * sγ + cα * cγ,
         cβ * sα],
        [cα * sβ * cγ - sα * sγ,  # 修正第三行第一列（原公式第三行第一列应为该值）
         cα * sβ * sγ + sα * cγ,
         cα * cβ]
    ])

def f_t(t):
    """计算f(t)向量"""
    θ = θ_t(t)
    φ = φ_t(t)
    return np.array([
        cos(θ) * cos(φ),
        sin(θ) * cos(φ),
        sin(φ)
    ])

def xyz_b(t):
    """计算[ẋb, ẏb, żb]^T（载体坐标系分量）"""
    R = R_ub()
    f = f_t(t)
    return -np.dot(R.T, f)  # 等价于 -R(ub)^T · f(t)

def tilde_θ_φ(t):
    """计算θ̃b(t)和φ̃b(t)（载体坐标系下的角度）"""
    xb, yb, zb = xyz_b(t)
    # 计算θ̃b(t) = π/2 - arccos(zb)
    tilde_θ = pi/2 - acos(zb)
    # 计算φ̃b(t)（含象限修正）
    denom = sqrt(xb**2 + yb**2)
    if denom < 1e-10:  # 避免除零
        tilde_φ = 0.0
    else:
        cos_phi = xb / denom
        # 根据yb符号确定η值（象限修正）
        eta = 1 if yb >= 0 else -1
        tilde_φ = eta * acos(cos_phi)
    return tilde_θ, tilde_φ

def A_H(tilde_φ):
    """计算水平分量A_H(φ̃)"""
    term = (12 * (tilde_φ / φ3db))** 2
    return -min(term, G_v)

def A_V(tilde_θ):
    """计算垂直分量A_V(θ̃)"""
    term = (12 * (tilde_θ / θ3db))** 2
    return -min(term, G_v)

def A(tilde_θ, tilde_φ):
    """计算A(θ̃, φ̃)"""
    term = -A_H(tilde_φ) + A_V(tilde_θ)
    return G_max - min(term, G_s)

def gb(t):
    """计算增益函数gb(t)"""
    tilde_θ, tilde_φ = tilde_θ_φ(t)
    a = A(tilde_θ, tilde_φ)
    return 10 **(a / 10)


# -------------------------- 计算与绘图（前20秒） --------------------------
# 时间范围：0~20秒，取2000个点（保证曲线平滑）
t = np.linspace(0, 20, 2000)
gb_values = [gb(ti) for ti in t]

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 绘制图像
plt.figure(figsize=(12, 6))
plt.plot(t, gb_values, color='forestgreen', linewidth=2)
plt.xlabel('时间 t (秒)', fontsize=12)
plt.ylabel('$g_b(t)$ 值', fontsize=12)
plt.title('前20秒内 $g_b(t)$ 随时间变化曲线', fontsize=14)
plt.grid(alpha=0.3)
plt.xlim(0, 20)
plt.ylim(0, max(gb_values)*1.1)  # 从0开始显示更直观
plt.show()

# 输出关键时间点数值（验证用）
print("前20秒关键时间点的gb(t)值：")
for ti in [0, 5, 10, 15, 20]:
    print(f"t={ti}s: {gb(ti):.6f}")