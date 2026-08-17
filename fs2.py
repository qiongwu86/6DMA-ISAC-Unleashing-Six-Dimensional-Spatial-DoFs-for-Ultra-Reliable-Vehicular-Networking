import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'   # 使用 STIX 数学字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
matplotlib.rcParams['axes.unicode_minus'] = False   # 正常显示负号


# ========== 参数（按你要求） ==========
R = 200.0        # 车辆轨迹到基站横向固定距离 y = R (m)
v = 60.0         # 车辆速度 (m/s)
h = 100.0        # 基站高度 (m)
L = 200.0        # 半路径长度，整个路径为 2*L = 400 m
delta_deg = 1.0  # 允许角度变化 (度)
delta_rad = np.deg2rad(delta_deg)

# ========== 时间设置 ==========
T = (2.0 * L) / v               # 总时间 = 路径长度 / 速度 = 400/60 ≈ 6.6667 s
t = np.linspace(0.0, T, 5000)   # 时间采样点（用于绘图）

# ========== 车辆轨迹（从 x=-L 到 x=+L） ==========
x = -L + v * t   # x 随时间线性增长
y = R            # y 恒定
r = np.sqrt(x**2 + y**2)

# ========== 角度（定义标准：az = atan2(y,x)） ==========
az = np.arctan2(y, x)        # 方位角（rad）
el = np.arctan2(h, r)        # 俯仰角（rad）

# ========== 解析角速度表达式（避免数值微分误差） ==========
# az_dot = (x * dy/dt - y * dx/dt) / (x^2 + y^2) ; dy/dt = 0, dx/dt = v
az_dot = - (y * v) / (x**2 + y**2)

# el_dot = - h * v * x / ( r * (r^2 + h^2) )
el_dot = - (h * v * x) / ( r * (r**2 + h**2) )

# 取绝对值以便展示对称性
abs_az_dot = np.abs(az_dot)
abs_el_dot = np.abs(el_dot)

# ========== 计算最大值与推荐采样率 ==========
i_az = np.argmax(abs_az_dot)
i_el = np.argmax(abs_el_dot)
t_az_max = t[i_az]
t_el_max = t[i_el]
Omega_az_max = abs_az_dot[i_az]
Omega_el_max = abs_el_dot[i_el]

fs_az = Omega_az_max / delta_rad   # 按 Δθ 限制得到的采样率 (Hz)
fs_el = Omega_el_max / delta_rad
fs_recommend = max(fs_az, fs_el)

# ========== 绘图 ==========
plt.figure(figsize=(10,6))
plt.plot(t, abs_az_dot, label=r'|$\dot{\psi}(t)$|  (rad/s)', lw=1.6)
plt.plot(t, abs_el_dot, label=r'|$\dot{\phi}(t)$|  (rad/s)', lw=1.6)

# 标注最大点与穿越时刻
t_cross = L / v
plt.scatter([t_az_max], [Omega_az_max], color='C0', zorder=6)
plt.annotate(f"dψ/dtmax={Omega_az_max:.4f} rad/s\n t={t_az_max:.3f}s",
             xy=(t_az_max, Omega_az_max), xytext=(t_az_max+0.15, Omega_az_max*1.05),
             arrowprops=dict(arrowstyle="->", color='C0'))

plt.scatter([t_el_max], [Omega_el_max], color='C1', zorder=6)
plt.annotate(f"dφ/dtmax={Omega_el_max:.4f} rad/s\n t={t_el_max:.3f}s",
             xy=(t_el_max, Omega_el_max), xytext=(t_el_max+0.15, Omega_el_max*1.05),
             arrowprops=dict(arrowstyle="->", color='C1'))

plt.axvline(t_cross, color='gray', linestyle='--', linewidth=0.9)
#plt.text(t_cross, plt.ylim()[1]*0.9, ' x=0 (穿越点)', rotation=90, va='top', ha='right', color='gray')

plt.xlabel('时间t(s)')
plt.ylabel('导数值(rad/s)')
#plt.title(f'(R=200 m, v={v} m/s, R={R} m, h={h} m)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ========== 打印数值结果 ==========
print(f"总时间 T = {T:.6f} s (R= 400 m, 速度 {v} m/s)")
print(f"穿越点时间 t_cross = {t_cross:.6f} s")
print(f"方位角最大角速度 Ω_ψ_max = {Omega_az_max:.6f} rad/s = {np.rad2deg(Omega_az_max):.3f} °/s  (t={t_az_max:.6f}s)")
print(f"俯仰角最大角速度 Ω_φ_max = {Omega_el_max:.6f} rad/s = {np.rad2deg(Omega_el_max):.3f} °/s  (t={t_el_max:.6f}s)")
print(f"按 Δθ = {delta_deg}° 限制, 推荐采样率 f_s >= {fs_recommend:.3f} Hz  (ψ:{fs_az:.3f} Hz, φ:{fs_el:.3f} Hz)")
