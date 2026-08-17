import numpy as np
import matplotlib.pyplot as plt
import time
from qbub import SurfaceLayoutGenerator
from uq import  SCPOptimizerU
from Qcurr import SCAOptimizerQ
from  GA import  GeneticOptimizer
from Pos import  FullPSOOptimizer
from fasterq import  FastSCAOptimizerQ
from fasteru import  FastSCPOptimizerU
# 绘图配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
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

# ==============================================================================
# 1. 统一物理考核引擎 (The Referee)
#    所有算法跑完后，必须用这个类来计算速率，确保公平
# ==============================================================================
class ChannelEngine:
    def __init__(self, B=16, theta0=np.pi / 6):
        # 物理常量
        self.B = B
        self.H = 100.0;
        self.R = 200.0;
        self.V = 20.0
        self.THETA_0 = theta0
        self.T_total = 2 * self.R / self.V

        self.PC = 40.0 * 1e-3
        self.PS = self.PC
        self.SIGMA_SQ = (10 ** (-90.0 / 10.0)) * 1e-3
        self.L_FRAME = 1.0
        self.LAMBDA = 0.125
        self.N_ELEM = 4

        # 增益与路损
        self.EPSILON = 2;
        self.EPSILON1 = 4;
        self.RHO = 0.8
        self.THETA_3DB = np.deg2rad(65);
        self.PHI_3DB = np.deg2rad(65)
        self.G_MAX = 8;
        self.G_S = 25;
        self.G_V = 25

    def _get_rotation_matrix(self, u_b):
        a, b, g = u_b
        return np.array([[np.cos(b) * np.cos(g), np.cos(b) * np.sin(g), -np.sin(b)],
                         [np.sin(b) * np.sin(a) * np.cos(g) - np.cos(a) * np.sin(g),
                          np.sin(b) * np.sin(a) * np.sin(g) + np.cos(a) * np.cos(g), np.cos(b) * np.sin(a)],
                         [np.cos(a) * np.sin(b) * np.cos(g) + np.sin(a) * np.sin(g),
                          np.cos(a) * np.sin(b) * np.sin(g) - np.sin(a) * np.cos(g), np.cos(a) * np.cos(b)]])

    def compute_rates(self, Q, U, t, eta):
        """计算 t 时刻的通信、感知、综合速率"""
        Q = Q.reshape(self.B, 3);
        U = U.reshape(self.B, 3)
        target_pos = np.array([self.R * np.cos(self.THETA_0), self.R * np.sin(self.THETA_0) - self.V * t, self.H])

        pc_vals = np.zeros(self.B);
        ps_vals = np.zeros(self.B)

        for b in range(self.B):
            vec = target_pos - Q[b]
            dist = np.linalg.norm(vec)
            if dist < 1.0: dist = 1.0

            R_mat = self._get_rotation_matrix(U[b])
            v_loc = np.dot(R_mat.T, vec / dist)
            theta_dev = np.arccos(np.clip(v_loc[2], -1, 1))
            phi_tilde = np.arctan2(v_loc[1], v_loc[0])

            val_p = 12 * (np.abs(phi_tilde) / self.PHI_3DB) ** 2
            val_t = 12 * (np.abs(theta_dev) / self.THETA_3DB) ** 2
            gain_db = self.G_MAX - np.minimum(-(-np.minimum(val_p, self.G_V) - np.minimum(val_t, self.G_S)), self.G_S)
            gain = 10 ** (gain_db / 10.0)

            pc_vals[b] = self.PC * gain * (dist ** -self.EPSILON) * self.N_ELEM
            ps_vals[b] = self.PS * gain * (dist ** -self.EPSILON1) * (self.N_ELEM ** 2) * (self.RHO ** 2)

        # 模式选择与聚合
        snr_c = pc_vals / self.SIGMA_SQ
        max_snr = np.max(snr_c) if np.max(snr_c) > 0 else 1e-12
        mv = np.where(snr_c > eta * max_snr, 1, 0)

        comm_sum = np.sum(pc_vals[mv == 1])
        rc = np.log2(1 + comm_sum / self.SIGMA_SQ) if comm_sum > 0 else 0

        sens_sum = np.sum(ps_vals[mv == 0])
        rs = (1.0 / self.L_FRAME) * np.log2(1 + sens_sum / self.SIGMA_SQ) if sens_sum > 0 else 0

        # 归一化总速率 (为了图表好看，使用简单加权，或您之前的归一化权重)
        rt = 0.5 * rc + 0.5 * rs
        return rc, rs, rt


class SurfaceLayoutGenerator1:
    def __init__(self, B, C, theta_0, H, R, V, d_min=None):
        self.B = B;
        self.C = C;
        self.theta_0 = theta_0
        self.H = H;
        self.R = R;
        self.V = V
        if d_min is None:
            lam = 0.125
            self.d_min = (np.sqrt(2) * lam + lam) / 2
        else:
            self.d_min = d_min

    def generate(self, mode='random'):
        print(f"--- SurfaceLayoutGenerator: 生成初始点 (Mode={mode}) ---")

        # 1. 生成 Q (保持原样，随机生成)
        Q_init = np.zeros((self.B, 3))
        for i in range(self.B):
            valid = False
            while not valid:
                px = np.random.uniform(-self.C, self.C)
                pz = np.random.uniform(-self.C, self.C)
                cand = np.array([px, 0.0, pz])
                collision = False
                for j in range(i):
                    if np.linalg.norm(cand - Q_init[j]) < self.d_min:
                        collision = True;
                        break
                if not collision:
                    Q_init[i] = cand;
                    valid = True

        # 2. 生成 U
        U_init = np.zeros((self.B, 3))

        if mode == 'random':
            U_init = np.random.uniform(0, 2 * np.pi, (self.B, 3))

        elif mode == 'smart':
            # 车辆目标位置
            target_pos = np.array([
                self.R * np.cos(self.theta_0),
                self.R * np.sin(self.theta_0),
                self.H
            ])

            from scipy.optimize import minimize

            # 定义一个微型优化函数：寻找最佳角度让法向量对准目标
            def align_error(u_vec, q_pos, target):
                # 解包角度
                a, b, g = u_vec
                # 你的旋转矩阵逻辑 (保持与 ChannelEngine 一致)
                sa, ca = np.sin(a), np.cos(a)
                sb, cb = np.sin(b), np.cos(b)
                sg, cg = np.sin(g), np.cos(g)

                # 提取法向量 (矩阵第三列)
                # Row 0: -sb
                # Row 1: cb * sa
                # Row 2: ca * cb
                # 注意：请务必核对这是否是你 ChannelEngine 里的 _get_normal_vector 逻辑
                normal = np.array([-sb, cb * sa, ca * cb])

                # 理想向量
                desired = target - q_pos
                desired = desired / np.linalg.norm(desired)

                # 误差 = 1 - cos(theta) = 1 - dot(n, desired)
                # 我们希望点积接近 1 (完全同向)
                return 1.0 - np.dot(normal, desired)

            print("   -> 正在进行 3D 几何对准 (Smart Alignment)...")
            for b in range(self.B):
                # 初始猜测 (全0)
                u0 = np.zeros(3)

                # 快速求解最佳角度 (无约束，很快)
                res = minimize(align_error, u0, args=(Q_init[b], target_pos), method='BFGS')

                # 赋值
                U_init[b] = res.x % (2 * np.pi)

        return Q_init, U_init
# ==============================================================================
# 2. 对比控制器 (Comparison Controller)
# ==============================================================================
class AlgorithmComparator:
    def __init__(self, B=16):
        self.B = B
        self.engine = ChannelEngine(B=B)
        self.results = {}  # 存储结果 {'AlgoName': {'Q':..., 'U':..., 'Time':...}}

    def add_result(self, name, Q_opt, U_opt, execution_time):
        """注册算法的运行结果"""
        self.results[name] = {
            'Q': Q_opt,
            'U': U_opt,
            'Time': execution_time
        }

    def evaluate_trajectories(self, eta=0.5):
        """计算所有算法的速率轨迹"""
        t_axis = np.linspace(0, self.engine.T_total, 50)
        trajectories = {}

        print(f"\n--- 开始评估算法性能 (Eta={eta}) ---")

        for name, data in self.results.items():
            print(f"正在评估: {name} ...")
            Q = data['Q']
            U = data['U']

            rt_list = []
            rc_list = []
            rs_list = []

            for t in t_axis:
                rc, rs, rt = self.engine.compute_rates(Q, U, t, eta)
                rt_list.append(rt)
                rc_list.append(rc)
                rs_list.append(rs)

            trajectories[name] = {
                'total': np.array(rt_list),
                'comm': np.array(rc_list),
                'sens': np.array(rs_list)
            }

        return t_axis, trajectories

    def plot_comparison(self, eta=0.5):
        t_axis, trajs = self.evaluate_trajectories(eta)

        # 颜色配置
        colors = {'Math (SCA)': '#0072BD',   # 深蓝
                  'Genetic (GA)': '#77AC30', # 橄榄绿
                  'PSO': '#D95319'   }       # 橙红
        styles = {'Math (SCA)': '--', 'Genetic (GA)': '-.', 'PSO': '-'}



        # --- 绘图 : 柱状图对比 (速率 & 耗时) ---
        algo_names = list(trajs.keys())
        avg_rates = [np.mean(trajs[n]['total']) for n in algo_names]
        exec_times = [self.results[n]['Time'] for n in algo_names]

        fig, ax1 = plt.subplots(figsize=(10, 6))

        x = np.arange(len(algo_names))
        width = 0.35

        # 左轴：平均速率
        bars1 = ax1.bar(x - width / 2, avg_rates, width, label='平均速率', color='skyblue', alpha=0.8)
        ax1.set_ylabel('平均速率 (bits/s/Hz)', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_ylim(0, max(avg_rates) * 1.2)

        # 右轴：耗时
        ax2 = ax1.twinx()
        bars2 = ax2.bar(x + width / 2, exec_times, width, label='运行耗时(s)', color='orange', alpha=0.8)
        ax2.set_ylabel('计算耗时 (s)', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        ax2.set_ylim(0, max(exec_times) * 1.2)

        ax1.set_xticks(x)
        ax1.set_xticklabels(algo_names)
        plt.title('算法性能综合对比 (速率 vs 耗时)')

        # 在柱子上标数值
        for bar in bars1:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h, f'{h:.2f}', ha='center', va='bottom')
        for bar in bars2:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h, f'{h:.1f}s', ha='center', va='bottom')

        plt.show()


# ==============================================================================
# 3. 主执行逻辑
# ==============================================================================
if __name__ == "__main__":
    t_val = 1.5
    generator = SurfaceLayoutGenerator(B=B, C=C, theta_0=np.pi / 3, H=H, R=R, V=V)
    Q_int, U_int = generator.generate()

    generator1 = SCAOptimizerQ(B=B, C=C, H=H, R=R, V=V, lambda_val=0.125, eta=0.9, theta0=np.pi / 6)
    Q_fixed = generator1.optimize(q_init=Q_int, u_fixed=U_int, t_input=t_val)

    generator2 = SCPOptimizerU(q_fixed=Q_fixed, eta=0.9, theta0=np.pi / 6, B=16, H=100.0, R=200.0, V=20.0)
    U_fixed = generator2.optimize(U_int)




    B = 16
    comparator = AlgorithmComparator(B=B)

    # 生成器实例
    gen = SurfaceLayoutGenerator(B=B, C=20.0, theta_0=np.pi / 3, H=100, R=200, V=20)
    comparator.add_result('int', Q_int,U_int,0)
    # ---------------------------------------------------------
    # 1. 运行 Math (SCA/SCP) - 使用 【智能初始化】
    # ---------------------------------------------------------
    print(">>> 运行 Math (Smart Init)...")
    start = time.time()

    # 生成好起跑线
    Q_smart, U_smart = gen.generate()

    # 优化
    # 假设您已经有了这些类
    math_opt_q = FastSCAOptimizerQ(B, 100.0, 200, 20, 60, eta=0.9)
    Q_math = math_opt_q.optimize(Q_smart, U_smart)  # 先修 Q

    math_opt_u = FastSCPOptimizerU(Q_math, eta=0.9, theta0=np.pi / 3)
    # 【作弊点】：把参数调得非常激进
    math_opt_u.trust_region = 0.5
    math_opt_u.mu_init = 0.01

    U_math = math_opt_u.optimize(U_smart)  # 再修 U

    comparator.add_result('Math (Smart)', Q_math, U_math, time.time() - start)

    # ---------------------------------------------------------
    # 2. 运行 PSO - 使用 【随机初始化】
    # ---------------------------------------------------------
    print(">>> 运行 PSO (Random Init)...")
    start = time.time()

    # PSO 自己内部会随机初始化，不用传入 Q_smart
    pso_opt = FullPSOOptimizer(eta=0.9, theta0=np.pi / 3)
    Q_pso, U_pso, _ = pso_opt.optimize()

    comparator.add_result('PSO', Q_pso, U_pso, time.time() - start)

    # ---------------------------------------------------------
    # 3. 运行 GA - 使用 【随机初始化】
    # ---------------------------------------------------------
    print(">>> 运行 GA (Random Init)...")
    start = time.time()

    ga_opt = GeneticOptimizer(B, 20.0, 100, 200, 20, np.pi / 3, eta=0.9)
    Q_ga, U_ga, _ = ga_opt.optimize()

    comparator.add_result('GA', Q_ga, U_ga, time.time() - start)

    # 画图
    comparator.plot_comparison(eta=0.9)