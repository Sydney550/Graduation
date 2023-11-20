import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from vmdpy import VMD
import CNN_1D as cnn
from sko.PSO import PSO
from pyentrp import entropy as ent
import EntropyHub as EH


path = 'vib_signals'
dynamic = cnn.load_files(path).transpose(0, 2, 1)
data = dynamic[0][0]

tau = 0.            # noise-tolerance (no strict fidelity enforcement)
DC = 0             # no DC part imposed
init = 1           # initialize omegas uniformly
tol = 1e-7


# 计算每个IMF分量的包络熵
def calculate_entropy(imf):
    # 每个分量的包络信号为env
    env = np.abs(hilbert(imf))
    # 将每个包络信号归一化到 [0, 1] 区间内
    env_norm = env / np.max(env)  # 在计算包络熵的过程中，需要对包络信号进行归一化处理，以确保不同幅度的包络信号具有可比性。
    # 根据信息熵的定义，可以通过将包络信号的概率分布进行估计，
    # 并计算该概率分布的熵值来度量其不确定性。因此，在这段代码中，将归一化后的包络信号作为概率分布，通过p = env_norm / np.sum(env_norm)计算其概率分布。
    p = env_norm / np.sum(env_norm)
    return -np.sum(p * np.log2(p))


# 定义适应度函数，即最小包络熵
def fitness_func(x):
    alpha = int(round(x[0]))
    K = int(round(x[1]))
    u, u_hat, omega = VMD(data, alpha, tau, K, DC, init, tol)
    num_modes = u.shape[0]
    entro = []
    for i in range(num_modes):
        # entro.append(ent.sample_entropy(u[i], len(u[i])))
        entro.append(calculate_entropy(u[i]))
    # 找到最小的包络熵对应的模态
    print(entro)
    min_entropy = min(entro)
    # min_entropy_mode = u[min_entropy_index]

#     print("最小包络熵对应的模态：", min_entropy_index)
    # x为VMD参数向量
    # signal为要分解的信号
    # 分解信号并计算最小包络熵
    # 返回最小包络熵值
    return min_entropy

# def fitness_func(p):
#     # p is a vector of parameters [alpha, K]
#     alpha = int(round(p[0])) # Penalty factor
#     K = int(round(p[1])) # Number of modes
#     # Apply VMD to the signal
#     u, u_hat, omega = VMD(data, alpha, tau, K, DC, init, tol)
#     # Compute the sample entropy of the original signal
#     S_x = EH.SampEn(data, m=2, r=0.2*np.std(data))[0][-1]
#     # Compute the sample entropy of each mode
#     S_u = np.zeros(K)
#     for k in range(K):
#         S_u[k] = EH.SampEn(u[k], m=2, r=0.2*np.std(u[k]))[0][-1]
#     # Compute the fitness value as the sum of the absolute differences between the entropies
#     F = np.sum(np.abs(S_x - S_u))
#     return F

# 使用PSO算法优化VMD参数
pso = PSO(func=fitness_func, n_dim=2, pop=8, max_iter=40, lb=[1000, 2], ub=[3001, 16], w=1.2, c1=2.0, c2=2.0)
pso.run()
print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)

# 输出结果图
plt.plot(pso.gbest_y_hist)
plt.show()