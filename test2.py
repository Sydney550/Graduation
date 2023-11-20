import CNN_1D as cnn
from PyEMD import CEEMDAN, Visualisation, EEMD, EMD
from vmdpy import VMD
import matplotlib.pyplot as plt
import numpy as np
import sys
from PyLMD import LMD
from scipy.stats import pearsonr
from scipy import stats

path = 'vib_signals'
dynamic = cnn.load_files(path).transpose(0, 2, 1)
datas = dynamic[0]
print(datas.shape)
fs = 10000
# ceemdan = EEMD()
# ceemdan.noise_seed(22)
# for S in datas:
#     ceemdan.eemd(S)
#     imfs, res = ceemdan.get_imfs_and_residue()
#     vis = Visualisation()
#     vis.plot_imfs(imfs=imfs, residue=res, include_residue=True)
#     # vis.plot_instant_freq(t, imfs=imfs)
#     vis.show()

# def ICEEMDAN(signal, num_sifts=10, num_trials=100):
#     emd = EMD()
#     emd.extrema_detection = "parabol"
#     imfs = []
#     residue = signal
#     while len(imfs) < num_sifts:
#         imf_trials = []
#         for _ in range(num_trials):
#             std = np.std(residue)
#             noise = np.random.normal(size=len(residue), scale=std)
#             noisy_signal = residue + noise
#             imf = emd.emd(noisy_signal, max_imf=1)[0]
#             imf_trials.append(imf)
#         imf_trials = np.array(imf_trials)
#         imf_final = imf_trials.mean(axis=0)
#         imfs.append(imf_final)
#         residue = residue - imf_final
#     return np.array(imfs)
# # 使用ICEEMDAN进行分解
# imfs = ICEEMDAN(datas[0])
# # 显示分解的结果
# for i, imf in enumerate(imfs):
#     plt.figure(figsize=(12, 9))
#     plt.subplot(len(imfs), 1, i+1)
#     plt.plot(imf)
#     plt.title("IMF {}".format(i+1))
# plt.show()


x = range(datas.shape[1])
for y in datas:
    # y = datas[0]
    lmd = LMD()
    PFs, res = lmd.lmd(y)
    print(PFs.shape)
    for i, pf in enumerate(PFs):
        print(f'PF{i}')
        corr, pvalue = pearsonr(y, pf)
        kur = stats.kurtosis(pf)
        print(f'相关系数 {corr}  p值 {pvalue}  峭度 {kur}')
    print(f'res')
    cor_res, p_res = pearsonr(y, res)
    kur_res = stats.kurtosis(res)
    print(f'相关系数 {cor_res}  p值 {p_res}  峭度 {kur_res}')
    # plotnum = PFs.shape[0] + 2
    # plt.figure(figsize=(12, 12))
    # plt.subplot(plotnum, 1, 1)
    # plt.title('original signal')
    # plt.plot(x, y)
    # for i, pf in enumerate(PFs):
    #     plt.subplot(plotnum, 1, i+2)
    #     plt.title(f'PF{i+1}')
    #     plt.plot(x, pf)
    # plt.subplot(plotnum, 1, plotnum)
    # plt.title('residue')
    # plt.plot(x, res)
    # plt.subplots_adjust(hspace=0.5)
    # plt.tight_layout()
    # plt.show()


# alpha = 2000
# K = 13
# tau = 0.            # noise-tolerance (no strict fidelity enforcement)
# DC = 0             # no DC part imposed
# init = 1           # initialize omegas uniformly
# tol = 1e-7
# for K in range(2,18):
#     u, u_hat, omega = VMD(datas[0], alpha, tau, K, DC, init, tol)
#     print(omega[-1])
# # plt.figure()
# # plt.scatter(np.array(range(2, 16)), np.array(omg_list))
# # plt.show()


# for i,S in enumerate(datas):
#     u, u_hat, omega = VMD(S, alpha, tau, K[i], DC, init, tol)
#     # 显示分解结果
#     t = range(u.shape[1])
#     plt.figure()
#     for j in range(K[i]):
#         plt.subplot(K[i],1,j+1)
#         plt.plot(t, u[j])
#         plt.title(f'IMF {j+1}')
#     # plt.tight_layout()
#     plt.show()
# for i,S in enumerate(datas):
#     u, u_hat, omega = VMD(S, alpha, tau, K, DC, init, tol)
#     # 显示分解结果
#     t = range(u.shape[1])
#     plt.figure(figsize=(8,28))
#     for j in range(K):
#         plt.subplot(K,1,j+1)
#         plt.plot(t, u[j])
#         plt.title(f'IMF {j+1}')
#     # plt.tight_layout()
#     plt.show()
