import numpy as np
import gpflow
from gpflow.models import GPR
from gpflow import kernels
import pandas as pd
from sklearn.model_selection import train_test_split
from fractions import Fraction
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import stats, signal
import CNN_1D as cnn
from pyentrp import entropy as ent
from PyLMD import LMD

# 获取时域或频域的统计特征
def get_signal_feature(data, domin='time'):
    """
    提取 N个 时域特征

    @param data: shape 为 (m, n) 的 2D array 数据，其中，m 为信号个数， n 为信号长度
    @return: shape 为 (m, N)  的 2D array 数据，其中，m 为信号个数。即 每个信号的N个时域特征
    """
    rows, cols = data.shape
    ax = -1

    # 有量纲统计量
    max_value = np.max(data, axis=ax)  # 最大值
    min_value = np.min(data, axis=ax)  # 最小值
    peak_value = np.max(abs(data), axis=ax)  # 最大绝对值
    mean = np.mean(data, axis=ax)  # 均值
    rms = np.sqrt(np.sum(data ** 2, axis=ax) / cols)  # 均方根值
    variance = np.var(data, axis=ax)  # 方差
    std = np.std(data, axis=ax)  # 标准差
    kurtosis = stats.kurtosis(data, axis=ax)  # 峭度
    skewness = stats.skew(data, axis=ax)  # 偏度
    p_p_value = max_value - min_value  # 峰峰值
    abs_mean = np.mean(abs(data), axis=ax)  # 绝对平均值（整流平均值）
    square_root_amplitude = (np.sum(np.sqrt(abs(data)), axis=ax) / cols) ** 2  # 方根幅值
    # mean_amplitude = np.sum(np.abs(data), axis=ax) / cols  # 平均幅值 == 绝对平均值

    # 无量纲统计量
    if domin == 'time':
        shape_factor = rms / abs_mean  # 波形因子
        crest_factor = peak_value / rms  # 峰值因子
        # clearance_factor = peak_value / square_root_amplitude  # 裕度因子
        # impulse_factor = peak_value / abs_mean  # 脉冲因子
        # kurtosis_factor = kurtosis / (rms**4)  # 峭度因子

        entropy = np.apply_along_axis(ent.shannon_entropy, axis=ax, arr=data)  # 熵
        features = [mean, rms, variance, std, kurtosis, skewness, entropy, p_p_value,
                    shape_factor, crest_factor]
    elif domin == 'freq':
        features = [mean, rms, variance, std, kurtosis, skewness, p_p_value]
    else:
        raise Exception('domin value error!')

    return np.array(features).T


# 相关性分析
def pearson_func(x_arr, y_arr, threshold=0.3):
    """
    通过皮尔逊相关系数筛选特征

    :param x_arr: shape为(m,n)的特征数组，其中m为样本数，n为特征个数
    :param y_arr: shape为(m,1)的目标数组，其中m为样本数
    :param threshold:
    :return: 经过相关系数筛选后的新数组，shape为(m,k)，其中m为样本数，k为筛选后的特征个数
    """
    # 初始化一个空的列表来保存每个特征的皮尔逊相关系数
    pearson_correlations = []
    # 对于X中的每个特征
    for i in range(x_arr.shape[1]):
        feature = x_arr[:, i]
        # 计算该特征与Y的相关系数
        # print(f'feature{i}')
        corr, pvalue = stats.pearsonr(feature, y_arr[:, 0])
        # pvalue = round(pvalue, 5)
        # print(f'相关系数 {corr}  p值 {pvalue}')
        correlations = [corr]
        # if pvalue > 0.05:
        pearson_correlations.append(correlations)
    pearson_correlations = np.array(pearson_correlations)
    # print(f'pearson shape:{pearson_correlations.shape}')
    print(f'pearson:{pearson_correlations}')
    del_list = []
    # 过滤掉相关系数小于阈值的特征
    for i in range(pearson_correlations.shape[0]):
        if abs(pearson_correlations[i][0]) < threshold:
            del_list.append(i)
    print(del_list)
    x_new = np.delete(x_arr, del_list, axis=1)
    print(x_new.shape)
    return x_new


def Envelop(x, fs):
    """
    包络分析，获取包络谱
    :param x: 信号数组
    :param fs: 信号频率
    :return: (fre,mag) 其中fre是包络谱频率，mag是包络谱
    """
    xh = signal.hilbert(x)  # 希尔伯特变换，得到解析信号
    xe = np.abs(xh)  # 解析信号取模，得到包络信号
    xe = xe - np.mean(xe)  # 去除直流分量
    xh3 = np.fft.rfft(xe) / len(xe)  # 傅里叶变换，得到包络谱
    mag = abs(xh3) * 2
    fre = np.linspace(0, fs / 2, int(len(xe) / 2 + 1))
    return fre, mag


# 进行LMD分解，返回重构信号
def get_lmd(signal):
    lmd = LMD()
    PFs, res = lmd.lmd(signal)
    # print(PFs.shape)
    PFlist = []
    for i, pf in enumerate(PFs):
        # print(f'PF{i}')
        corr, pvalue = stats.pearsonr(signal, pf)
        # pvalue = round(pvalue, 5)
        # kur = stats.kurtosis(pf)
        # print(f'相关系数 {corr}  p值 {pvalue}  峭度 {kur}')
        if corr < 0.3:
            PFlist.append(pf)
    # print(f'res')
    # cor_res, p_res = stats.pearsonr(signal, res)
    # kur_res = stats.kurtosis(res)
    # print(f'相关系数 {cor_res}  p值 {p_res}  峭度 {kur_res}')
    PFlist.append(res)
    return np.sum(PFlist, axis=0)


def lmd_proc(PFs):
    return np.sum(PFs, axis=0)


vib_folder = '../autodl-tmp/vib_signals'
dynamic = cnn.load_files(vib_folder).transpose(0, 2, 1)
# dynamic = np.load('../autodl-tmp/lmd_origin.npy')
print(f'Dynamic data shape: {dynamic.shape}')
time_fea_list = []
freq_fea_list = []
fs = 10000
for sample in dynamic:
    # sample = np.apply_along_axis(lmd_proc, axis=1, arr=sample)
    time_fea_list.append(get_signal_feature(sample, domin='time'))
    # fs_arr, Pxx_spec = signal.periodogram(sample, fs, scaling='spectrum')  # 原始信号转功率谱
    _, Pxx_spec = Envelop(sample, fs)  # 原始信号转包络谱
    freq_fea_list.append(get_signal_feature(Pxx_spec, domin='freq'))

time_fea_np = np.array(time_fea_list)
freq_fea_np = np.array(freq_fea_list)
time_fea = time_fea_np.reshape(time_fea_np.shape[0], -1)
freq_fea = freq_fea_np.reshape(freq_fea_np.shape[0], -1)
print(f'time fea shape:{time_fea.shape}, freq fea shape:{freq_fea.shape}')

# 读取输入和输出数据
data = pd.read_excel('s45c.xlsx').values
Y = data[:, [-1]]
time_fea = pearson_func(time_fea, Y, threshold=0.3)
freq_fea = pearson_func(freq_fea, Y, threshold=0.3)
X = np.concatenate((data[:, 0:-1], time_fea, freq_fea), axis=1)
print(f'X shape:{X.shape}, Y shape:{Y.shape}')

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)
# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 定义MOGPR的核函数，这里使用RBF核
kernel = kernels.RBF()

# 创建MOGPR模型
m = GPR((X_train, Y_train), kernel=kernel)

# 训练模型
opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(m.training_loss, m.trainable_variables)

# 测试
Y_pred, var = m.predict_f(X_test)

print(f'真实值：\n{Y_test}， \n预测值：\n{Y_pred}')

# 计算预测值与真实值之间的MSE
mse = mean_squared_error(Y_test, Y_pred.numpy())
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, Y_pred.numpy())
mape = mean_absolute_percentage_error(Y_test, Y_pred.numpy())
print(f'RMSE: {rmse}, R2: {r2}, MAPE: {mape}')

'''
plt.scatter(Y_test, Y_pred, alpha=0.7)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.legend(["value predicted", "y = x"])
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
'''
plt.plot(Y_test, 'r--', label='Real Ra')
plt.plot(Y_pred, 'b--', label='Predicted Ra')
plt.xlabel('Sample Index')
plt.ylabel('Ra')
plt.title('GPR Results')
plt.legend()
plt.show()