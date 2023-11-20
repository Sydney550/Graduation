from scipy import stats
import CNN_1D as cnn
import numpy as np
from pyentrp import entropy as ent
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import sys


def get_time_domain_feature(data):
    """
    提取 15个 时域特征

    @param data: shape 为 (m, n) 的 2D array 数据，其中，m 为样本个数， n 为样本（信号）长度
    @return: shape 为 (m, 15)  的 2D array 数据，其中，m 为样本个数。即 每个样本的16个时域特征
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
    std = np.std(data, axis=1)  # 标准差
    kurtosis = stats.kurtosis(data, axis=ax)  # 峭度
    skewness = stats.skew(data, axis=ax)  # 偏度
    entropy = np.apply_along_axis(ent.shannon_entropy, axis=ax, arr=data)  # 熵
    p_p_value = max_value - min_value  # 峰峰值
    abs_mean = np.mean(abs(data), axis=ax)  # 绝对平均值（整流平均值）
    # square_root_amplitude = (np.sum(np.sqrt(abs(data)), axis=ax) / cols) ** 2  # 方根幅值
    # mean_amplitude = np.sum(np.abs(data), axis=ax) / cols  # 平均幅值 == 绝对平均值

    # 无量纲统计量
    shape_factor = rms / abs_mean  # 波形因子
    crest_factor = peak_value / rms  # 峰值因子
    # clearance_factor = peak_value / square_root_amplitude  # 裕度因子
    # impulse_factor = peak_value / abs_mean  # 脉冲因子
    # kurtosis_factor = kurtosis / (rms**4)  # 峭度因子

    features = [max_value, min_value, peak_value, mean, rms, variance, std, kurtosis, skewness, entropy, p_p_value,
                shape_factor, crest_factor]
    return np.array(features).T


path = 'vib_signals'
# filelist = [f for f in os.listdir(path) if f.endswith('.mat')]
# f = filelist[1]
# data = scio.loadmat(f'{path}/{f}')
# key = f.split('.')[0]
# time_fea = get_time_domain_feature(data[key])
# print(f'time fea shape:{time_fea.shape}')
# print('time features:\n', time_fea)

# filelist = [f for f in os.listdir(path) if f.endswith('.mat')]
# npylist = []
# for f in filelist:
#     data = scio.loadmat(f'{path}/{f}')
#     key = f.split('.')[0]
#     time_fea = get_time_domain_feature(data[key])
#     npylist.append(time_fea)
# dynamic = np.stack([li for li in npylist], axis=1)

dynamic = cnn.load_files(path).transpose(0, 2, 1)
print(f'Dynamic data shape: {dynamic.shape}')
time_fea_list = []
for signal in dynamic:
    time_fea_list.append(get_time_domain_feature(signal))
time_fea = np.array(time_fea_list)
print(f'time fea shape:{time_fea.shape}')
# print('time features:\n', time_fea)
reshaped = time_fea.reshape(time_fea.shape[0], -1)
print(f'reshaped time fea shape:{reshaped.shape}')
# print('reshaped time features:\n', reshaped)
# print(f'have nan:{np.isnan(reshaped).sum()}')
# np.savetxt('time_fea.csv', reshaped, delimiter=",")

s45c_file = 's45c.xlsx'
df = pd.read_excel(s45c_file).values
X = reshaped
Y = df[:, [-1]]
print(f'Y shape:{Y.shape}')

# 初始化一个空的列表来保存每个特征的皮尔逊相关系数
pearson_correlations = []

# 对于X中的每个特征
for i in range(X.shape[1]):
    feature = X[:, i]
    # 计算该特征与Y的皮尔逊相关系数
    correlations = [pearsonr(feature, Y[:, 0])[0]]
    # 将结果添加到列表中
    pearson_correlations.append(correlations)

# 将结果转换为numpy数组
pearson_correlations = np.array(pearson_correlations)

# 打印结果
print(f'pearson shape:{pearson_correlations.shape}')
print(pearson_correlations)
del_list = []
for i in range(pearson_correlations.shape[0]):
    if abs(pearson_correlations[i][0]) < 0.1:
        del_list.append(i)
print(del_list)
pearson_correlations = np.delete(pearson_correlations, del_list, axis=0)
print(pearson_correlations)
print(f'deled pearson shape:{pearson_correlations.shape}')
reshaped = np.delete(reshaped, del_list, axis=1)
print(reshaped.shape)