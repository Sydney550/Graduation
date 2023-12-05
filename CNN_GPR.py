import sys

import numpy as np
import gpflow
from gpflow.models import GPR
from gpflow.kernels import RBF
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import CNN_1D as cnn
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import pearsonr, spearmanr
from pyentrp import entropy as ent
from scipy import stats, signal
from pathos.multiprocessing import ProcessPool

# 显存按需分配
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


class CNN_GPR():
    def get_signal_feature(self, data, domin='time'):
        """
        提取 15个 时域特征

        :param data: shape 为 (m, n) 的 2D array 数据，其中，m 为样本个数， n 为样本（信号）长度
        :param domin: 提取时域特征('time')或频域特征('freq')
        :return: shape 为 (m, 15)  的 2D array 数据，其中，m 为样本个数。即 每个样本的16个时域特征
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

    def pearson_func(self, x_arr, y_arr):
        """
        通过皮尔逊相关系数筛选特征

        :param x_arr: shape为(m,n)的特征数组，其中m为样本数，n为特征个数
        :param y_arr: shape为(m,1)的目标数组，其中m为样本数
        :return: 经过相关系数筛选后的新数组，shape为(m,k)，其中m为样本数，k为筛选后的特征个数
        """
        # 初始化一个空的列表来保存每个特征的皮尔逊相关系数
        pearson_correlations = []
        # 对于X中的每个特征
        print('x_arr shape[1]:', x_arr.shape[1])
        for i in range(x_arr.shape[1]):
            feature = x_arr[:, i]
            # 计算该特征与Y的皮尔逊相关系数
            correlations = [pearsonr(feature, y_arr[:, 0])[0]]
            pearson_correlations.append(correlations)
        pearson_correlations = np.array(pearson_correlations)
        # print(f'pearson shape:{pearson_correlations.shape}')
        print(f'pearson:{pearson_correlations}')
        del_list = []
        # 过滤掉相关系数小于阈值的特征
        for i in range(pearson_correlations.shape[0]):
            if abs(pearson_correlations[i][0]) < 0.3:
                del_list.append(i)
        print('del list:\n',del_list)
        x_new = np.delete(x_arr, del_list, axis=1)
        print(x_new.shape)
        return x_new

    def Envelop(self, x):
        """
        包络分析，获取包络谱
        :param x: 信号数组
        :return : mag - 信号转换后的包络谱
        """
        xh = signal.hilbert(x)  # 希尔伯特变换，得到解析信号
        xe = np.abs(xh)  # 解析信号取模，得到包络信号
        xe = xe - np.mean(xe)  # 去除直流分量
        xh3 = np.fft.rfft(xe) / len(xe)  # 傅里叶变换，得到包络谱
        mag = abs(xh3) * 2
        return mag

    def train(self, static_data, dynamic_data, y_data):
        epoch = 80
        batch = 32
        learning_rate = 0.0001
        # 提取动态数据特征
        dynamic_features = cnn.feature_train(dynamic_data, y_data, epoch, batch, lr=learning_rate)
        # dynamic_features = self.pearson_func(dynamic_features, y_data)
        print(f'train static shape: {static_data.shape}')
        print(f'train dynamic shape: {dynamic_features.shape}')

        # 静态数据标准化
        scaler = StandardScaler()
        static_data = scaler.fit_transform(static_data)

        # 合并动静态数组
        X = np.concatenate((static_data, dynamic_features), axis=1)
        print(f'train X shape: {X.shape}')

        # 定义MOGPR的核函数，这里使用RBF核
        kernel = RBF()

        m = GPR((X, y_data), kernel=kernel)

        # 训练模型
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(m.training_loss, m.trainable_variables)
        # print('GPR training:\n', opt_logs)
        return m, scaler

    def predict(self, gpr_model, static, dynamic, Y_test, scaler):
        cnn_model = tf.keras.models.load_model('cnn1d.h5')
        dynamic_features = cnn.feature_predict(cnn_model, dynamic)
        # dynamic_features = self.pearson_func(dynamic_features, Y_test)
        static = scaler.transform(static)
        X_test = np.concatenate((static, dynamic_features), axis=1)
        print(f'predict static shape:{static.shape}, dynamic shape:{dynamic_features.shape}, X shape:{X_test.shape}')
        # 测试
        Y_pred, var = gpr_model.predict_f(X_test)
        print(f'真实值：\n{Y_test}， \n预测值：\n{Y_pred}')

        # 计算预测值与真实值之间的MSE
        mse = mean_squared_error(Y_test, Y_pred.numpy())
        rmse = np.sqrt(mse)
        r2 = r2_score(Y_test, Y_pred.numpy())
        mape = mean_absolute_percentage_error(Y_test, Y_pred.numpy())
        print(f'RMSE: {rmse}, R2: {r2}, MAPE: {mape}')
        self.draw_plt(Y_test, Y_pred)
        return

    def draw_plt(self, Y_true, Y_pred):
        # plt.scatter(Y_test, Y_pred, alpha=0.7)
        # plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--')
        # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        # plt.legend(["value predicted", "y = x"])
        # plt.xlabel('True Values')
        # plt.ylabel('Predicted Values')

        plt.plot(Y_true, 'r--', label='Real Ra')
        plt.plot(Y_pred, 'b--', label='Predicted Ra')
        plt.xlabel('Sample Index')
        plt.ylabel('Ra')
        plt.title('1D-CNN Regression Results')
        plt.title('CNN-GPR Results')
        plt.legend()
        plt.show()

    def main(self):
        # filter1 = 16
        # filter2 = 32
        # filter3 = 64
        # dense_units = 100
        # vib_folder = '../autodl-tmp/vib_signals'
        s45c_file = 's45c.xlsx'

        df = pd.read_excel(s45c_file).values
        # 读取静态数据
        static = df[:, :-1]
        print(f'static data shape: {static.shape}')
        # 读取质检数据
        quality = df[:, [-1]]
        print('quality data shape: ', quality.shape)

        # 读取动态数据
        original = np.load('../autodl-tmp/signals_origin.npy')
        print('original shape:', original.shape)
        '''
        lmded = np.load('../autodl-tmp/lmd_origin.npy')
        print('lmded shape:', lmded.shape)
        with ProcessPool(1) as pool:
            new_data_list = pool.map(cnn.load_lmd, lmded, original)
        dynamic = np.array(new_data_list).transpose(0, 2, 1)
        # dynamic = cnn.load_files(vib_folder)
        np.save('../autodl-tmp/gpr_dynamic.npy', dynamic)
        '''

        dynamic = np.load('../autodl-tmp/gpr_dynamic.npy')
        print(f'Dynamic data shape: {dynamic.shape}')

        # 提取时域频域特征
        # original_trans = original.transpose(0, 2, 1)
        time_fea_list = []
        freq_fea_list = []
        for signal in original:
            time_fea_list.append(self.get_signal_feature(signal, 'time'))
            Pxx_spec = self.Envelop(signal)  # 原始信号转包络谱
            freq_fea_list.append(self.get_signal_feature(Pxx_spec, 'freq'))
        time_fea_np = np.array(time_fea_list)
        time_fea = time_fea_np.reshape(time_fea_np.shape[0], -1)
        freq_fea_np = np.array(freq_fea_list)
        freq_fea = freq_fea_np.reshape(freq_fea_np.shape[0], -1)
        print(f'time fea shape:{time_fea.shape}')
        print(f'freq fea shape:{freq_fea.shape}')
        time_fea = self.pearson_func(time_fea, quality)
        freq_fea = self.pearson_func(freq_fea, quality)
        static = np.concatenate((static, time_fea, freq_fea), axis=1)

        static_train, static_test, dynamic_train, dynamic_test, quality_train, quality_test = train_test_split(static,
                                                                                                               dynamic,
                                                                                                               quality,
                                                                                                               test_size=0.2,
                                                                                                               random_state=42)

        # train(filter1, filter2, filter3, dense_units, load_data)

        gpr_model, scaler = self.train(static_train, dynamic_train, quality_train)
        self.predict(gpr_model, static_test, dynamic_test, quality_test, scaler)


if __name__ == "__main__":
    gpr = CNN_GPR()
    gpr.main()
