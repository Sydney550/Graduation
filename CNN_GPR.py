import numpy as np
import gpflow
from gpflow.models import GPR
from gpflow.kernels import RBF
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import CNN_1D as cnn
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import pearsonr, spearmanr
from pyentrp import entropy as ent
from scipy import stats

# 显存按需分配
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


class CNN_GPR():
    def get_time_domain_feature(self, data):
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
        entropy = np.apply_along_axis(ent.shannon_entropy, axis=1, arr=data)  # 熵
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

    def pearson_func(self, x_arr, y_arr):
        # 初始化一个空的列表来保存每个特征的皮尔逊相关系数
        pearson_correlations = []
        # 对于X中的每个特征
        for i in range(x_arr.shape[1]):
            feature = x_arr[:, i]
            # 计算该特征与Y的皮尔逊相关系数
            correlations = [pearsonr(feature, y_arr[:, 0])[0]]
            # 将结果添加到列表中
            pearson_correlations.append(correlations)
        # 将结果转换为numpy数组
        pearson_correlations = np.array(pearson_correlations)
        # 打印结果
        # print(f'pearson shape:{pearson_correlations.shape}')
        print(f'pearson:{pearson_correlations}')
        del_list = []
        for i in range(pearson_correlations.shape[0]):
            if abs(pearson_correlations[i][0]) < 0.3:
                del_list.append(i)
        print(del_list)
        x_new = np.delete(x_arr, del_list, axis=1)
        print(x_new.shape)
        return x_new

    def train(self, static_data, dynamic_data, y_data):
        epoch = 100
        batch = 16
        # 提取动态数据特征
        dynamic_features = cnn.feature_train(dynamic_data, y_data, epoch, batch)
        dynamic_features = self.pearson_func(dynamic_features, y_data)
        print(f'static.shape: {static_data.shape}')
        print(f'dynamic shape :{dynamic_features.shape}')

        # 合并动静态数组
        X = np.concatenate((static_data, dynamic_features), axis=1)
        print(f'X shape: {X.shape}')

        # 数据集标准化
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # 定义MOGPR的核函数，这里使用RBF核
        kernel = RBF()

        # 创建MOGPR模型
        m = GPR((X, y_data), kernel=kernel, noise_variance=0.1)

        # 训练模型
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))
        # print('GPR training:\n', opt_logs)
        return m, scaler

    def predict(self, gpr_model, static, dynamic, Y_test, scaler):
        cnn_model = tf.keras.models.load_model('cnn1d.h5')
        dynamic_features = cnn.feature_predict(cnn_model, dynamic)
        dynamic_features = self.pearson_func(dynamic_features, Y_test)
        X_test = np.concatenate((static, dynamic_features), axis=1)
        X_test = scaler.transform(X_test)
        # 测试
        Y_pred, var = gpr_model.predict_f(X_test)
        print(f'真实值：\n{Y_test}， \n预测值：\n{Y_pred}')

        # 计算预测值与真实值之间的MSE
        mse = mean_squared_error(Y_test, Y_pred.numpy())
        rmse = np.sqrt(mse)
        r2 = r2_score(Y_test, Y_pred.numpy())
        print(f'RMSE: {rmse}, R2: {r2}')
        self.draw_plt(Y_test, Y_pred)
        return

    def draw_plt(self, Y_test, Y_pred):
        plt.scatter(Y_test, Y_pred, alpha=0.7)
        plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--')
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        plt.legend(["value predicted", "y = x"])
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('CNN-GPR Results')
        plt.show()

    def main(self):
        # filter1 = 16
        # filter2 = 32
        # filter3 = 64
        # dense_units = 100
        vib_folder = 'vib_signals'
        s45c_file = 's45c.xlsx'

        df = pd.read_excel(s45c_file).values
        # 读取静态数据
        static = df[:, :-1]
        print(f'static data shape: {static.shape}')
        # 读取质检数据
        quality = df[:, [-1]]
        # 读取动态数据
        dynamic = cnn.load_files(vib_folder)
        print(f'Dynamic data shape: {dynamic.shape}')

        dynamic_trans = dynamic.transpose(0, 2, 1)
        time_fea_list = []
        for signal in dynamic_trans:
            time_fea_list.append(self.get_time_domain_feature(signal))
        time_fea_np = np.array(time_fea_list)
        time_fea = time_fea_np.reshape(time_fea_np.shape[0], -1)
        print(f'time fea shape:{time_fea.shape}')
        time_fea = self.pearson_func(time_fea, quality)
        static = np.concatenate((static, time_fea), axis=1)

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
