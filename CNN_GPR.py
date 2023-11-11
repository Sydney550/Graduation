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

# 显存按需分配
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


class CNN_GPR():

    def train(self, static_data, dynamic_data, y_data):
        epoch = 100
        batch = 16
        # 提取动态数据特征
        dynamic_features = cnn.feature_train(dynamic_data, y_data, epoch, batch)
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
        vib_folder = '/kaggle/input/data-for-graduation/vib_signals'
        s45c_file = '/kaggle/input/data-for-graduation/s45c.xlsx'

        df = pd.read_excel(s45c_file).values
        # 读取静态数据
        static = df[:, :-1]
        print(f'static data shape: {static.shape}')
        # 读取质检数据
        quality = df[:, -1]
        # 读取动态数据
        dynamic = cnn.load_files(vib_folder)
        print(f'Dynamic data shape: {dynamic.shape}')

        static_train, static_test, dynamic_train, dynamic_test, quality_train, quality_test = train_test_split(static,
                                                                                                               dynamic,
                                                                                                               quality,
                                                                                                               test_size=0.2,
                                                                                                               random_state=22)

        # train(filter1, filter2, filter3, dense_units, load_data)
        gpr_model, scaler = self.train(static_train, dynamic_train, quality_train)
        self.predict(gpr_model, static_test, dynamic_test, quality_test, scaler)


if __name__ == "__main__":
    gpr = CNN_GPR()
    gpr.main()
