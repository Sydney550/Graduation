import numpy as np
import gpflow
from gpflow.models import GPR
from gpflow.kernels import RBF
import pandas as pd
from sklearn.model_selection import train_test_split
from fractions import Fraction
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 读取输入和输出数据
data = pd.read_excel('/kaggle/input/data-for-graduation/s45c.xlsx').values
X = data[:, 0:-1]
Y = data[:, [-1]]
print(f'X shape:{X.shape}, Y shape:{Y.shape}')

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)
# 标准化
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# 定义MOGPR的核函数，这里使用RBF核
kernel = RBF()

# 创建MOGPR模型
m = GPR((X_train, Y_train), kernel=kernel)

# 训练模型
opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

# 测试
Y_pred, var = m.predict_f(X_test)

print(f'真实值：\n{Y_test}， \n预测值：\n{Y_pred}')

# 计算预测值与真实值之间的MSE
mse = mean_squared_error(Y_test, Y_pred.numpy())
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, Y_pred.numpy())
print(f'RMSE: {rmse}, R2: {r2}')

plt.scatter(Y_test, Y_pred, alpha=0.7)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--')
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.legend(["value predicted", "y = x"])
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('MOGPR Results')
plt.show()