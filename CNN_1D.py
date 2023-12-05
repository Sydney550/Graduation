import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import keras
import tensorflow as tf
from keras import metrics
from scikeras.wrappers import KerasRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Masking, BatchNormalization, Activation
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.callbacks import EarlyStopping
from sklearn.model_selection import cross_val_score, KFold
from keras import optimizers
from keras import Model
from keras.regularizers import l2
import scipy.io as scio
from scipy import stats, signal
from tensorflow_addons.metrics import RSquare
import pandas as pd
from PyLMD import LMD
import time
import concurrent.futures
from pathos.multiprocessing import ProcessPool

'''用1D-CNN提取动态数据的特征'''

# 早停机制
earlystop_callback = EarlyStopping(monitor='r_square', min_delta=0.001, patience=2)


# 绘制训练过程的损失曲线
def plt_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.show()


# 绘制预测值与真实值的散点图
def plt_true_pred(y_true, y_pred):
    # plt.figure(figsize=(8, 8))
    # plt.scatter(y_true, y_pred, alpha=0.7)
    # plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # plt.legend(["Value predicted", "y = x"])
    # plt.xlabel('True Values')
    # plt.ylabel('Predicted Values')

    plt.plot(y_true, 'r--', label='Real Ra')
    plt.plot(y_pred, 'b--', label='Predicted Ra')
    plt.xlabel('Sample Index')
    plt.ylabel('Ra')
    plt.title('1D-CNN Regression Results')
    plt.legend()
    # plt.savefig('predict.png')
    # print('Figure saved.')
    plt.show()


# 转为包络信号
def Envelop(x):
    xh = signal.hilbert(x)  # 希尔伯特变换，得到解析信号
    xe = np.abs(xh)  # 解析信号取模，得到包络信号
    xe = xe - np.mean(xe)  # 去除直流分量
    xh3 = np.fft.rfft(xe) / len(xe)  # 傅里叶变换，得到包络谱
    mag = abs(xh3) * 2
    return mag


# 进行LMD分解，返回重构信号
def get_lmd(signal):
    # signal = Envelop(signal)
    lmd = LMD()
    PFs, res = lmd.lmd(signal)
    # print(PFs.shape)
    PFlist = []
    corr_list = []
    for pf in PFs:
        # print(f'PF{i}')
        corr, pvalue = stats.pearsonr(signal, pf)
        # pvalue = round(pvalue, 5)
        # kur = stats.kurtosis(pf)
        # print(f'相关系数 {corr}  p值 {pvalue}  峭度 {kur}')
        # if corr < 0.2:
        # PFlist.append(pf)
        corr_list.append(corr)
    index = np.argsort(corr_list)  # 从小到大排序，返回索引值
    # PFlist.append(PFs[index[-1]])
    PFlist.append(PFs[index[-2]])
    PFlist.append(PFs[index[-3]])
    PFlist.append(PFs[index[-4]])
    # PFlist.pop(index[-1])
    # print(f'res')
    # cor_res, p_res = stats.pearsonr(signal, res)
    # kur_res = stats.kurtosis(res)
    # print(f'相关系数 {cor_res}  p值 {p_res}  峭度 {kur_res}')
    PFlist.append(res)
    return np.sum(PFlist, axis=0)


def lmd_sample(data):
    start_time = time.time()
    new_data = np.apply_along_axis(get_lmd, axis=0, arr=data)
    end_time = time.time()
    print('完成时间：', end_time, "处理时间：", end_time - start_time)
    return new_data


def lmd_proc(PFs, origin=None):
    if origin is None:
        pass
        # PFs = np.delete(PFs, [3, 4, 5, 6, 7], axis=0)
    else:
        del_list = []
        indicators = []
        PFlist = []
        for i, pf in enumerate(PFs[:-1]):
            corr, _ = stats.pearsonr(origin, pf)  # 相关系数
            # kur = stats.kurtosis(pf)  # 峭度
            # 指标：相关系数0.5以上，p值0.05以下，峭度0.3以下
            # if corr > 0.3:
            #     del_list.append(i)
            indicators.append(corr)
        index = np.argsort(indicators)  # 从小到大排序返回索引
        # PFlist.append(PFs[index[-2]])
        # PFlist.append(PFs[index[-3]])
        # PFlist.append(PFs[index[-4]])
        # PFlist.append(PFs[-1])
        del_list = index[:-3]
        PFs = np.delete(PFs, del_list, axis=0)

    return np.sum(PFs, axis=0)


def load_lmd(data, origin=None):
    """
    加载LMD后的信号
    :param data: shape为(n,m,k)的LMD信号数组，其中n为原信号维度，m为一个信号的PFs个数，k为一个PFs的长度
    :param origin: shape为(n,k)的原始信号数组，其中n为信号维度，k为一个信号的长度
    :return: 合并PFs后得到的shape为(n,k)的数组，其中n为信号维度，k为一个信号的长度。若将信号转为了包络谱，则信号长度减半。
    """
    if origin is None:
        print('--------Original signals is None.--------')
        new_data = np.apply_along_axis(lmd_proc, axis=1, arr=data)
    else:
        start_time = time.time()
        new_data_list = []
        for i in range(data.shape[0]):
            new_data_list.append(lmd_proc(data[i], origin[i]))
        new_data = np.array(new_data_list)
        new_data = np.apply_along_axis(Envelop, axis=1, arr=new_data)  # 对合并PFs后的LMD信号做包络分析
        end_time = time.time()
        print('完成时间：', end_time, "处理时间：", end_time - start_time)

    return new_data


# 加载所有的 npy 文件
def load_files(path):
    print('------------loading data--------------')
    filelist = [f for f in os.listdir(path) if f.endswith('.mat')]
    npylist = []
    for f in sorted(filelist):
        data = scio.loadmat(f'{path}/{f}')
        key = f.split('.')[0]
        print(f'{key} shape:', data[key].shape)
        npylist.append(data[key])
    resized = np.stack([li for li in npylist], axis=1).transpose(0, 2, 1)
    print(f'resized data shape: {resized.shape}')
    return resized


def data_proc(all_data, all_y):
    print('------------Processing data--------------')
    # 划分训练集和测试集，比例9:1（random_state的大小任意，固定一个值可以使每次划分结果一样，可以复现），默认打乱顺序划分
    X_train, X_test, Y_train, Y_test = train_test_split(all_data, all_y, test_size=0.2, random_state=42, shuffle=True)
    # # 将训练集再划分为训练集和验证集
    # X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=22)

    # 数据标准化,只使用训练集数据来拟合scaler，然后用这个scaler来转换训练集和测试集
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    return X_train, X_test, Y_train, Y_test, all_data


# 构建 1D-CNN 模型
# 振动模型
def cnn_vibra(in_shape, lr=0.0001):
    print(f'input shape:{in_shape}, learning rate:{lr}')
    model = Sequential()
    model.add(Conv1D(16, 64, activation='relu', input_shape=in_shape))
    model.add(MaxPooling1D(16, strides=16))
    model.add(Conv1D(64, 32, activation='relu'))
    model.add(MaxPooling1D(4, strides=4))
    model.add(Conv1D(128, 16, activation='relu'))
    model.add(MaxPooling1D(2, strides=2))
    model.add(Conv1D(128, 16, activation='relu'))
    model.add(MaxPooling1D(2, strides=2))
    model.add(Conv1D(128, 8, activation='relu'))
    model.add(MaxPooling1D(2, strides=2))
    model.add(Conv1D(128, 8, activation='relu'))
    model.add(MaxPooling1D(2, strides=2))
    model.add(Flatten())  # 压平操作，把多维的输入一维化
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(1))
    print(model.summary())
    optimizer = optimizers.Adam(learning_rate=lr)  # 创建一个优化器对象并设置学习率
    model.compile(optimizer=optimizer, loss='mse', metrics=[metrics.RootMeanSquaredError(), RSquare()])
    return model


# def ticnn(in_shape, f1, f2, f3, f4, units):
def ticnn(in_shape):
    model = Sequential()  # 使用序列函数，让数据按照序列排队输入到卷积层
    model.add(Conv1D(16, 64, strides=8, padding='same', activation='relu', input_shape=in_shape))  # 第一个卷积层
    model.add(Dropout(0.5))  # 将经过第一个卷积层后的输出数据按照0.5的概率随机置零，也可以说是灭活
    model.add(BatchNormalization())
    # 添加批量标准层，将经过dropout的数据形成正态分布，有利于特征集中凸显，里面参数不需要了解和改动，直接黏贴或者删去均可。
    model.add(MaxPooling1D(2, strides=2))
    # 添加池化层，池化核大小为2步长为2，padding数据尾部补零。池化层不需要设置通道数，但卷积层需要。
    model.add(Conv1D(32, 3, padding='same', activation='relu'))  # 第二个卷积层，第二个卷积层则不在需要设置输入数据情况。
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2, strides=2))
    model.add(Conv1D(64, 3, padding='same', activation='relu'))  # 第三个卷积层
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2, strides=2))
    model.add(Conv1D(64, 3, padding='same', activation='relu'))  # 第四个卷积层
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2, strides=2))
    model.add(Conv1D(64, 3, padding='same', activation='relu'))  # 第五个卷积层
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2, strides=2))

    model.add(Conv1D(64, 3, activation='relu'))  # 第六个卷积层
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2, strides=2))
    model.add(Flatten())  # 将经过卷积和池化的数据展平，具体操作方式可以理解为，有n个通道的卷积输出，将每个通道压缩成一个数据，这样展评后就会出现n个数据
    model.add(Dense(100))
    model.add(BatchNormalization())
    model.add(Dense(1))  # 最后一层的参数设置要和标签种类一致
    print(model.summary())  # 模型小结，在训练时可以看到网络的结构参数
    model.compile(loss='mse', optimizer='adam', metrics=[metrics.RootMeanSquaredError(), RSquare()])

    return model


def ticnn2(in_shape):
    model = Sequential()  # 使用序列函数，让数据按照序列排队输入到卷积层
    model.add(Conv1D(16, 64, strides=8, padding='same', kernel_regularizer=l2(1e-4), input_shape=in_shape))  # 第一个卷积层
    #     model.add(Dropout(0.5))  # 将经过第一个卷积层后的输出数据按照0.5的概率随机置零，也可以说是灭活
    model.add(Activation('relu'))
    # 添加批量标准层，将经过dropout的数据形成正态分布，有利于特征集中凸显，里面参数不需要了解和改动，直接黏贴或者删去均可。
    model.add(MaxPooling1D(2, strides=2))
    # 添加池化层，池化核大小为2步长为2，padding数据尾部补零。池化层不需要设置通道数，但卷积层需要。
    model.add(Conv1D(32, 3, padding='same', kernel_regularizer=l2(1e-4)))  # 第二个卷积层，第二个卷积层则不在需要设置输入数据情况。
    model.add(Activation('relu'))
    model.add(MaxPooling1D(2, strides=2))
    model.add(Conv1D(64, 3, padding='same', kernel_regularizer=l2(1e-4)))  # 第三个卷积层
    model.add(Activation('relu'))
    model.add(MaxPooling1D(2, strides=2))
    model.add(Conv1D(64, 3, padding='same', kernel_regularizer=l2(1e-4)))  # 第四个卷积层
    model.add(Activation('relu'))
    model.add(MaxPooling1D(2, strides=2))
    model.add(Conv1D(64, 3, padding='same', kernel_regularizer=l2(1e-4)))  # 第五个卷积层
    model.add(Activation('relu'))
    model.add(MaxPooling1D(2, strides=2))
    model.add(Conv1D(64, 3, kernel_regularizer=l2(1e-4)))  # 第六个卷积层
    model.add(Activation('relu'))
    model.add(MaxPooling1D(2, strides=2))
    model.add(Flatten())  # 将经过卷积和池化的数据展平，具体操作方式可以理解为，有n个通道的卷积输出，将每个通道压缩成一个数据，这样展评后就会出现n个数据
    model.add(Dense(100, activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(Dense(1))  # 最后一层的参数设置要和标签种类一致
    print(model.summary())  # 模型小结，在训练时可以看到网络的结构参数
    optimizer = optimizers.Adam(learning_rate=0.001)  # 创建一个优化器对象并设置学习率
    model.compile(loss='mse', optimizer=optimizer, metrics=[metrics.RootMeanSquaredError(), RSquare()])

    return model


# 编译模型
def nn_train(model, X_train, Y_train, X_val, Y_val, epoch=100, bs=8):
    print("--------------Training-------------")
    # 训练模型
    '''
    X为输入数据，y为数据标签；batch_size：每次梯度更新的样本数，默认为32。
    verbose: 0,1,2. 0=训练过程无输出，1=显示训练过程进度条，2=每训练一个epoch打印一次信息
    validation_data: 验证集
    '''
    # tensorboard
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir="../tf-logs")
    history = model.fit(X_train, Y_train, epochs=epoch, batch_size=bs, validation_data=(X_val, Y_val), verbose=1,
                        callbacks=[tb_callback])
    # plt_loss(history)
    # 评估模型
    score = model.evaluate(X_val, Y_val, verbose=1)  # 准确率
    print(f'\n Training {model.metrics_names[0]}: {score[0]}, {model.metrics_names[1]}: {score[1]}')

    return model


# 预测测试集
def nn_test(model, X_test, Y_test, epoch=100, batch=8):
    print('-------------Testing-----------------')
    Y_pred = model.predict(X_test)
    print('\n', Y_test, '\n', Y_pred)

    plt_true_pred(Y_test, Y_pred)

    # 计算均方误差MSE和决定系数R2
    mse = mean_squared_error(Y_test, Y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_test, Y_pred)
    print('Test MSE:', mse, '\t, RMSE:', rmse, '\t, R2:', r2)
    with open("test.txt", "w") as f:
        f.write(f'Test MSE: {mse},\tRMSE: {rmse},\tR2: {r2}')
    return


# 交叉验证
def cross_val(X_train, Y_train, epoch=100, bs=8):
    print('-------------Cross Validation-----------')
    # 包装模型
    model1 = KerasRegressor(model=ticnn, epochs=epoch, batch_size=bs)
    # 10折交叉验证，评价函数用R2，打乱数据顺序
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)

    # my_r2scorer = make_scorer(my_r2, greater_is_better=True)  # 自定义评价函数，greater_is_better：是否指标越大越好
    scores = cross_val_score(model1, X_train, Y_train, scoring='r2', cv=kfold, verbose=1)
    print('10折交叉验证的R2系数:', scores)
    print('R2的平均值 / 标准差:', scores.mean(), ' / ', scores.std())

    '''
    # 有归一化的CV
    scores = []
    for train_index, val_index in kfold.split(X_train, Y_train):
        # train_index 就是分类的训练集的下标，val_index 就是分配的验证集的下标
        train_x, train_y = X_train[train_index], Y_train[train_index]  # 本组训练集
        val_x, val_y = X_train[val_index], Y_train[val_index]  # 本组验证集
        # 数据归一化,只使用训练集数据来拟合scaler，然后用这个scaler来转换训练集和测试集
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x.reshape(-1, train_x.shape[-1])).reshape(train_x.shape)
        val_x = scaler.transform(val_x.reshape(-1, val_x.shape[-1])).reshape(val_x.shape)
        # 训练本组的数据，并计算准确率
        model1.fit(train_x, train_y)
        prediction = model1.predict(val_x)
        score = r2_score(val_y, prediction)
        print('R2:', score)
        scores.append(score)
    print('10折交叉验证的R2系数:', scores)
    print('R2的平均值 / 标准差:', np.mean(scores), ' / ', np.std(scores))
    '''
    return


# 提取特征
def feature_train(load_data, targets, epoch, bs, lr):
    # 加载信号数据
    # load_data = load_files(x_folder, cut_time)
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test, all_data = data_proc(load_data, targets)
    in_shape = x_train.shape[1:]
    # 建立模型
    model = cnn_vibra(in_shape, lr=lr)
    # 训练模型
    nn_model = nn_train(model, x_train, y_train, x_test, y_test, epoch, bs)
    nn_model.save('cnn1d.h5')
    features = feature_predict(nn_model, all_data)
    # np.save('features.npy', features)
    return features


def feature_predict(model, x_data):
    # 以最后一个全连接层的输出为提取的特征
    feature_model = keras.Model(inputs=model.inputs, outputs=model.layers[-1].output)
    features = feature_model.predict(x_data)
    return features


if __name__ == '__main__':
    # 显存按需分配
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print('GPU available:', tf.test.is_gpu_available())
    vib_folder = '../autodl-tmp/vib_signals'
    drop = 0.5
    lr = 0.0001
    epoch = 60
    batch = 32

    # 读取S45C数据
    df = pd.read_excel('s45c.xlsx').values
    targets = df[:, [-1]]
    print('targets shape:', targets.shape)

    # 加载数据
    '''
    # data = load_files(vib_folder)
    original = np.load('../autodl-tmp/signals_origin.npy')
    print('original shape:', original.shape)
    data = np.load('../autodl-tmp/lmd_origin.npy')
    print('data shape:', data.shape)
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     new_data_list = list(executor.map(lmd_sample, data))
    with ProcessPool(1) as pool:
        new_data_list = pool.map(load_lmd, data, original)
    new_data = np.array(new_data_list).transpose(0, 2, 1)
    print(f'data shape after lmd: {new_data.shape}')
    '''
    new_data = np.load('../autodl-tmp/data_lmd.npy')
    # 处理数据
    x_train, x_test, y_train, y_test, all_data = data_proc(new_data, targets)
    train_shape = x_train.shape[1:]

    model = cnn_vibra(train_shape, lr=lr)
    #     nn_model = ticnn2(train_shape)
    print('train shape:', x_train.shape)
    # 训练模型
    nn_model = nn_train(model, x_train, y_train, x_test, y_test, epoch, batch)

    # 测试
    nn_test(nn_model, x_test, y_test)

    # 交叉验证
    # cross_val(all_data, targets, epoch, batch)

    # features = get_features(nn_model, all_data)
    # print('Features shape: ', features.shape)
    # print('Features: \n', features)
