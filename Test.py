# 导入需要用到的package
import numpy as np
import json
from sklearn import datasets  # 导入sklearn中集成的数据集
import pandas as pd  # 导入pandas
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 数据处理
def load_data():
    # 从文件导入数据
    datafile = 'D:/Users/文档/program/Python/Machine Learning/boston house price prediction/housing.data'
    data = np.fromfile(datafile, sep=' ')
    # 从sklearn导入波士顿房价数据
    # df_boston = datasets.load_boston()
    # print("波士顿房价特征:\n", df_boston.data[:5])  # print前5个特征
    # print("波士顿房价特征的维度：\n", df_boston.data.shape)
    # print("波士顿房价标签:\n", df_boston.target[:5])  # print前5个标签
    # print("波士顿房价标签的维度：\n", df_boston.target.shape)

    # 每条数据包括14项，其中前面13项是影响因素，第14项是相应的房屋价格中位数
    feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', \
                      'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
    feature_num = len(feature_names)
    # 将原始数据进行Reshape，变成[N, 14]这样的形状
    # # " // " 表示整数除法,返回不大于结果的一个最大的整数
    data = data.reshape([data.shape[0] // feature_num, feature_num])
    # # 查看数据
    # x = data[0]
    # print(x.shape) (14,)
    # print(x)

    # 将原数据集拆分成训练集和测试集
    # 这里使用80%的数据做训练，20%的数据做测试
    # 测试集和训练集必须是没有交集的
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]
    # print(data.shape) (506, 14)
    # print(data.shape[0]) 506
    # print(data.shape[1]) 14
    # print(type(data)) <class 'numpy.ndarray'>

    # 计算train数据集的最大值，最小值，平均值
    maximums, minimums, avgs = training_data.max(axis=0), training_data.min(axis=0), \
                                 training_data.sum(axis=0) / training_data.shape[0]
    # 对数据进行归一化处理
    for i in range(feature_num):
        #print(maximums[i], minimums[i], avgs[i])
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])
    # 训练集和测试集的划分比例
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data

# 构建神经网络
class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，此处设置固定的随机数种子
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0.
    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return z
    def loss(self, z, y):
        error = z - y
        num_samples = error.shape[0]
        cost = error * error
        cost = np.sum(cost) / num_samples
        return cost
    def gradient(self, x, y):
        z = self.forward(x)
        gradient_w = (z-y)*x
        gradient_w = np.mean(gradient_w, axis=0)
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = (z - y)
        gradient_b = np.mean(gradient_b)        
        return gradient_w, gradient_b
    def update(self, gradient_w, gradient_b, eta = 0.01):
        self.w = self.w - eta * gradient_w
        self.b = self.b - eta * gradient_b
    def train(self, x, y, iterations=100, eta=0.01):
        losses = []
        for i in range(iterations):
            z = self.forward(x)
            L = self.loss(z, y)
            gradient_w, gradient_b = self.gradient(x, y)
            self.update(gradient_w, gradient_b, eta)
            losses.append(L)
            if (i+1) % 10 == 0:
                print('iter {}, loss {}'.format(i, L))
        return losses
# 获取数据
train_data, test_data = load_data()
x = train_data[:, :-1]
y = train_data[:, -1:]

# 创建网络
net = Network(13)
num_iterations=10000
# 启动训练
losses = net.train(x,y, iterations=num_iterations, eta=0.03)
# 画出损失函数的变化趋势
plot_x = np.arange(num_iterations)
plot_y = np.array(losses)
plt.plot(plot_x, plot_y)
plt.show()

# 数据可视化
x_Test = test_data[:, :-1]
y_pres = net.forward(x_Test)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.plot(test_data[:, -1:] ,"r--") 
plt.plot(y_pres,"b--") 
plt.title("红色为真实值，蓝色为预测值 eta = 0.05")  
plt.show()  


datafile = 'D:/Users/文档/program/Python/Machine Learning/boston house price prediction/housing.data'
data = np.fromfile(datafile, sep=' ')

feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', \
                      'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
feature_num = len(feature_names)
data = data.reshape([data.shape[0] // feature_num, feature_num])
ratio = 0.8
offset = int(data.shape[0] * ratio)
training_data = data[:offset]
maximums, minimums, avgs = training_data.max(axis=0), training_data.min(axis=0), \
                                training_data.sum(axis=0) / training_data.shape[0]
    # 对数据进行归一化处理
y_pres = y_pres * (maximums[13] - minimums[13]) + avgs[13]
for i in range(feature_num):
    #print(maximums[i], minimums[i], avgs[i])
    test_data[:, i] = test_data[:, i] * (maximums[i] - minimums[i]) + avgs[i]

# 输出预测结果
x_Test = test_data[:, :-1]
# print(test_data[:, -1:])
# print(y_pres)
y = np.concatenate((test_data[:, -1:],y_pres),axis=1)
print(y)
# 真实值与预测值对比  
# y_preds = np.apply_along_axis(net.forward, 1, test_data[:, :-1])  
# plt.plot(test_data[:, -1:] ,"r--")  
# plt.plot(y_preds,"b--")  
# plt.title("红色为真实值，蓝色为预测值 eta = 0.03")  
# plt.show()  

# # 小批量随机梯度下降法
# class Network(object):
#     def __init__(self, num_of_weights):
#         # 随机产生w的初始值
#         # 为了保持程序每次运行结果的一致性，此处设置固定的随机数种子
#         #np.random.seed(0)
#         self.w = np.random.randn(num_of_weights, 1)
#         self.b = 0.
#     def forward(self, x):
#         z = np.dot(x, self.w) + self.b
#         return z
#     def loss(self, z, y):
#         error = z - y
#         num_samples = error.shape[0]
#         cost = error * error
#         cost = np.sum(cost) / num_samples
#         return cost
#     def gradient(self, x, y):
#         z = self.forward(x)
#         N = x.shape[0]
#         gradient_w = 1. / N * np.sum((z-y) * x, axis=0)
#         gradient_w = gradient_w[:, np.newaxis]
#         gradient_b = 1. / N * np.sum(z-y)
#         return gradient_w, gradient_b
#     def update(self, gradient_w, gradient_b, eta = 0.01):
#         self.w = self.w - eta * gradient_w
#         self.b = self.b - eta * gradient_b
#     def train(self, training_data, num_epoches, batch_size=10, eta=0.01):
#         n = len(training_data)
#         losses = []
#         for epoch_id in range(num_epoches):
#             # 在每轮迭代开始之前，将训练数据的顺序随机的打乱，
#             # 然后再按每次取batch_size条数据的方式取出
#             np.random.shuffle(training_data)
#             # 将训练数据进行拆分，每个mini_batch包含batch_size条的数据
#             mini_batches = [training_data[k:k+batch_size] for k in range(0, n, batch_size)]
#             for iter_id, mini_batch in enumerate(mini_batches):
#                 #print(self.w.shape)
#                 #print(self.b)
#                 x = mini_batch[:, :-1]
#                 y = mini_batch[:, -1:]
#                 a = self.forward(x)
#                 loss = self.loss(a, y)
#                 gradient_w, gradient_b = self.gradient(x, y)
#                 self.update(gradient_w, gradient_b, eta)
#                 losses.append(loss)
#                 print('Epoch {:3d} / iter {:3d}, loss = {:.4f}'.
#                                  format(epoch_id, iter_id, loss))
#         return losses
# # 获取数据
# train_data, test_data = load_data()
# # 创建网络
# net = Network(13)
# # 启动训练
# losses = net.train(train_data, num_epoches=50, batch_size=100, eta=0.1)
# # 画出损失函数的变化趋势
# plot_x = np.arange(len(losses))
# plot_y = np.array(losses)
# plt.plot(plot_x, plot_y)
# plt.show()