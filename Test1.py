#coding=utf-8
import numpy as np
import matplotlib.pylab as plt
import random

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

class NeuralNetwork(object):
    def __init__(self, sizes, act, act_derivative, cost_derivative):
        #sizes表示神经网络各层的神经元个数，第一层为输入层，最后一层为输出层
        #act为神经元的激活函数
        #act_derivative为激活函数的导数
        #cost_derivative为损失函数的导数
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(nueron_num, 1) for nueron_num in sizes[1:]]
        self.weights = [np.random.randn(next_layer_nueron_num, nueron_num)
            for nueron_num, next_layer_nueron_num in zip(sizes[:-1], sizes[1:])]
        self.act=act
        self.act_derivative=act_derivative
        self.cost_derivative=cost_derivative
 
    #前向反馈（正向传播）
    def feedforward(self, a):
        #逐层计算神经元的激活值，公式(4)
        for b, w in zip(self.biases, self.weights):
            a = self.act(np.dot(w,a)+b)
        return a
 
    #随机梯度下降算法
    def SGD(self, training_data, epochs, batch_size, learning_rate):
        #将训练样本training_data随机分为若干个长度为batch_size的batch
        #使用各个batch的数据不断调整参数，学习率为learning_rate
        #迭代epochs次
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            batches = [training_data[k:k+batch_size] for k in range(0, n, batch_size)]
            for batch in batches:
                self.update_batch(batch, learning_rate)
            print("Epoch {0} complete".format(j))
 
    def update_batch(self, batch, learning_rate):
        #根据一个batch中的训练样本，调整各个参数值
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in zip(batch[:-1], batch[-1:]) :
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        #计算梯度，并调整各个参数值
        self.weights = [w-(learning_rate/len(batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(learning_rate/len(batch))*nb for b, nb in zip(self.biases, nabla_b)]
 
    #反向传播
    def backprop(self, x, y):
        #保存b和w的偏导数值
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #正向传播
        activation = x
        #保存每一层神经元的激活值
        activations = [x]
        #保存每一层神经元的z值
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.act(z)
            activations.append(activation)
        #反向传播得到各个参数的偏导数值
        #公式(13)
        d = self.cost_derivative(activations[-1], y) * self.act_derivative(zs[-1])
        #公式(17)
        nabla_b[-1] = d
        #公式(14)
        nabla_w[-1] = np.dot(d, activations[-2].transpose())
        #反向逐层计算
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.act_derivative(z)
            #公式(36)，反向逐层求参数偏导
            d = np.dot(self.weights[-l+1].transpose(), d) * sp
            #公式(38)
            nabla_b[-l] = d
            #公式(37)
            nabla_w[-l] = np.dot(d, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
 
#距离函数的偏导数
def distance_derivative(output_activations, y):
    #损失函数的偏导数
    return 2*(output_activations-y)
 
# sigmoid函数
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
 
# sigmoid函数的导数
def sigmoid_derivative(z):
    return sigmoid(z)*(1-sigmoid(z))
 
if __name__ == "__main__":
    #创建一个5层的全连接神经网络，每层的神经元个数为13，8，5，3，1
    #其中第一层为输入层，最后一层为输出层
    network=NeuralNetwork([13,8,5,3,1],sigmoid,sigmoid_derivative,distance_derivative)
 
    #训练集样本
    # 获取数据
    train_data, test_data = load_data()
    x = train_data[:, :-1]
    y = train_data[:, -1:]
 
    #使用随机梯度下降算法（SGD）对模型进行训练
    #迭代5000次；每次随机抽取40个样本作为一个batch；学习率设为0.1
    # training_data = [(np.array([x_value]),np.array([y_value])) for x_value,y_value in zip(x,y)]
    network.SGD(train_data,10,40,0.1)

    #测试集样本
    x_test = test_data[:, :-1].T
    #测试集结果
    y_predict = network.feedforward(x_test)
 
    #图示对比训练集和测试集数据
    plt.plot(x,y,'r',x_test.T,y_predict.T,'*')
    plt.show()