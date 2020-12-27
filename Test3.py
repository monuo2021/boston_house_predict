import keras
from sklearn.datasets import load_boston
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD,Adam
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

plt.rcParams['font.sans-serif'] = ['SimHei']

# 载入数据
house = load_boston()
x = house.data
y = house.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

#归一化
mean = x.mean(axis=0)
std = x.std(axis=0)
x_train -= mean
x_train /= std
x_test -= mean
x_test /= std

def build_model():
    model = Sequential()
    model.add(Dense(64, activation = 'sigmoid',input_shape = (x_train.shape[1],)))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(1))
    adam = Adam(lr = 0.003) # 优化器，lr为学习率
    # metrics: 列表，包含评估模型在训练和测试时的性能的指标
    model.compile(optimizer = adam, loss = 'mse', metrics = ['accuracy']) # optimizer = 优化器 均方误差（MSE）
    return model

model = build_model()
def train(iterations=100):
    losses = []
    for step in range(iterations):
        cost = model.train_on_batch(x_train,y_train)
        losses.append(cost)
        if (step+1) % 10 == 0:
            print('iter {}, loss {}'.format(step, cost))
    return losses

num_iterations = 2500
# 启动训练
losses = train(iterations=num_iterations)
# 画出损失函数的变化趋势
plot_x = np.arange(num_iterations)
plot_y = np.array(losses)
plt.plot(plot_x, plot_y)
plt.title("梯度下降") 
plt.xlabel('迭代次数', fontsize = 10)
plt.ylabel('损失函数', fontsize = 10)
plt.show()


for step in range(num_iterations):
    cost = model.train_on_batch(x_train,y_train)

#评估模型
test_loss,test_accuracy = model.evaluate(x_test,y_test)
print('loss:',test_loss)

y_pred = model.predict(x_test)
y_test = y_test.reshape(-1,1)
print(y_test.shape)
print(y_pred.shape)
y = np.concatenate((y_test,y_pred),axis=1)
print(y)



plt.plot(y_test ,"r--") 
plt.plot(y_pred,"b--") 
plt.title("红色为真实值，蓝色为预测值 ")  
plt.xlabel('数据个数', fontsize = 10)
plt.ylabel('房价', fontsize = 10)
plt.show()  