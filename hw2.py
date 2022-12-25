import pandas as pd
df=pd.read_csv("abalone.csv")

#1数据预处理
#data['Sex'][data['Sex']=='M']=1
#data['Sex'][data['Sex']=='F']=2
#data['Sex'][data['Sex']=='I']=3
df.replace({'Sex': 'M'}, 1, inplace=True)
df.replace({'Sex': 'F'}, 2, inplace=True)
df.replace({'Sex': 'I'}, 3, inplace=True)
#data.loc[0,'Sex']=1
print(df.head(10))

#2特征量分布
import matplotlib.pyplot as plt
print(df.describe())
#绘制直方图
df['Sex'].plot(kind='hist')
plt.show()
# 绘制箱形图
df.plot(kind='box')
plt.show()


#3数据集划分
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(df.iloc[:,0:8],df['Rings'],test_size=0.3)
#print(x_test.values)



#4回归
import torch
import torch.nn as nn


# 定义网络结构
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size,output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x=torch.relu(x)
        x=self.fc3(x)
        return x

#实例化模型，设置输入特征量的数量为 8，输出量的数量为 1
model = LinearRegression(8, 1)
# # 实例化模型，设置输入特征量的数量为 8，隐藏层的大小为 16，输出量的数量为 1
# model = MLP(8, 16, 1)
# 将模型移到 GPU 上
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# inputs = torch.Tensor(x_train.values)
# print(inputs)
# targets = torch.Tensor(y_train.values)
# print(targets)
# outputs = model(inputs)
# print(outputs)
# 训练模型
for epoch in range(10000):
    # 获取输入和目标输出
    inputs = torch.Tensor(x_train.values).to(device)
    targets = torch.Tensor(y_train.values).to(device)

    # 前向传播，计算预测值
    outputs = model(inputs)

    # 计算损失
    loss = criterion(outputs, targets)

    # 反向传播，更新模型参数
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 打印损失
    print(f'Epoch {epoch + 1}: loss = {loss.item():.4f}')

# 测试模型
inputs = torch.Tensor(x_test.values).to(device)
output = model(inputs)
#print(f'Prediction: {output.item():.4f}')

import numpy
print('测试RMSE:',numpy.sqrt(criterion(output,torch.Tensor(y_test.values).to(device)).item()))