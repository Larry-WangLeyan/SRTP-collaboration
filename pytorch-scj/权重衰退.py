import torch
from torch import nn
from torch.utils import data
def synthetic_data(w,b,num_examples):#创建数据集
  """生成y=Xw+b+噪声"""
  X=torch.normal(0,1,(num_examples,len(w)),dtype=torch.float32)
  y=torch.matmul(X,w)+b#临驾于mm和mv，mm只能用于两个二维矩阵相称，mv只能用于矩阵与向量
  y+=torch.normal(0,0.01,y.shape,dtype=torch.float32)
  return X,y.reshape((-1,1))#-1代表自动求值
  #转换成一个列向量
def init_weights(m):
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight,0,0.01)
def l2_penalty(w):
    return torch.sum(w.power(2))/2
#l2范数，惩罚权重退位机制
class Accumulator:
    def __init__(self,n):
        self.data=[0.0]*n
    def add(self,*args):#任意数量的参数
        self.data=[a+float(b) for a , b in zip(self.data,args)]
    def reset(self):
        self.data=[0.0]*len(self.data)
    def __getitem__(self,idx):
        return self.data[idx]
def evaluate_loss(net,data_iter,loss):
    metric=Accumulator(2)
    for X,y in data_iter:
        out=net(X)
        y=y.reshape(out.shape)#主要是为了防止行列引起的变化
        l=loss(out,y)
        metric.add(l.sum(),l.numel())
    return metric[0]/metric[1]#返回的是损失函数的平均值
if __name__ == '__main__':
    n_train,n_test,num_inputs,batch_size=20,100,200,10
    true_w,true_b=torch.ones((num_inputs,1))*0.01,0.05
    train_data=data.TensorDataset(*synthetic_data(true_w,true_b,n_train))
    #先创建X，X列长代表着有几个样本，行长代表着每个样本量化参数个数（与w的值相同）
    #再把X与W进行线性组合，使X的输入值通过线性组合变成一个标量，存储到y上
    #再最后给每个y加上噪音
    #*表示分别遍历取元组元素
    train_iter=data.DataLoader(train_data,batch_size=batch_size,shuffle=True)
    test_data=data.TensorDataset(*synthetic_data(true_w,true_b,n_test))
    test_iter=data.DataLoader(test_data,batch_size=batch_size,shuffle=False)
    net=nn.Sequential(nn.Linear(200,1))
    net.apply(init_weights)
    num_epochs,lr=100,0.001
    loss=nn.MSELoss()
    trainer=torch.optim.SGD([{"params":net[0].weight,"weight_decay":9},{"params":net[0].bias}],lr=lr)#weight_decay#惩罚率
    for epoch in range(num_epochs):
       for X,y in train_iter:
            l=loss(net(X),y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
       print(f"第{epoch+1}轮，训练损失是{evaluate_loss(net,train_iter,loss)}\n测试损失是{evaluate_loss(net,test_iter,loss)}"
             f"\n,训练损失-测试损失是{evaluate_loss(net,train_iter,loss)-evaluate_loss(net,test_iter,loss)}")

