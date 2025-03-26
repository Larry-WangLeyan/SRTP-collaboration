import math
import numpy as np
import torch
from torch import nn
from torch.utils import data
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
def train(train_features,test_features,train_labels,test_labels,num_epoches=400):
    loss=nn.MSELoss()
    input_shape=train_features.shape[-1]#一个一个样本的取
    net=nn.Sequential(nn.Linear(input_shape,1,bias=False))#最基本的线性层
    batch_size=min(10,train_labels.shape[0])
    train_iter=data.DataLoader(data.TensorDataset(train_features,train_labels.reshape(-1,1)),batch_size=batch_size,shuffle=True,num_workers=4)
    test_iter=data.DataLoader(data.TensorDataset(test_features,test_labels.reshape(-1,1)),batch_size=batch_size,num_workers=4)
    trainer=torch.optim.SGD(net.parameters(),0.1)
    for epoch in range(num_epoches):
        for X,y in train_iter:
            l=loss(net(X),y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        print(f"weight:{net[0].weight.data.numpy()},训练损失-测试损失：{evaluate_loss(net,train_iter,loss)-evaluate_loss(net,test_iter,loss)}")
if __name__=="__main__":
    max_degree=20
    n_train,n_test=100,100
    true_w=np.zeros(max_degree)
    true_w[0:4]=np.array([5,1.2,-3.4,5.6])
    # 除了前四项，后面项都是一些噪音
    features=np.random.normal(size=(n_train+n_test,1))#数据取的点是随机的，数据的x
    np.random.shuffle(features)
    poly_features=np.power(features,np.arange(max_degree).reshape(1,-1))
    #用广播机制，将x带入函数求出函数值
    for i in range(max_degree):
        poly_features[:,i]/=math.gamma(i+1)#防止样本的值太大
        #gamma(n)=(n-1)!
    labels=np.dot(poly_features,true_w)
    labels+=np.random.normal(scale=0.1,size=labels.shape)
    #正常
    #train_features = torch.tensor(poly_features[0:100,:4], dtype=torch.float32)
    #test_features = torch.tensor(poly_features[100:200,:4], dtype=torch.float32)
    #train_labels = torch.tensor(labels[0:100], dtype=torch.float32)
    #test_labels = torch.tensor(labels[100:200], dtype=torch.float32)
    #train(train_features, test_features,train_labels, test_labels, num_epoches=400)
    #欠拟合
    #train_features = torch.tensor(poly_features[0:100,:2], dtype=torch.float32)
    #test_features = torch.tensor(poly_features[100:200,:2], dtype=torch.float32)
    #train_labels = torch.tensor(labels[0:100], dtype=torch.float32)
    #test_labels = torch.tensor(labels[100:200], dtype=torch.float32)
    #train(train_features, test_features,train_labels, test_labels, num_epoches=400)
    # 过拟合
    train_features = torch.tensor(poly_features[0:100,:20], dtype=torch.float32)
    test_features = torch.tensor(poly_features[100:200,:20], dtype=torch.float32)
    train_labels = torch.tensor(labels[0:100], dtype=torch.float32)
    test_labels = torch.tensor(labels[100:200], dtype=torch.float32)
    train(train_features, test_features,train_labels, test_labels, num_epoches=1000)
