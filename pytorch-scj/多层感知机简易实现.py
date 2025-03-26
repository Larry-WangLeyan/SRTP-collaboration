from itertools import accumulate

import torch
from torch import nn
import torchvision
from torch.utils import data
from torchvision import transforms
def load_data_fashion_mnist(batch_size,num_workers,resize=None,):
    '''下载数据集，并读取到内存中'''
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans=transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
            root=".../data", train=True, transform=trans, download=True
        )
    mnist_test = torchvision.datasets.FashionMNIST(
            root=".../data", train=False, transform=trans, download=True
        )

    def get_dataloader_workers():
        return num_workers
    return(data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()),
               data.DataLoader(mnist_test,batch_size,shuffle=True,num_workers=get_dataloader_workers()))
def init_weight(m):
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight,0,0.01)
class Accumulator:
    def __init__(self,n):
        Accumulator.data=[0.0]*n
    def add(self, *args):
        self.data=[a+float(b) for a,b in zip(self.data,args)]
    def reset(self):
        self.data=[0.0]*len(self.data)
    def __getitem__(self,idx):
        return self.data[idx]
def accuracy(y_hat,y):
    if len(y_hat.shape)>1 and y_hat.shape[0]>1:
        y_hat=y_hat.argmax(axis=1)
    correct=y_hat==y
    return float(correct.type(y.dtype).sum())
def evaluate_accuracy(net,data_iter):
    if isinstance(net,torch.nn.Module):
        net.eval()#评估模式，不计算梯度
    metric=Accumulator(2)
    for X,y in data_iter:
        metric.add(accuracy(net(X),y),y.numel())
    return metric[0]/metric[1]
if __name__ == '__main__':
     train_iter,test_iter=load_data_fashion_mnist(256,5,resize=None,)
     num_input,num_H1,num_H2,num_H3,num_output=784,256,128,256,10
     net=nn.Sequential(nn.Flatten(),nn.Linear(num_input,num_H1),nn.ReLU(),nn.Linear(num_H1,num_H2),nn.ReLU(),nn.Linear(num_H2,num_H3),nn.ReLU(),nn.Linear(num_H3,num_output))
     #outputfuture也可以理解为有几个w
     net.apply(init_weight)
     trainer=torch.optim.SGD(net.parameters(),0.05)
     loss=nn.CrossEntropyLoss()
     epoch=1000
     for i in range(epoch):
         for X,y in train_iter:
             l=loss(net(X),y)
             trainer.zero_grad()
             l.backward()
             trainer.step()
         print(f"第{i+1}的训练正确率是{evaluate_accuracy(net,train_iter)}，测试正确率是{evaluate_accuracy(net,test_iter)}")