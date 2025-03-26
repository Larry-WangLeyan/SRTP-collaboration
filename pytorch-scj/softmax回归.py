import torch
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
def softmax(X):
    X_exp=torch.exp(X)
    partition=X_exp.sum(1,keepdim=True)
    return X_exp/partition#广播机制
#softmax的例子
#X=torch.normal(0,1,(2,5))
#X_prob=softmax(X)
#print(X_prob,X_prob.sum(1))
def net(X):
    return softmax(torch.matmul(X.reshape(-1,W.shape[0]),W)+b)#这里执行了压缩，把图片都拉成了一条向量，本来的三维数据集变成了二维数据集
#花式索引
#y=torch.tensor([0,2])
#y_hat=torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5]])
#print(y_hat[[0,1],y])
#tensor([0.1000, 0.5000])
def cross_entropy(y_hat,y):
    return -torch.log(y_hat[range(len(y_hat)),y])
#print(cross_entropy(y_hat,y))
#tensor([2.3026, 0.6931])
def accuracy(y_hat,y):
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:
        y_hat=y_hat.argmax(axis=1)
    cmp=(y_hat.type(y.dtype)==y)
    return float(cmp.type(y.dtype).sum())
class Accumulator:
    def __init__(self,n):
        self.data=[0.0]*n
    def add(self,*args):#任意数量的参数
        self.data=[a+float(b) for a , b in zip(self.data,args)]
    def reset(self):
        self.data=[0.0]*len(self.data)
    def __getitem__(self,idx):
        return self.data[idx]
def evaluate_accuracy(net,data_iter):
    if isinstance(net,torch.nn.Module):
        net.eval()#评估模式，不计算梯度
    metric=Accumulator(2)
    for X,y in data_iter:
        metric.add(accuracy(net(X),y),y.numel())
    return metric[0]/metric[1]
def sgd(params,lr,batch_size):#lr也叫做学习率
  with torch.no_grad():#更新的时候不要计算梯度
    for param in params:
      param-=lr*param.grad/batch_size
      param.grad.zero_()
def updater(batch_size):
    return sgd([W,b],0.1,batch_size)
def train_epoch_ch3(net,train_iter,loss,updater):
    if isinstance(net,torch.nn.Module):
        net.train()#需要计算梯度
    metric=Accumulator(3)
    for X,y in train_iter:
        y_hat=net(X)
        l=loss(y_hat,y)
        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(float(1)*len(y),accuracy(y_hat,y),y.numel())
        else:
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()),accuracy(y_hat,y),y.numel())
    return metric[0]/metric[2],metric[1]/metric[2]
def train_ch3(net,train_iter,test_iter,loss,num_epochs,updater):
    print(f"原始的正确率为{evaluate_accuracy(net, test_iter)}")
    for epoch in range(num_epochs):
        train_epoch_ch3(net,train_iter,loss,updater)
        print(f"第{epoch+1}轮的正确率为{evaluate_accuracy(net,test_iter)}")
if __name__ == '__main__':
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size, 4, False)
    num_inputs = 784  # 28*28
    num_outputs = 10
    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)
    train_ch3(net,train_iter,test_iter,cross_entropy,100,updater)





