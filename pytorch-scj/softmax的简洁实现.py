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
               data.DataLoader(mnist_test,batch_size,shuffle=False,num_workers=get_dataloader_workers()))
def init_weights(m):
    if type(m)==nn.Linear:#判断该层是不是线性层
        nn.init.normal_(m.weight,std=0.01)#其实是判断线性层
def train_epoch_ch3(net,train_iter,loss):
    for X,y in train_iter:
        l=loss(net(X),y)
        #由于这个loss函数比较特殊这里其实做了三步
        # 第一步是对X进行线性处理
        # 第二步对x进行softmax处理
        # 第三步是进行交叉熵损失的运算
        # 值得注意的是，这里没有求最大值，即与周边概率比较到底谁大，而是预测概率与真实概率进行比较
        # 这也符合对于softmax函数的预期，因为算法的底层逻辑是忽略其余概率，使预测概率不断接近真实概率
        # 而其实在这过程中对于另外概率的处理是无目的性的，从而达到让正确分类的概率远远大于其余分类
        trainer.zero_grad()
        l.backward()
        trainer.step()
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
if __name__ == '__main__':
    batch_size=256
    train_iter,test_iter=load_data_fashion_mnist(batch_size,7)
    net=nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    #其实就是神经网络的雏形，我第一层先展平，第二层做线性运算
    net.apply(init_weights)
    #等价代码
    #init_weights(net[1])
    #print(net[1].weight)
    #Flatten表示把0维度保留,另外的展成一维向量，相当于resize(X.reshape(-1,W.shape[0]))
    #nn.Linear()等效代码,线性层
    #num_inputs = 784  # 28*28
    #num_outputs = 10
    #W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    loss=nn.CrossEntropyLoss()
    #自动softmax运算运算后计算交叉熵损失
    #def cross_entropy(y_hat,y):
    #        return -torch.log(y_hat[range(len(y_hat)),y])
    trainer=torch.optim.SGD(net.parameters(),lr=0.1)
    #等价代码
    #def sgd(params, lr, batch_size):  # lr也叫做学习率
    #    with torch.no_grad():  # 更新的时候不要计算梯度
    #        for param in params:
    #            param -= lr * param.grad / batch_size
    #            param.grad.zero_()
    print(f"原始正确率{evaluate_accuracy(net, test_iter)}")
    for i in range(10):
            train_epoch_ch3(net, train_iter, loss)
            print(f"第{i + 1}次正确率{evaluate_accuracy(net, test_iter)}")



