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
def relu(X):
    a=torch.zeros_like(X)
    return torch.max(X,a)
def net(X):
    X=X.reshape((-1,num_inputs))#展平，第一层flatten
    H=relu(X@W1+b1)# 输入层处理，把输入的1*n的向量通过W（n*m的线性变化）（m种线性组合）
    # 变成1*m的隐藏层
    return (H@W2+b2)#输出层，把1*m的隐藏层通过W2（m*o）的矩阵变成o个标量组成的向量
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
    train_iter,test_iter=load_data_fashion_mnist(batch_size,6,resize=None,)
    num_inputs=784
    num_outputs=10
    num_hiddens=256
    #输入层(输入是一个num_inputs维度的向量,一共num_inputs个变量，1*num_inputs)
    W1=nn.Parameter(torch.randn(num_inputs,num_hiddens,requires_grad=True))
    #将输入转化为num_hiddens种线性组合
    #等效于W1=nn.Parameter(torch.randn(num_inputs,num_hiddens,requires_grad=True))
    b1=nn.Parameter(torch.zeros(num_hiddens,requires_grad=True))
    #b1=nn.Parameter(torch.zeros(num_hiddens,requires_grad=True))
    #输出层（通过对隐藏层的元素进行线性组合输出num_output个标量）
    W2=nn.Parameter(torch.randn(num_hiddens,num_outputs, requires_grad=True))
    b2=nn.Parameter(torch.zeros(num_outputs,requires_grad=True))
    params=[W1,b1,W2,b2]
    loss=nn.CrossEntropyLoss()
    # 自动softmax运算运算后计算交叉熵损失
    # def cross_entropy(y_hat,y):
    #        return -torch.log(y_hat[range(len(y_hat)),y]
    updater=torch.optim.SGD(params,0.1)
    for i in range(100):
        for X,y in train_iter:
            l=loss(net(X),y)
             #之前只有一层线性层没法接解决Xory逻辑的问题，现在可以了
             # 由于这个loss函数比较特殊这里其实做了三步
             # 第一步是对X进行线性处理
             # 第二步对x进行softmax处理
             # 第三步是进行交叉熵损失的运算
              # 值得注意的是，这里没有求最大值，即与周边概率比较到底谁大，而是预测概率与真实概率进行比较
              # 这也符合对于softmax函数的预期，因为算法的底层逻辑是忽略其余概率，使预测概率不断接近真实概率
              # 而其实在这过程中对于另外概率的处理是无目的性的，从而达到让正确分类的概率远远大于其余分类
            updater.zero_grad()
            l.backward()
            updater.step()
        print(f"{evaluate_accuracy(net,test_iter)}")



