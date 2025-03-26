import torch
from torch import nn
def dropout_layer(X,dropout):
    assert 0<=dropout<=1
    if dropout==1:
        return torch.zeeros_like(X)
    if dropout==0:
        return X
    mask=(torch.rand(X.shape)>dropout)
    return X[mask]
X=torch.tensor([1,2,3,4,5])
print(dropout_layer(X,0.5))