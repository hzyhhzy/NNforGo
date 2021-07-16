import torch
import torch.nn as nn
import math
import torch.functional as F
import numpy as np

h1=torch.tensor(np.arange(9).reshape(1,1,3,3),requires_grad=True,dtype=torch.float32)
uf=nn.Unfold(3,padding=1)
fo=nn.Fold(output_size=(3,3),kernel_size=3,padding=1)
h=uf(h1)
#h1=h1[1:4,1:4,1:4]
print(h1,h)
h=fo(h)
print(h)
#
# print(h1)
#
# pad=nn.ZeroPad2d(1)
#
# h=pad(h1)
# print(h)
#
# h=torch.stack((h[0:-2,1:-1],h[2:,1:-1],h[1:-1,0:-2],h[1:-1,2:]),dim=1)
# print(h,h.shape)
#
# loss=torch.sum(h,dim=(0,1,2))
# loss.backward()
# print(h1.grad)
#
# h2=torch.tensor(np.arange(9).reshape(3,3),requires_grad=True,dtype=torch.float32)
#
# print(h1,h2,h1+h2)