from typing import Iterable
import torch
import torch.nn as nn
import math
import torch.functional as F
from torch import randn

boardH = 19
boardW = 19


class CNNLayer(nn.Module):
    def __init__(self, in_c, out_c, stride=1, padding=1, kernel_size=3):
        super(CNNLayer, self).__init__()

        self.conv_net = nn.Sequential(
            nn.Conv2d(in_c,
                      out_c,
                      kernel_size,
                      stride=stride,
                      padding=padding,
                      dilation=1,
                      groups=1,
                      bias=False,
                      padding_mode='zeros'),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv_net(x)
        return x

class ResnetLayer(nn.Module):
    def __init__(self, inout_c, mid_c):
        super(ResnetLayer, self).__init__()
        self.conv_net = nn.Sequential(
            CNNLayer(inout_c, mid_c),
            CNNLayer(mid_c, inout_c)
        )

    def forward(self, x,x1=None,g=None):
        x = self.conv_net(x) + x
        return x

class Outputhead_v1(nn.Module):

    def __init__(self,out_c,head_mid_c):
        super(Outputhead_v1, self).__init__()
        self.cnn=CNNLayer(out_c, head_mid_c)
        self.valueHeadLinear = nn.Linear(head_mid_c, 2)
        self.policyHeadLinear = nn.Conv2d(head_mid_c, 1, 1)

    def forward(self, h):
        x=self.cnn(h)

        # value head
        value = x.mean((2, 3))
        value = self.valueHeadLinear(value)

        # policy head
        policy = self.policyHeadLinear(x)
        policy = policy.squeeze(1)

        return value, policy



class Model_v1(nn.Module):

    def __init__(self,b,f):
        super(Model_v1, self).__init__()
        self.model_name = "res1"
        self.model_size=(b,f)
        input_c =4

        self.inputhead=CNNLayer(input_c, f)
        trunk=[]
        for i in range(b):
            trunk.append(ResnetLayer(f,f))
        self.trunk=nn.Sequential(*trunk)
        self.outputhead=Outputhead_v1(f,f)

    def forward(self, x):
        h=self.inputhead(x)

        h=self.trunk(h)

        return self.outputhead(h)

ModelDic = {
    "res1": Model_v1
}
