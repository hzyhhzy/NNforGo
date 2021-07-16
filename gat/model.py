from typing import Iterable
import torch
import torch.nn as nn
import torchvision.transforms as tt
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




class GATCNN_type1(nn.Module):
    def __init__(self, in_c, nhead, out_c,att_c,useMask=0,useRelu=1,useSelfAtt=0,usenormmean=0,useAtt=1,useAttActi=0,useFinalConv=0,useEarlyVrelu=0,use3x3Conv=0
                 ):
        super(GATCNN_type1, self).__init__()

        self._in_c=in_c
        self._nhead=nhead
        self._out_c=out_c
        self._att_c=att_c
        self._useMask=useMask
        self._useRelu=useRelu
        self._useSelfAtt=useSelfAtt
        self._usenormmean=usenormmean
        self._useAtt=useAtt
        self._useAttActi=useAttActi
        self._useFinalConv=useFinalConv
        self._useEarlyVrelu=useEarlyVrelu
        self._use3x3Conv=use3x3Conv

        if(use3x3Conv>0):
            self.Qconv=nn.Conv2d(in_c,nhead*att_c,3,padding=1)
            self.Kconv=nn.Conv2d(in_c,nhead*att_c,3,padding=1)
            self.Vconv=nn.Conv2d(in_c,nhead*out_c,3,padding=1)
        else:
            self.Qconv=nn.Conv2d(in_c,nhead*att_c,1)
            self.Kconv=nn.Conv2d(in_c,nhead*att_c,1)
            self.Vconv=nn.Conv2d(in_c,nhead*out_c,1)


        self.kvPad=nn.ZeroPad2d(1)
        self.bnQ=nn.BatchNorm2d(nhead*att_c)
        self.bnK=nn.BatchNorm2d(nhead*att_c)
        self.bnV=nn.BatchNorm2d(nhead*out_c)

        if(useFinalConv>0):
            self.finalConv=nn.Conv2d(nhead*out_c,nhead*out_c,3,padding=1)
            self.finalBN=nn.BatchNorm2d(nhead*out_c)

        p=5 if useSelfAtt>0 else 4
        self.att_mask=torch.zeros((p,boardH,boardW))
        self.att_mask[0,0,:]=-100
        self.att_mask[1,-1,:]=-100
        self.att_mask[2,:,0]=-100
        self.att_mask[3,:,-1]=-100
        self.att_mask=self.att_mask.reshape(p,1,boardH,boardW)

    def forward(self, x):
        Q=self.Qconv(x)
        K=self.Kconv(x)
        V=self.Vconv(x)
        Q=self.bnQ(Q)
        K=self.bnK(K)
        V=self.bnV(V)
        if(self._useEarlyVrelu):
            V=torch.relu(V)


        if(self._useAttActi>0):
            Q=torch.sigmoid(Q)

        #move one space towards -H,+H,-W,+W , so one node can contact its four neighbors
        Kpad=self.kvPad(K) #NCHW
        Vpad=self.kvPad(V) #NCHW
        if(self._useSelfAtt>0):
            K = torch.stack((Kpad[:,:,0:-2, 1:-1],
                            Kpad[:,:,2:, 1:-1],
                            Kpad[:,:,1:-1, 0:-2],
                            Kpad[:,:,1:-1, 2:],
                            K),dim=1).view(-1,5,self._nhead,self._att_c,boardH,boardW) #NphCHW
        else:
            K = torch.stack((Kpad[:,:,0:-2, 1:-1],
                            Kpad[:,:,2:, 1:-1],
                            Kpad[:,:,1:-1, 0:-2],
                            Kpad[:,:,1:-1, 2:]
                             ),dim=1).view(-1,4,self._nhead,self._att_c,boardH,boardW) #NphCHW

        # V = torch.stack((V[:,:,0:-2, 1:-1],
        #                   V[:,:,2:, 1:-1],
        #                   V[:,:,1:-1, 0:-2],
        #                   V[:,:,1:-1, 2:]),dim=1).view(-1,4,self._nhead,self._out_c,boardH,boardW) #NphCHW
        Q=Q.view(-1,1,self._nhead,self._att_c,boardH,boardW)
        Vpad=Vpad.view(-1,self._nhead,self._out_c,boardH+2,boardW+2)

        a=(Q*K).mean(3) #NphHW
        if(self._usenormmean==0):
            a=a*(self._att_c**0.5) #NphHW
            #similar to the formula of Transformer, but it only calculate attention with 4 neighbors


        if(self._useMask):
            att_mask=self.att_mask.to(a.device)
            a=a+att_mask #set outboard attention to -inf
        a=torch.softmax(a,dim=1).unsqueeze(dim=3)#NphHW
        #out=torch.einsum('npdhw , npdchw-> ndchw',a,V)

        #print(Q.shape,K.shape,V.shape,a[:,0].shape,V[:,:,0:-2, 1:-1].shape)


        #move one space towards -H,+H,-W,+W , so one node can contact its four neighbors
        #this step cost a lot of time
        if(self._useAtt):
            out=a[:,0]*Vpad[:,:,:,0:-2, 1:-1]\
                +a[:,1]*Vpad[:,:,:,2:, 1:-1]\
                +a[:,2]*Vpad[:,:,:,1:-1, 0:-2]\
                +a[:,3]*Vpad[:,:,:,1:-1, 2:]
        else:
            out=(Vpad[:,:,:,0:-2, 1:-1]\
                +Vpad[:,:,:,2:, 1:-1]\
                +Vpad[:,:,:,1:-1, 0:-2]\
                +Vpad[:,:,:,1:-1, 2:])/4
            #just to check whether attention is stupid


        if(self._useSelfAtt):#self loop
            out+=a[:,4]*V.view(-1,self._nhead,self._out_c,boardH,boardW)

        out=out.reshape(-1,self._nhead*self._out_c,boardH,boardW)
        if(self._useRelu):
            out=torch.relu(out)

        if(self._useFinalConv):
            out=self.finalConv(out)
            out=self.finalBN(out)
            out=torch.relu(out)

        out+=x #Res
        return out


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




class Model_gat1(nn.Module):

    def __init__(self,b,h,f,af,useMask=1,useRelu=1,useSelfAtt=0,usenormmean=0,useAtt=1,useAttActi=0,useFinalConv=0,useEarlyVrelu=0,use3x3Conv=0):
        super(Model_gat1, self).__init__()
        self.model_name = "gat1"
        self.model_size=(b,h,f,af,useMask,useRelu,useSelfAtt,usenormmean,useAtt,useAttActi,useFinalConv,useEarlyVrelu,use3x3Conv)
        input_c =4

        self.inputhead=CNNLayer(input_c, h*f)
        trunk=[]
        for i in range(b):
            trunk.append(GATCNN_type1(h*f,h,f,af,useMask,useRelu,useSelfAtt,usenormmean,useAtt,useAttActi,useFinalConv,useEarlyVrelu,use3x3Conv))
        self.trunk=nn.Sequential(*trunk)
        self.outputhead=Outputhead_v1(h*f,h*f)

    def forward(self, x):
        h=self.inputhead(x)

        h=self.trunk(h)

        return self.outputhead(h)


ModelDic = {
    "gat1": Model_gat1
}
