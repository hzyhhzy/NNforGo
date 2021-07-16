
from torch.utils.data import Dataset, DataLoader
from dataset import trainset
from model import ModelDic
import argparse
import glob
import sys
import matplotlib.pyplot as plt
import torch.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import os
import time
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="0"


def depth_weight(d):
    return 1
def cross_entropy_loss(output, target):
    t = torch.log_softmax(output,dim=1)
    loss = torch.mean(torch.sum(-t*target, dim=1), dim=0)
    return loss


def train(batch_size, lr, device, myDataSet, model, maxstep):
    totalstep = 0
    #print(model.device)
    optimizer = optim.Adam(model.parameters(), lr)
    dataloader = DataLoader(myDataSet, shuffle=True, batch_size=batch_size)
    model.train()


    time0=0
    for step, (board,  valueTarget, policyTarget) in enumerate(dataloader):
        # data
        board = board.to(device)
        valueTarget = valueTarget.to(device)
        policyTarget = policyTarget.to(device)
        # optimize
        optimizer.zero_grad()
        value, policy = model(board)
        vloss = cross_entropy_loss(value, valueTarget)
        ploss = cross_entropy_loss(policy.flatten(start_dim=1), policyTarget.flatten(start_dim=1))
        loss = 1.2*vloss+1.0*ploss

        loss.backward()
        optimizer.step()

        # logs
        totalstep += 1
        if(totalstep  == 3):
            time0=time.time()
        if(totalstep  == 3+maxstep):
            time0=time.time()-time0
            break
    #print(f"{maxstep} batches, {time0} seconds, {maxstep/time0} batches/s, {batch_size*maxstep/time0} samples/s")
    speed=batch_size * maxstep / time0

    return speed



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str,
                        default='cuda', help='cpu or cuda')
    parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
    parser.add_argument('--data', type=str,
                        default='../20k.npz', help='trainset path')
    parser.add_argument('--type', type=str,default='res1', help='model type defined in model.py')
    parser.add_argument('--step', type=int,
                        default=5, help='max step to train')
    parser.add_argument('-s', type=int,
                        default=8, help='batchsize in range(s,e,j)')
    parser.add_argument('-e', type=int,
                        default=2060, help='batchsize in range(s,e,j)')
    parser.add_argument('-j', type=int,
                        default=8, help='batchsize in range(s,e,j)')

    parser.add_argument('-b', type=int,
                        default=6, help='block depth')
    parser.add_argument('-f', type=int,
                        default=64, help='block channels')


    opt = parser.parse_args()

    batchsize_x = []
    speed_y = []
    best_speed=0
    best_bs=0

    myDataSet = trainset(opt.data)
    model = ModelDic[opt.type](opt.b,opt.f).to(opt.device)
    for bs in range(opt.s,opt.e+1,opt.j):
        try:
            speed=train(batch_size=bs,
                  lr=opt.lr,
                  device=opt.device,
                  myDataSet=myDataSet,
                  model=model,
                  maxstep=opt.step
                  )
        except:
            print(f"Failed to test batchsize {bs}")
            print(f"Best batchsize is {best_bs}, speed is {best_speed}")
            break
        else:
            print(f"bs={bs},  {speed} samples/s, bestbs={best_bs}, bestspeed={best_speed}")
            batchsize_x.append(bs)
            speed_y.append(speed)
            if(speed>best_speed):
                best_speed=speed
                best_bs=bs


    plt.plot(batchsize_x,speed_y)
    np.savez(f"benchmark_{opt.type}_{opt.b}b{opt.f}f.npz",batchsize_x=np.array(batchsize_x),speed_y=np.array(speed_y))
    plt.show()