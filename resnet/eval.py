
from dataset import trainset
from model import ModelDic


import argparse
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import os
import time

save_points=[]


def cross_entropy_loss(output, target):
    t = torch.log_softmax(output,dim=1)
    loss = torch.mean(torch.sum(-t*target, dim=1), dim=0)
    return loss




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str,
                        default='0', help='which gpu')
    parser.add_argument('--device', type=str,
                        default='cuda', help='cpu or cuda')
    parser.add_argument('--bs', type=int,
                        default=256, help='batch size')
    parser.add_argument('--data', type=str,
                        default='../alldata_p1_v1.npz', help='trainset path')
    parser.add_argument('--save', type=str , help='model save pth')
    parser.add_argument('--epoch', type=int,
                        default=10000000, help='epoch num')
    parser.add_argument('--maxstep', type=int,
                        default=2000, help='max step to test')
    parser.add_argument('-r', type=int,
                        default=10, help='rnn repeat times')
    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    device=opt.device
    batch_size=opt.bs
    trainset_path=opt.data
    save_name=opt.save
    maxepoch=opt.epoch
    maxstep=opt.maxstep
    repeattimes=opt.r






    print("Loading data")
    myDataSet = trainset(trainset_path)
    print("Finished loading data")
    totalstep = 0
    file_path = f'saved_models/{save_name}.pth'

    data = torch.load(file_path)
    model = ModelDic[data['model_name']](*data['model_size']).to(device)
    model.load_state_dict(data['state_dict'])
    totalstep = data['step']
    print(f"loaded model: type={data['model_name']}, size={data['model_size']}, totalstep={totalstep}")
    totalstep=0

    dataloader = DataLoader(myDataSet, shuffle=True, batch_size=batch_size)
    model.train()


    time0=time.time()
    loss_record=np.zeros((repeattimes,3),np.float32)
    print("Start testing")
    for epochs in range(maxepoch):
        for step, (board, globalFeature, valueTarget, policyTarget) in enumerate(dataloader):
            # data
            board = board.to(device)
            globalFeature = globalFeature.to(device)
            valueTarget = valueTarget.to(device)
            policyTarget = policyTarget.to(device)

            values, policys = model(board, repeattimes)
            for i in range(repeattimes):
                value=values[i]
                policy=policys[i]
                vloss = cross_entropy_loss(value, valueTarget)
                ploss = cross_entropy_loss(policy.flatten(start_dim=1), policyTarget.flatten(start_dim=1))
                loss = 1.2*vloss+1.0*ploss
                loss_record[i,0]+=vloss.detach()
                loss_record[i,1]+=ploss.detach()
                loss_record[i,2]+=loss.detach()


            # logs
            totalstep += 1
            if(totalstep >= maxstep):

                print(f"time : {time.time()-time0} s, step : {totalstep}, loss_matrix:")
                print(loss_record/maxstep)
                exit(0)