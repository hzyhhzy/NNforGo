
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


try:
    os.mkdir("saved_models")
except:
    pass
else:
    pass

try:
    os.mkdir("logs")
except:
    pass
else:
    pass


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
    parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
    parser.add_argument('--wd', type=float, default=0, help='weight decay')
    parser.add_argument('--data', type=str,
                        default='../data_shuffled.npz', help='trainset path')
    parser.add_argument('--type', type=str, default='res1',help='model type defined in model.py')
    parser.add_argument('--save', type=str , help='model save pth')
    parser.add_argument('--epoch', type=int,
                        default=1, help='epoch num')
    parser.add_argument('--maxstep', type=int,
                        default=5000000000, help='max step to train')
    parser.add_argument('--savestep', type=int,
                        default=1000, help='step to save')
    parser.add_argument('--infostep', type=int,
                        default=100, help='step to logger')
    parser.add_argument('-b', type=int,
                        default=6, help='block depth')
    parser.add_argument('-f', type=int,
                        default=64, help='block channels')
    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    device=opt.device
    batch_size=opt.bs
    lr=opt.lr
    trainset_path=opt.data
    model_type=opt.type
    save_name=opt.save
    maxepoch=opt.epoch
    maxstep=opt.maxstep
    savestep=opt.savestep
    infostep=opt.infostep
    blocks=opt.b
    filters=opt.f






    print("Loading data")
    myDataSet = trainset(trainset_path)
    print("Finished loading data")
    totalstep = 0
    file_path = f'saved_models/{save_name}.pth'

    if os.path.exists(file_path):
        data = torch.load(file_path)
        model_param=(data['model_size'][0],data['model_size'][1])
        model = ModelDic[data['model_name']](*model_param).to(device)
        model.load_state_dict(data['state_dict'])
        totalstep = data['step']
        print(f"loaded model: type={data['model_name']}, size={model.model_size}, totalstep={totalstep}")
    else:
        model = ModelDic[model_type](blocks,filters).to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr,momentum=0.9,weight_decay=0,nesterov=True)
    dataloader = DataLoader(myDataSet, shuffle=True, batch_size=batch_size)
    model.train()


    time0=time.time()
    loss_record=[0,0,0,1e-7]
    print("Start training")
    for epochs in range(maxepoch):
        for step, (board, valueTarget, policyTarget) in enumerate(dataloader):
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

            loss_record[0]+=vloss.detach()
            loss_record[1]+=ploss.detach()
            loss_record[2]+=(vloss.detach()+ploss.detach())
            loss_record[3]+=1

            loss.backward()
            optimizer.step()

            # logs
            totalstep += 1
            if(totalstep % infostep == 0):

                print(f"time: {time.time()-time0} s, step: {totalstep}, vloss: {loss_record[0]/loss_record[3]}, ploss: {loss_record[1]/loss_record[3]}, totalloss: {loss_record[2]/loss_record[3]}")

                logfile = open(f'logs/log_{save_name}.txt','a')
                print(f"{save_name} {totalstep} {loss_record[0]/loss_record[3]} {loss_record[1]/loss_record[3]}",file=logfile)
                logfile.close()

                loss_record = [0, 0, 0, 1e-7]
                time0=time.time()

            if totalstep in save_points:
                file_path_mid = f'saved_models/{save_name}_s{totalstep}.pth'
                print(f"Finished training {totalstep} steps")
                torch.save(
                    {'step': totalstep, 'state_dict': model.state_dict(), 'model_name': model.model_name,'model_size':model.model_size}, file_path_mid)
                print('Model saved in {}\n'.format(file_path_mid))

            if totalstep%savestep==0:
                print(f"Finished training {totalstep} steps")
                torch.save(
                    {'step': totalstep, 'state_dict': model.state_dict(), 'model_name': model.model_name,'model_size':model.model_size}, file_path)
                print('Model saved in {}\n'.format(file_path))

            if step >= maxstep:
                break

    print(f"Finished training {totalstep} steps")
    torch.save(
        {'step': totalstep, 'state_dict': model.state_dict(), 'model_name': model.model_name,
         'model_size': model.model_size}, file_path)
    print('Model saved in {}\n'.format(file_path))