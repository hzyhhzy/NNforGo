#import torch
from torch.utils.data import Dataset, DataLoader

import os
import numpy as np


class trainset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.bf=data["bf"]
        self.gf=data["gf"]
        self.vt=data["vt"]
        self.pt=data["pt"]
        #print(f"Total {self.vt.shape[0]} rows")
    def __getitem__(self, index):


        bf1=self.bf[index].astype(np.float32)
        pt1=self.pt[index].astype(np.float32)
        assert (bf1.ndim==3)


        #concat bf and gf
        gf1=self.gf[index].astype(np.float32)
        gf1 = gf1.reshape((gf1.shape[0], 1, 1)).repeat(bf1.shape[1], axis=1).repeat(bf1.shape[2], axis=2)
        bf1 = np.concatenate((bf1, gf1), axis=0)

        vt1=self.vt[index].astype(np.float32)
        # if(len(gf1.shape)==1):
        #     #print("type1")
        #     gf1 = gf1.reshape((gf1.shape[0], 1, 1)).repeat(bf1.shape[1], axis=1).repeat(bf1.shape[2], axis=2)
        #     bf1 = np.concatenate((bf1, gf1), axis=0)
        # elif(len(gf1.shape)==2):
        #     #print("type2")
        #     gf1 = gf1.reshape((gf1.shape[0], gf1.shape[1], 1, 1)).repeat(bf1.shape[2], axis=2).repeat(bf1.shape[3], axis=3)
        #     bf1 = np.concatenate((bf1, gf1), axis=1)
        # else:
        #     print("Unknown index type")

        return bf1,vt1,pt1
    def __len__(self):
        return self.vt.shape[0]



