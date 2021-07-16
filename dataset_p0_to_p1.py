#import torch
from torch.utils.data import Dataset, DataLoader
#import torchvision.transforms as transforms
#import json
#import cv2
import os
#import string
import multiprocessing
import numpy as np

boardH =19
boardW = 19
dataL =19
assert(dataL == boardW)


def unpackBoardFeatures(packedData):
    # 原始的训练数据是二进制uint8格式
    # 棋盘100个格点（前70个有效，后30个为0），每8个数合成一个uint8，所以最后一维是100/8=13，需要拆开
    dataNum, featureNum, dataLen = np.shape(packedData)

    # 原来22个通道，大部分没用
    usefulChannels = [0, 1, 2, 6]  # onboard,my,opp,ko_ban
    packedData = packedData[:, usefulChannels, :]
    featureNum = len(usefulChannels)

    unpackedData = np.zeros(
        [dataNum, featureNum, dataLen, 8], dtype=np.uint8)
    for i in range(8):
        unpackedData[:, :, :, 7 - i] = packedData % 2
        packedData = packedData // 2
    unpackedData = np.reshape(
        unpackedData, ([dataNum, featureNum, dataLen * 8]))
    unpackedData = unpackedData[:, :, 0:boardW * boardH]
    unpackedData = np.reshape(
        unpackedData, ([dataNum, featureNum, boardH, boardW]))

    legalList=unpackedData[:,0,dataL-1,dataL-1]>0.5
    unpackedData=unpackedData[:,[1,2,3],:,:]
    return unpackedData,legalList


def unpackGlobalFeatures(d):

    # 原来19个通道，大部分没用
    #5 komi/20
    #6,7 ko_rule
    #9 scoring rule
    #10 tax rule
    #15 has_pda
    #17 button
    legalList=np.all([
    #    d[:,6]>0.5,
        d[:,9]<0.5,
        d[:,11]<0.5,
        d[:,15]<0.5,
        d[:,17]<0.5
    ],axis=0)
    return d[:, [5]],legalList


def unpackValueTarget(packedData):

    legalList=np.all([
        packedData[:,26]>0.5, # ifHasPolicyTarget
        packedData[:,2]<0.2
    ],axis=0)
    return packedData[:, [0, 1]],legalList  # win loss draw


def unpackPolicyTarget(packedData):
    #print("unpacking PolicyTarget")
    dataNum, featureNum, dataLen = np.shape(packedData)
    packedData = packedData[:, 0, 0:boardW * boardH]
    packedData = np.reshape(packedData, ([dataNum, boardH, boardW]))
    packedData = packedData+1e-7
    wsum = np.sum(packedData, axis=(1,2), keepdims=True)
    #print(f"\twsum.shape = {wsum.shape}")
    packedData = packedData/wsum
    return packedData.astype(np.float32)

def processData(loadpath):
    #处理单个文件
    data = np.load(loadpath)
    bf,ll1 = unpackBoardFeatures(data["binaryInputNCHWPacked"])
    gf,ll2 = unpackGlobalFeatures(data["globalInputNC"])
    vt,ll3 = unpackValueTarget(data["globalTargetsNC"])
    pt = unpackPolicyTarget(data["policyTargetsNCMove"])

    ll=np.all([ll1,ll2,ll3],axis=0)
    bf,gf,vt,pt=bf[ll],gf[ll],vt[ll],pt[ll]
    print(f"total rows {ll.shape[0]}, use rows {ll.sum()}")
    return bf,gf,vt,pt

def processAndSave(loadpath,savepath):

    bf,gf,vt,pt=processData(loadpath)
    np.savez_compressed(savepath,bf=bf,gf=gf,vt=vt,pt=pt)

def processDirThread(files,savedir,startID):
    i=0
    for f in files:
        savename=f"data_{i+startID}.npz"
        savename=os.path.join(savedir, savename)
        processAndSave(f,savename)
        i=i+1
        print(f"{i} of {len(files)}")

def processDir(loaddir,savedir,num_threads):

    try:
        os.mkdir(savedir)
    except:
        pass
    else:
        pass

    all_files=[]
    for (path,dirnames,filenames) in os.walk(loaddir):
        filenames = [os.path.join(path,filename) for filename in filenames if filename.endswith('.npz')]
        all_files.extend(filenames)

    print("Processing-------------------------------------------------------------------------")
    filenum=len(all_files)
    file_each_thread=filenum//num_threads
    start_ids=list(range(0,num_threads*file_each_thread,file_each_thread))
    end_ids=start_ids[1:]
    end_ids.append(filenum)
    print(start_ids,end_ids)
    all_file_split=[(all_files[start_ids[i]:end_ids[i]],savedir ,start_ids[i]) for i in range(num_threads)]
    print(all_file_split)
    with multiprocessing.Pool(num_threads) as pool:
        pool.starmap(processDirThread,all_file_split)



class trainset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)

        #通道0是自己的棋子，通道1是对手棋子  shape=(batchsize,2,15,15)
        self.boardFeature = unpackBoardFeatures(data["binaryInputNCHWPacked"])

        #通道0是黑白，黑是-1，白是1  shape=(batchsize,1)
        self.globalFeature = unpackGlobalFeatures(data["globalInputNC"])

        #终局结果，胜负和  shape=(batchsize,3)
        self.valueTarget = unpackValueTarget(data["globalTargetsNC"])

        #计算量分布（policy的目标值），shape=(batchsize,15,15)
        self.policyTarget = unpackPolicyTarget(data["policyTargetsNCMove"])

    def __getitem__(self, index):
        return self.boardFeature[index], self.globalFeature[index], self.valueTarget[index], self.policyTarget[index]

    def __len__(self):
        return self.valueTarget.shape[0]



def testDataloader():
    # test dataloader
    myDataSet = trainset("example.npz")
    dataloader = DataLoader(myDataSet, shuffle=False, batch_size=1)
    for id, (board, globalFeature, valueTarget, policyTarget) in enumerate(dataloader):
        print(f"batch : {id} : board : {board}, globalFeature : {globalFeature}, valueTarget : {valueTarget}, policyTarget : {policyTarget}")


if __name__ == '__main__':
    processDir("data_p0","data_p1",8)
