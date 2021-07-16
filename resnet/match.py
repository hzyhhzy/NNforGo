import tkinter as tk
import numpy as np
from dataset import boardW, boardH
import argparse
import torch
import torch.nn as nn
import torch.functional as F
from numpy.random import randn
from model import ModelDic
import random

parser = argparse.ArgumentParser()
#parser.add_argument('--device', type=str,
#                    default='cpu', help='cpu or cuda')
parser.add_argument('-a', type=str, help='model 1 to test')
parser.add_argument('-b', type=str, help='model 2 to test')
parser.add_argument('--gamenum', type=int,default=100, help='game num')

opt = parser.parse_args()


COLOR_W=1
COLOR_B=2

board= np.zeros((boardH, boardW),dtype=np.uint8)




def is_win(x,y,side):
    row=y
    col=x
    def detect_continuous(array, value):
        count = 0
        for i in range(len(array)):
            if array[i] == value:
                count += 1
            else:
                count = 0
            if count == 5:
                return True
        return False
    min_row = row - min(row, col)
    min_col = col - min(row, col)
    max_row = row + min(boardH - row, boardW - col)
    max_col = col + min(boardH - row, boardW - col)
    # print(min_row, max_row, min_col, max_col)

    is_win = detect_continuous(
        board[np.arange(min_row, max_row), np.arange(min_col, max_col)], side)

    min_row = row - min(row, boardW - 1 - col)
    max_col = col + min(row, boardW - 1 - col)
    max_row = row + min(boardH - 1 - row, col)
    min_col = col - min(boardH - 1 - row, col)
    # print(min_row, max_row, min_col, max_col)
    is_win |= detect_continuous(
        board[np.arange(min_row, max_row+1), np.flip(np.arange(min_col, max_col+1))], side)
    is_win |= detect_continuous(board[row, :], side)
    is_win |= detect_continuous(board[:, col], side)
    return is_win

def isLegal(x,y):
    return board[y,x]==0

def play(x,y,color):
    if(isLegal(x,y)):
        board[y,x]=color
    else:
        print("illegal move")

def printboard():
    colorstr=['.','o','x']
    for y in range(boardH):
        for x in range(boardW):
            print(colorstr[board[y,x]],end=' ')
        print("")
    print("")


def genmove(model,color,temp=0):
    opp=3-color
    board_self=(board==color)
    board_opp=(board==opp)
    side=3-color*2

    boardFeature=torch.unsqueeze(torch.tensor([board_self,board_opp],dtype=torch.float32),0)
    model.eval()
    value, policy = model(boardFeature,model.model_size[2])
    value=value[-1]
    policy=policy[-1]
    value = value.detach().squeeze()
    value = torch.softmax(value, dim=0)

    policy = policy.detach().reshape(-1)
    policy=policy-100000*board.reshape(-1)
    temp=temp+0.0001
    policy=torch.softmax(policy/temp, dim=0).numpy()
    #print(np.sum(policy))
    move=np.random.choice(np.arange(boardW*boardH),p=policy)
    movex=move%boardW
    movey=move//boardW

    #printboard()
    #print(color,value)
    #print(policy)
    #print(move)
    return movex,movey

def run_a_game(modelB,modelW):
    maxMovenum=boardW*boardH
    global board
    board = np.zeros((boardH, boardW),dtype=np.uint8)

    def tempFunc(movenum):
        return 1*(0.5**(movenum/10))+0.2

    for movenum in range(maxMovenum):

        if(movenum==0):#第一手随即撒子
            x=random.randint(0,boardW-1)
            y=random.randint(0,boardH-1)
            play(x,y,COLOR_B)
            continue

        color=COLOR_W if movenum%2==1 else COLOR_B
        model=modelB if color==COLOR_B else modelW
        temp=tempFunc(movenum)
        x,y=genmove(model,color,temp)
        play(x,y,color)
        #printboard()
        if(is_win(x,y,color)):
            return color

    return 0


if __name__ == '__main__':

    def loadModel(path):
        file_path="saved_models\\"+path+".pth"
        data = torch.load(file_path)
        model = ModelDic[data['model_name']](*data['model_size'])
        model.load_state_dict(data['state_dict'])
        model.eval()
        return model

    model1 = loadModel(opt.a)
    model2=loadModel(opt.b)
    #以下都是对于1号model的结果
    winB=0
    lossB=0
    drawB=0
    winW=0
    lossW=0
    drawW=0

    for i in range(opt.gamenum//2):

        res=run_a_game(model1,model2)
        if(res==0):
            drawB+=1
        elif(res==COLOR_B):
            winB+=1
        elif(res==COLOR_W):
            lossB+=1

        res=run_a_game(model2,model1)
        if(res==0):
            drawW+=1
        elif(res==COLOR_W):
            winW+=1
        elif(res==COLOR_B):
            lossW+=1

        print("")
        print("\twin\tloss\tdraw")
        print(f"black\t{winB}\t{lossB}\t{drawB}")
        print(f"white\t{winW}\t{lossW}\t{drawW}")

