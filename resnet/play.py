import tkinter as tk
import numpy as np
from dataset import boardW, boardH
import argparse
import numpy as np
import npnn as nn
import npF as F
from numpy.random import randn

MAX_NUM = 999999999
side = 1
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str,
                    default='cpu', help='cpu or cuda')
parser.add_argument('--model', type=str, help='model to test')

opt = parser.parse_args()

with open('model.py', 'r') as file:
    lines = file.read()
    lines = lines[lines.find('board'):]
    lines = lines.replace('torch.cat', 'np.concatenate')
    lines = lines.replace(', _ = torch.max', ' = np.max')
    lines = lines.replace('torch', 'np')
    # print(lines)
    exec(lines)

if 'npy' in opt.model:
    data = np.load(opt.model, allow_pickle=True).item()
elif 'json' in opt.model:
    with open(opt.model, 'r') as file:
        str = file.read()
        data = eval(str)
model = ModelDic[data['model_name']]()
model.load_state_dict(data['state_dict'])

window = tk.Tk()
window.geometry('1000x600')
window.configure(bg='gray')

board_self = np.zeros((boardH, boardW))
board_machine = np.zeros((boardH, boardW))
buttons = []


def detect_continuous(array, value):
    count = 0
    for i in range(len(array)):
        if array[i] == value:
            count += 1
        else:
            count = 0
        if count == 4:
            return True
    return False


def end_game():
    for c in range(boardW):
        for r in range(boardH):
            buttons[r][c].configure(state='disabled')


def is_win(board, row, col):
    min_row = row - min(row, col)
    min_col = col - min(row, col)
    max_row = row + min(boardH - row, boardW - col)
    max_col = col + min(boardH - row, boardW - col)
    # print(min_row, max_row, min_col, max_col)

    is_win = detect_continuous(
        board[np.arange(min_row, max_row), np.arange(min_col, max_col)], 1)

    min_row = row - min(row, boardW - 1 - col)
    max_col = col + min(row, boardW - 1 - col)
    max_row = row + min(boardH - 1 - row, col)
    min_col = col - min(boardH - 1 - row, col)
    # print(min_row, max_row, min_col, max_col)
    is_win |= detect_continuous(
        board[np.arange(min_row, max_row+1), np.flip(np.arange(min_col, max_col+1))], 1)
    is_win |= detect_continuous(board[row, :], 1)
    is_win |= detect_continuous(board[:, col], 1)
    return is_win


def step(col, row, is_machine):
    name = ['I', 'machine']
    color = ['black', 'white']
    board = [board_self, board_machine]
    result = ['win', 'lose']
    print(f"{name[is_machine]} put on {row}, {col}")
    buttons[row][col].configure(bg=color[is_machine])
    board[is_machine][row, col] = 1
    buttons[row][col].configure(state='disabled')
    if row > 0:
        buttons[row-1][col].configure(state='normal')
    if is_win(board[is_machine], row, col):
        print(result[is_machine])
        end_game()
        return True
    return False


def push(col, row):
    if step(col, row, False):
        return

    value, out = model(
        np.array([board_machine, board_self])[np.newaxis, :],
        np.array([[side]])
    )
    value = value.squeeze()
    value = F.softmax(value, dim=0)
    for i in range(len(valueLabel)):
        valueLabel[i].configure(
            text=f'{value_text[i]} rate : {value[i]*100:.1f}%')
    out = out.squeeze()
    out = F.softmax(out, dim=0)
    for c in range(boardW):
        s = f'{out[c]:.2f}'
        policyLabel[c].configure(text=s)
    out -= board_machine[0, :] * MAX_NUM
    out -= board_self[0, :] * MAX_NUM
    machine_col = out.argmax()
    points = np.argwhere(
        (board_machine+board_self)[:, machine_col] == 0).reshape((-1,))
    machine_row = points[-1]
    step(machine_col, machine_row, True)


for r in range(boardH):
    buttons.append([])
    buttons[r] = [0]*boardW

for r in range(boardH):
    for c in range(boardW):
        b = tk.Button(window,
                      command=lambda col=c, row=r: push(col, row),
                      bg='gray',
                      activebackground="black", state='disabled')
        b.grid(row=r, column=c,
               padx=10, pady=10, ipadx=10, ipady=10)
        buttons[r][c] = b
        if r >= boardH - 1:
            b.configure(state='normal')
valueLabel = []
value_text = ['win', 'lose', 'tie']
for i in range(3):
    valueLabel.append(tk.Label(window))
    valueLabel[i].grid(row=2+i, column=boardW, padx=30)
policyLabel = []
for c in range(boardW):
    policyLabel.append(tk.Label(window))
    policyLabel[c].grid(row=boardH, column=c)

window.mainloop()
