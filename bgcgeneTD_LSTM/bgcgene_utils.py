#!/usr/bin/env python3
import torch
import pandas as pd
import numpy as np

def read_file_train(file_path):
    dataset = pd.read_csv(file_path)
    TDsentences = list(dataset['TDsentence'])
    TDsentences = list(map(lambda x: x.split(), TDsentences))
    TDlabels = list(dataset['TDlabels'])
    TDlabels = list(map(lambda x: list(map(int,x.split())), TDlabels))
    isBGC = list(dataset['isBGC'])
    for i in range(len(TDsentences)):
        if isBGC[i] == 'Yes':
            TDlabels[i].append(1)
        else:
            TDlabels[i].append(0)
    return TDsentences, TDlabels

def read_file_test(max_len,dataset):
    TDsentences = list(dataset['TDsentence'])
    TDsentences = list(map(lambda x: x.split(), TDsentences))
    TDlabels = list(dataset['TDlabels'])
    TDlabels = list(map(lambda x: list(map(int,x.split())), TDlabels))
    isBGC = list(dataset['isBGC'])
    for i in range(len(TDsentences)):
        # 补齐label长度至max_len
        if len(TDlabels[i]) < max_len:
            pad_len = max_len - len(TDlabels[i])
            zero_list = [0 for _ in range(pad_len)]
            TDlabels[i] = TDlabels[i] + zero_list
        else:
            TDlabels[i] = TDlabels[i][:max_len]
        if isBGC[i] == 'Yes':
            TDlabels[i].append(1)
        else:
            TDlabels[i].append(0)
    return TDsentences, TDlabels

def read_file_search(max_len,dataset):
    TDsentences = list(dataset['TDsentence'])
    TDsentences = list(map(lambda x: x.split(), TDsentences))
    return TDsentences

def evaluate_bgc(yp,yt):
    yp[yp >= 0.5] = 1
    yp[yp < 0.5] = 0
    corrects = torch.sum(torch.eq(yp,yt)).item()
    return corrects

def evaluate_gene(yp,yt):
    yp[yp >= 0.5] = 1
    yp[yp < 0.5] = -1
    corrects = torch.sum(torch.eq(yp,yt)).item()
    return corrects