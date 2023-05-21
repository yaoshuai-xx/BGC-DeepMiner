import os
import numpy as np
import pandas as pd
import pickle
import random

with open('/home/yaoshuai/data/corpus/corpus_mibig_deBGC_all.txt', 'r') as fp:
    concated_no_BGC_corpus = fp.read()
    concated_no_BGC_corpus = concated_no_BGC_corpus.split()
    print(f'去掉BGC后的corpus长度：{len(concated_no_BGC_corpus)}')
BGC_labels_dataset = pd.read_csv('/home/yaoshuai/tools/BGC_labels_pred/lstm_bgc/data/BGC_labels_dataset.csv')
BGC_origin_dataset = BGC_labels_dataset[:2502]

def generateInsertList(num, maxLen=128, notInsertProb:float=0.5, maxNonBGCgenes:int=3):
    '''
    生成插入到序列每个空位非BGC的基因个数，首尾也插入，有插入概率与长度的限制
    num: BGC序列长度
    notInsertPro: 该空位不插入非BGC基因的概率，此处为概率而非比例，有可能每个空位都插入也有可能每个空位均不插入
    maxNonBGCgenes: 一个空位插入非BGC基因的最大长度
    '''
    randomNonBGClen = list(range(0, maxNonBGCgenes+1))
    randomNonBGClenWeight = [notInsertProb, *[(1-notInsertProb)/maxNonBGCgenes]*maxNonBGCgenes]
    randomLen = random.choices(population=randomNonBGClen, weights=randomNonBGClenWeight, k=num-1)
    randomStart, randomEnd = [0], [0]
    leftLen = maxLen-num-sum(randomLen)
    randomStart = [0]
    randomEnd = [0]
    if leftLen>0:
        randomStart = random.randint(0, leftLen)
        randomEnd = [leftLen - randomStart]
        randomStart = [randomStart]
    return randomStart + randomLen + randomEnd


len_corpus = len(concated_no_BGC_corpus)
concated_no_BGC_corpus_plus = concated_no_BGC_corpus + concated_no_BGC_corpus[:128]

def getOneAugSeq(seq, maxLen, notInsertProb=0.8, maxNonBGCgenes=2):
    num = len(seq)
    insertList = generateInsertList(num=num, maxLen=maxLen, notInsertProb=notInsertProb, maxNonBGCgenes=maxNonBGCgenes)
    # print(insertList)
    random_start = random.randint(0, len_corpus)
    random_end = random_start + maxLen - num
    random_genes = concated_no_BGC_corpus_plus[random_start:random_end]
    trainSeq = []
    trainSeqLabels = []
    for i in range(len(insertList)):
        insert_seq_num = insertList[i]
        insert_seq = random_genes[:insert_seq_num]
        random_genes = random_genes[insert_seq_num:]
        trainSeq += insert_seq
        trainSeqLabels += [0]*insert_seq_num
        if i<num:
            trainSeq += [seq[i]]
            trainSeqLabels += [1]
    trainSeq = trainSeq[:maxLen]
    trainSeqLabels = trainSeqLabels[:maxLen]
    return trainSeq, trainSeqLabels

def getOneOriginSeq(seq, maxLen):
    num = len(seq)
    trainSeq = seq[:]
    trainSeqLabels = [1]*num
    random_start = random.randint(0, len_corpus)
    random_end = random_start + maxLen - num
    random_genes = concated_no_BGC_corpus_plus[random_start:random_end]
    trainSeq += random_genes
    trainSeqLabels += [0] * (maxLen-num)
    return trainSeq, trainSeqLabels

random.seed(42)
def generatePositiveSamples(dataset, maxLen, notInsertProb, maxNonBGCgenes, times):
    BGC_postive_dataset = pd.DataFrame(columns=['ID', 'sentence', 'labels', 'isBGC', 'TDsentence', 'TDlabels'])
    for i in range(len(dataset)):
        ID = dataset.iloc[i]['ID']
        seq = dataset.iloc[i]['sentence']
        labels = dataset.iloc[i]['labels']
        isBGC = dataset.iloc[i]['isBGC']
        trainSeq, trainSeqLabels = getOneOriginSeq(seq=seq, maxLen=maxLen)
        BGC_postive_dataset.loc[len(BGC_postive_dataset)] = [ID+'_0',seq, labels, isBGC, trainSeq, trainSeqLabels]

        for j in range(times):
            trainSeq, trainSeqLabels = getOneAugSeq(seq=seq, maxLen=maxLen, notInsertProb=notInsertProb, maxNonBGCgenes=maxNonBGCgenes)
            BGC_postive_dataset.loc[len(BGC_postive_dataset)] = [ID+f'_{j+1}',seq, labels, isBGC, trainSeq, trainSeqLabels]
    return BGC_postive_dataset

BGC_origin_dataset['sentence'] = BGC_origin_dataset['sentence'].apply(lambda x:x.split())
BGC_postive_dataset = generatePositiveSamples(dataset=BGC_origin_dataset, maxLen=128, notInsertProb=0.8, maxNonBGCgenes=2, times=10)
origin_index = [i*11 for i in range(len(BGC_origin_dataset))]
five_times_index_ = [[i*11+1,i*11+2,i*11+3,i*11+4,i*11+5] for i in range(len(BGC_origin_dataset))]
five_times_index = []
for l in five_times_index_:
    five_times_index += l

BGC_postive_dataset_5 = BGC_postive_dataset.iloc[origin_index+five_times_index]
BGC_postive_dataset_5 = BGC_postive_dataset_5.reset_index()


def getNegative(maxLen):
    random_start = random.randint(0, len_corpus)
    random_end = random_start + maxLen
    random_genes = concated_no_BGC_corpus_plus[random_start:random_end]
    trainSeq = random_genes[:]
    trainSeqLabels = [0] * maxLen
    return trainSeq, trainSeqLabels

BGC_negative_dataset = pd.DataFrame(columns=['ID', 'sentence', 'labels', 'isBGC', 'TDsentence', 'TDlabels'])
for i in range(len(BGC_postive_dataset)):
    trainSeq, trainSeqLabels = getNegative(maxLen=128)
    BGC_negative_dataset.loc[len(BGC_negative_dataset)] = [f'Negative_{i}',trainSeq, 'Unknown', 'No', trainSeq, trainSeqLabels]
BGC_TD_dataset_5 = pd.concat([BGC_postive_dataset_5, BGC_negative_dataset[:len(BGC_postive_dataset_5)]], ignore_index=True)
BGC_TD_dataset_5 = BGC_TD_dataset_5.drop(columns='index')
BGC_TD_dataset_5['sentence'] = BGC_TD_dataset_5['sentence'].apply(lambda x:' '.join(x))
BGC_TD_dataset_5['TDsentence'] = BGC_TD_dataset_5['TDsentence'].apply(lambda x:' '.join(x))
BGC_TD_dataset_5['TDlabels'] = BGC_TD_dataset_5['TDlabels'].apply(lambda x:[str(_) for _ in x]).apply(lambda x:' '.join(x))
BGC_TD_dataset_5.to_csv('./BGC_TD_dataset_5.csv')
BGC_TD_dataset_10 = pd.concat([BGC_postive_dataset, BGC_negative_dataset], ignore_index=True)
BGC_TD_dataset_10['sentence'] = BGC_TD_dataset_10['sentence'].apply(lambda x:' '.join(x))
BGC_TD_dataset_10['TDsentence'] = BGC_TD_dataset_10['TDsentence'].apply(lambda x:' '.join(x))
BGC_TD_dataset_10['TDlabels'] = BGC_TD_dataset_10['TDlabels'].apply(lambda x:[str(_) for _ in x]).apply(lambda x:' '.join(x))
BGC_TD_dataset_10.to_csv('./BGC_TD_dataset_10.csv')