#!/usr/bin/env python3
import torch as pt
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from gensim.models.callbacks import CallbackAny2Vec
import datetime
from sklearn.model_selection import KFold
import random
import numpy as np
import os
import argparse
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from bgcgene_utils import *
from bgcgene_data import *
from bgcgene_model import *
from final_val import final_validation

start_time = datetime.datetime.now()
#参数设置
torch.autograd.set_detect_anomaly(True)
device = 'cuda' if pt.cuda.is_available() else 'cpu'
#requires_grad = False

# word2vec模型中自带callback()，换模型可以删除
#loss_list=[]
#class callback(CallbackAny2Vec):
#        '''Callback to print loss after each epoch.'''
#
#        def __init__(self):
#                self.epoch = 1
#
#        def on_epoch_end(self, model):
#                loss = model.get_latest_training_loss()
#                loss_list.append(loss)
#                print('Loss after epoch {}: {}'.format(self.epoch, loss))
#                self.epoch += 1

def train_step(train_loader, model, criterion, optimizer, max_len, epoch, pattern):
    model.train()#训练模式，使其可以进行前向传播和反向传播
    train_len = len(train_loader)

    for i, (inputs,labels) in enumerate(train_loader):
        inputs = inputs.to(device, dtype=pt.float)
        labels = labels.to(device, dtype=pt.float)
        optimizer.zero_grad()
        bgc_outputs,gene_outputs = model(inputs)
        bgc_outputs = bgc_outputs.squeeze(-1)# (batch_size, max_len)
        gene_outputs = gene_outputs.squeeze(-1)# (batch_size, max_len)
        # 计算loss
        bgc_labels = labels[:,-1]
        gene_labels = labels[:,:-1]
        bgc_loss = criterion(bgc_outputs,bgc_labels)
        gene_loss = criterion(gene_outputs,gene_labels)
        con_loss = (bgc_loss*max_len/8 + gene_loss)/2 # bgc_loss占50%
        # 计算acc
        bgc_correct = evaluate_bgc(bgc_outputs.clone().detach(), bgc_labels)
        gene_correct= evaluate_gene(gene_outputs.clone().detach(), gene_labels)
        gene_correct_all= evaluate_bgc(gene_outputs.clone().detach(), gene_labels)
        bgc_acc = bgc_correct/batch_size
        sumnum = torch.sum(gene_labels)
        if sumnum != 0:
            gene_acc = gene_correct/sumnum
        else:
             print('torch.sum(gene_labels)=0')
             gene_acc = 0
        allnum = gene_labels.numel()
        gene_acc_all = gene_correct_all/allnum
        #tatal_correct = (bgc_acc + gene_acc)/2
        if pattern == 'gene':
            gene_loss.backward()
        elif pattern == 'bgc':
            bgc_loss.backward()
        elif pattern == 'total':
            con_loss.backward()
        optimizer.step()

        if(i % 60 == 0):
            print('#Epoch:%d\t%d/%d\tcon_loss:%.5f\tbgc_loss:%.5f\tbgc_acc:%.3f\tgene_loss:%.5f\tgene_acc:%.3f\tgene_acc_all:%.3f' 
            % (epoch, i, train_len, con_loss.item(), bgc_loss.item(), bgc_acc*100, gene_loss.item(), gene_acc*100, gene_acc_all*100))

def validate_step(val_loader, model, criterion, max_len, pattern):
    model.eval()#进入评估模式，此时模型不会进行梯度计算，而是直接对输入数据进行前向传播，输出预测结果。
    val_len = len(val_loader)
    with pt.no_grad():
        bgc_total_loss, gene_total_loss, con_loss, bgc_total_correct, gene_total_correct, gene_total_correct_all, pnum , sumnum= 0, 0, 0, 0, 0, 0, 0, 0
        for i, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device, dtype=pt.float)
            labels = labels.to(device, dtype=pt.float)
            bgc_outputs,gene_outputs = model(inputs)
            bgc_outputs = bgc_outputs.squeeze(-1)# (batch_size)
            gene_outputs = gene_outputs.squeeze(-1)# (batch_size, max_len)
            bgc_labels = labels[:,-1]# (batch_size)
            gene_labels = labels[:,:-1]# (batch_size, max_len)
            bgc_loss = criterion(bgc_outputs,bgc_labels)
            gene_loss = criterion(gene_outputs,gene_labels)
            #loss = (bgc_loss + gene_loss)/2 # bgc_loss占50%
            #total_loss += loss.item()
            bgc_total_loss += bgc_loss.item()
            gene_total_loss += gene_loss.item()
            bgc_correct = evaluate_bgc(bgc_outputs.clone().detach(), bgc_labels)
            gene_correct = evaluate_gene(gene_outputs.clone().detach(), gene_labels)
            gene_correct_all = evaluate_bgc(gene_outputs.clone().detach(), gene_labels)
            #tatal_correct = (bgc_correct*max_len + gene_correct)/(2*max_len)
            bgc_total_correct += bgc_correct
            gene_total_correct += gene_correct
            gene_total_correct_all += gene_correct_all
            pnum += torch.sum(gene_labels)
            sumnum += gene_labels.numel()
        bgc_total_acc = bgc_total_correct*max_len/sumnum
        gene_total_acc = gene_total_correct/pnum
        gene_total_acc_all = gene_total_correct_all/sumnum
        con_loss += (bgc_total_loss*max_len/8 + gene_total_loss)/2

        if pattern == 'gene':
             result_loss = gene_total_loss/val_len
        elif pattern == 'bgc':
             result_loss = bgc_total_loss/val_len
        elif pattern == 'total':
             result_loss = con_loss/val_len

        print('#con_loss:%.5f\tbgc_loss:%.5f\tbgc_acc:%.3f\tgene_loss:%.5f\tgene_acc:%.3f\tall_gene_acc:%.3f' 
              % (con_loss/val_len, bgc_total_loss/val_len, 100*bgc_total_acc, gene_total_loss/val_len, 100*gene_total_acc, 100*gene_total_acc_all))
    return result_loss
    
def final_validate(val_loader, model, outpath):
    model.eval()#进入评估模式
    with pt.no_grad():
        bgc_outputs_all = np.empty(0)
        gene_outputs_all = np.empty(0)
        bgc_labels_all = np.empty(0)
        gene_labels_all = np.empty(0)
        for i, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device, dtype=pt.float)
            labels = labels.to(device, dtype=pt.float)
            bgc_outputs,gene_outputs = model(inputs)
            bgc_outputs = bgc_outputs.squeeze(-1)# (batch_size)
            gene_outputs = gene_outputs.squeeze(-1)# (batch_size, max_len)
            bgc_outputs=bgc_outputs.cpu().numpy()
            gene_outputs=gene_outputs.cpu().numpy().flatten()
            bgc_labels = labels[:,-1]# (batch_size)
            gene_labels = labels[:,:-1]# (batch_size, max_len)
            bgc_labels=bgc_labels.cpu().numpy()
            gene_labels=gene_labels.cpu().numpy().flatten()
            #print(f'bgc_outputs.shape={bgc_outputs.shape}\nbgc_labels.shape={bgc_labels.shape}\ngene_outputs.shape={gene_outputs.shape}\ngene_labels.shape={gene_labels.shape}')
            bgc_outputs_all = np.concatenate((bgc_outputs_all, bgc_outputs))
            gene_outputs_all = np.concatenate((gene_outputs_all, gene_outputs))
            bgc_labels_all = np.concatenate((bgc_labels_all, bgc_labels))
            gene_labels_all = np.concatenate((gene_labels_all, gene_labels))
    #print(f'bgc_outputs_all.shape={bgc_outputs_all.shape}\nbgc_labels_all.shape={bgc_labels_all.shape}\ngene_outputs_all.shape={gene_outputs_all.shape}\ngene_labels_all.shape={gene_labels_all.shape}')
    tmp = final_validation(bgc_labels_all, bgc_outputs_all, gene_labels_all, gene_outputs_all, outpath)
    tmp.result()

def train(train_loader, val_loader, model, criterion, max_len, batch_size, max_epochs, lr, gamma, dropout, num_layers, k, pattern):
    print(f'## {pattern} classification model is training !')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma, last_epoch=-1, verbose=True)
    best_loss = float("inf")
    flag = 0
    for epoch in range(max_epochs):
        train_step(train_loader, model, criterion, optimizer, max_len, epoch, pattern)
        print('validate set:')
        total_loss = validate_step(val_loader, model, criterion, max_len, pattern)
        print('train set:')
        total_loss_train = validate_step(train_loader, model, criterion, max_len, pattern)
        if(total_loss < best_loss):
             best_loss = total_loss
             #lr = scheduler.get_last_lr()[0]
             model_path = f'/home/yaoshuai/tools/lstm_bgc/model_save/bgcgeneTD/{pattern}/{train_pattern}_{max_len}_{batch_size}_{lr}v{gamma}_{dropout}_{num_layers}_f{k}.pt'
             print(f'# Save {pattern}-model fold{k} epoch{epoch} !')
             pt.save(model, model_path)
        else:
             flag+=1
             if flag>=5:# loss上升超过5个epoch就结束训练
                  break
        if scheduler.get_last_lr()[0]>0.00005:
            scheduler.step()
    outpath = f'/home/yaoshuai/tools/lstm_bgc/formal/result_save/train/ROC/{train_pattern}_{pattern}_{max_len}_{batch_size}_{lr}v{gamma}_{dropout}_{num_layers}_f{k}'
    final_validate(val_loader, model, outpath)
    print(f'# Path of definitive model: {model_path}')
    return model_path

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

#ef weight_init(m):
#   # 是否是线性层
#   if isinstance(m, nn.Linear):
#       nn.init.xavier_normal_(m.weight)
#       try:
#           nn.init.constant_(m.bias, 0)
#       except:
#           pass
#    # 是否为批归一化层
#   elif isinstance(m, nn.LayerNorm):
#       m.reset_parameters()


if __name__ == '__main__':
    Parser = argparse.ArgumentParser()
    Parser.add_argument('-m', '--max_len', help='max_len', default=128, type=int)
    Parser.add_argument('-b', '--batch_size', help='batch_size', default=32, type=int)
    Parser.add_argument('-e', '--max_epochs', help='nmax_epochs', default=50, type=int)
    Parser.add_argument('-l', '--lr', help='lr', default=8e-4, type=float)
    Parser.add_argument('-g', '--gamma', help='gamma', default=0.9, type=float)
    Parser.add_argument('-d', '--dropout', help='dropout', default=0.5, type=float)
    Parser.add_argument('-n', '--num_layers', help='num_layers', default=2, type=int)
    Parser.add_argument('-t', '--train_pattern', help='train_pattern', default='gbgb', type=str)
    Parser.add_argument('-f', '--file_path', help='file_path', default='/home/yaoshuai/tools/BGC_labels_pred/lstm_bgc/data/BGC_TD_dataset_10.csv', type=str)
    args = Parser.parse_args()
    # 设置参数
    max_len = args.max_len
    batch_size = args.batch_size
    max_epochs = args.max_epochs
    lr = args.lr
    gamma = args.gamma
    dropout = args.dropout
    num_layers = args.num_layers
    train_pattern = args.train_pattern
    file_path = args.file_path
    pattern_dic = {'b':'bgc','g':'gene','t':'total','B':'BGC','G':'Gene'}
    print(f'max_len = {max_len}\nbatch_size = {batch_size}\nmax_epochs = {max_epochs}\nlr = {lr}\ngamma = {gamma}\ndropout = {dropout}\nnum_layers = {num_layers}\ntrain_pattern = {train_pattern}')
    # 设置随机数种子
    setup_seed(1)
    # 预处理数据以及训练模型
    data_x, data_y = read_file_train(file_path)
    preprocess = DataPreprocess(data_x, data_y, max_len, w2vmodel='/home/yaoshuai/tools/lstmdemo/corpus_word2vec_skipgram/min3size200iter10neg20alpha-3/corpus_word2vec.sav')
    data_x = preprocess.sentences2embedding()
    data_y = preprocess.labels2tensor()
    model = LSTMNET(embedding_dim=200, hidden_dim=256, num_layers=num_layers, dropout=dropout, max_len=max_len).to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nStart training, parameter total:{}, trainable:{}\n'.format(total, trainable))
    criterion = nn.BCELoss(reduction='sum')

    # 设置kfold拆分器
    kf = KFold(n_splits=5,shuffle=True,random_state=1)
    for k,(train_index,test_index) in enumerate(kf.split(data_x,data_y)):
        print(f'## Fold--{k} ##')
        # 模型参数初始化
        if k >0:
            model = LSTMNET(embedding_dim=200, hidden_dim=256, num_layers=num_layers, dropout=dropout, max_len=max_len).to(device)
        x_train=data_x[train_index]
        x_test=data_x[test_index]
        y_train=data_y[train_index]
        y_test=data_y[test_index]
        train_dataset = BGCDataset(x_train,y_train)
        val_dataset = BGCDataset(x_test,y_test)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=5)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=5)
        ## 训练模型和测试
        for i, pattern in enumerate(train_pattern):
            if i == 0:
                model_path = train(train_loader, val_loader, model, criterion, max_len, batch_size, max_epochs, lr, gamma, dropout, num_layers, k, pattern=pattern_dic[pattern])
            else:
                model = pt.load(model_path)
                model_path = train(train_loader, val_loader, model, criterion, max_len, batch_size, max_epochs, lr, gamma, dropout, num_layers, k, pattern=pattern_dic[pattern])

        #model_path = '/home/yaoshuai/tools/lstm_bgc/model_save/bgcgeneTD/bgc/len128_batch_32_lr0.8vary0.00032_drop0.5_fold0.sav'
        print('# Final validation #')
        val_model = pt.load(model_path)
        outpath = f'/home/yaoshuai/tools/lstm_bgc/result_save/ROC/{train_pattern}_{max_len}_{batch_size}_{lr}v{gamma}_{dropout}_{num_layers}_f{k}'
        final_validate(val_loader, val_model, outpath)
        print(f'# Path of definitive model: {model_path}')
        print(val_model)

    end_time = datetime.datetime.now()
    print('总时间：', (end_time-start_time))
