#!/usr/bin/env python3
import torch as pt
import pandas as pd
import sys
from torch.utils.data import DataLoader
from bgcgene_utils import *
from bgcgene_data import *
from final_val import final_validation
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    bgc_labels_all = bgc_labels_all.astype('int')
    bgc_outputs_all_new = [int(i>=0.5) for i in bgc_outputs_all]
    gene_labels_all = gene_labels_all.astype('int')
    gene_outputs_all_new = [int(i>=0.5) for i in gene_outputs_all]
    tmp_labels_list = []
    tmp_outputs_list = []
    for i in range(128, len(gene_labels_all)+1, 128):
        tmp_labels_list.append(' '.join(list(map(str,gene_labels_all[i-128:i]))))
        tmp_outputs_list.append(' '.join(list(map(str,gene_outputs_all_new[i-128:i]))))
    df = pd.DataFrame({"cluster_label": bgc_labels_all, "cluster_score" : bgc_outputs_all, "Gene_label": tmp_labels_list, "Gene_result": tmp_outputs_list})
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/mnt/hdd0/syao/lstm_bgc/result_save/model_save/9genomes_2021-08-11-15-47-47.pt', help='model path')
    parser.add_argument('--file_path', type=str, default='/mnt/hdd0/syao/data/validation_data/9genomes/sentence_out/9genomes.csv', help='file path')
    parser.add_argument('--max_len', type=int, default=128, help='max length of sentence')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--out_factor', type=str, required=True, help='output file factor')
    args = parser.parse_args()
    
    max_len = args.max_len
    batch_size = args.batch_size
    model_path = args.model_path
    file_path = args.file_path
    out_factor = args.out_factor
    dataset = pd.read_csv(file_path)
    x, y = read_file_test(max_len,dataset)
    prepocess = DataPreprocess(x, y, max_len, w2vmodel='/home/yaoshuai/tools/lstmdemo/corpus_word2vec_skipgram/min3size200iter10neg20alpha-3/corpus_word2vec.sav')
    data = prepocess.sentences2embedding()
    labels = prepocess.labels2tensor()
    test_dataset = BGCDataset(data, labels)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    print('Testing loader prepared.')

    model = pt.load(model_path)
    outpath = '/home/yaoshuai/tools/lstm_bgc/formal/result_save/test/ROC/'+ out_factor + model_path.split('/')[-1].replace('.pt', '')
    result_path = '/home/yaoshuai/tools/lstm_bgc/formal/result_save/test/file/'+ out_factor + model_path.split('/')[-1].replace('.pt', '') + '.csv'
    result = final_validate(test_loader, model, outpath)
    dataset["Gene_result"] = result["Gene_result"]
    dataset["cluster_score"] = result["cluster_score"]
    print("Saving csv ...")
    dataset.to_csv(result_path, index=False)
    print("Predicting finished.")
