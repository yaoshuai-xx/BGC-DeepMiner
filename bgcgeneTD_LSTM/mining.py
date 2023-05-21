#!/usr/bin/env python3
import torch as pt
import pandas as pd
from torch.utils.data import DataLoader
from bgcgene_utils import *
from bgcgene_data import *
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def final_validate(val_loader, model):
    model.eval()#进入评估模式
    with pt.no_grad():
        bgc_outputs_all = np.empty(0)
        gene_outputs_all = np.empty(0)
        for i, (inputs) in enumerate(val_loader):
            inputs = inputs.to(device, dtype=pt.float)
            bgc_outputs,gene_outputs = model(inputs)
            bgc_outputs = bgc_outputs.squeeze(-1)# (batch_size)
            gene_outputs = gene_outputs.squeeze(-1)# (batch_size, max_len)
            bgc_outputs=bgc_outputs.cpu().numpy()
            gene_outputs=gene_outputs.cpu().numpy().flatten()
            bgc_outputs_all = np.concatenate((bgc_outputs_all, bgc_outputs))
            gene_outputs_all = np.concatenate((gene_outputs_all, gene_outputs))
    gene_outputs_all_new = [int(i>=0.5) for i in gene_outputs_all]
    tmp_outputs_list = []
    for i in range(128, len(gene_outputs_all_new)+1, 128):
        tmp_outputs_list.append(' '.join(list(map(str,gene_outputs_all_new[i-128:i]))))
    df = pd.DataFrame({ "cluster_score" : bgc_outputs_all,"Gene_result": tmp_outputs_list})
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
    x = read_file_search(max_len,dataset)
    prepocess = DataPreprocess(x, None, max_len, w2vmodel='/home/yaoshuai/tools/lstmdemo/corpus_word2vec_skipgram/min3size200iter10neg20alpha-3/corpus_word2vec.sav')
    data = prepocess.sentences2embedding()
    test_dataset = BGCDataset(data)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    print('Testing loader prepared.')
    model = pt.load(model_path)
    result_path = '/home/yaoshuai/tools/lstm_bgc/formal/result_save/search/'+ out_factor + model_path.split('/')[-1].replace('.pt', '') + '.csv'
    result = final_validate(test_loader, model)
    dataset["Gene_result"] = result["Gene_result"]
    dataset["cluster_score"] = result["cluster_score"]
    print("Saving csv ...")
    dataset.to_csv(result_path, index=False)
    print("Predicting finished.")
