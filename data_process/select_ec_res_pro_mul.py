import argparse
import multiprocessing
import os
from functools import partial
import json
import pandas as pd

def getECinfo(pathOfECPredicted,ECproteins):
    with open(pathOfECPredicted, "r") as f:
        proteins = {}
        for line in f:
            if "EC:" in line:
                protein = line.split()
                protein_name, protein_EC = protein[0], protein[1]
                proteins[protein_name] = protein_EC
    ECproteins.update(proteins)

def getSeqInfo(pathOfFaa,allProteins):
    with open(pathOfFaa, "r") as f:
        import re
        text = f.read()
        proteins = {}
        gene_list = []
        geno_name = pathOfFaa.split('/')[-1][:15]
        seqPattern = re.compile(r'>')
        seq_start = [substr.start() for substr in re.finditer(seqPattern, text)] + [len(text)]
        for i in range(len(seq_start)-1):
            seq_block = text[seq_start[i]:seq_start[i+1]]
            seq_lines = seq_block.split('\n')
            protein_name = seq_lines[0].split()[0][1:] # 只保留蛋白的AC号
            # 去除>
            protein_seq = seq_lines[1:]
            protein_seq = ''.join(protein_seq)
            proteins[protein_name] = protein_seq
            gene_list.append(protein_name)
    allProteins.update(proteins)
    return {geno_name:gene_list}

def getRes(ECProteins, allProteins):
    allProteinsSet = set(allProteins.keys()) # 含有基因组中所有蛋白的AC号
    ECProteinsSet = set(ECproteins.keys())
    resProteinsSet = allProteinsSet - ECProteinsSet
    print(len(allProteinsSet),'\n', len(ECProteinsSet),'\n', len(resProteinsSet))
    resProteinsSet = resProteinsSet
    resProteins = {}
    for pro in iter(resProteinsSet):
        resProteins[pro] = allProteins[pro]
    seqs = ''
    for pro, seq in resProteins.items():
        proBlock = '>' + pro + '\n' + seq + '\n'
        seqs += proBlock
    return seqs

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Select no EC protein'
    )
    parser.add_argument('--ecdirpath', '-e', required=True, help='Path of deepec-out directory')
    parser.add_argument('--faadirpath', '-f', default='', required=True, help='Path of faa directory')
    parser.add_argument('--outdir', '-o', default='./ec_res_pro/', required=False, help='Destination directory')
    args = parser.parse_args()

    ECPath_list = list(map(lambda x: args.ecdirpath+'/'+x, os.listdir(args.ecdirpath)))
    FaaPath_list = list(map(lambda x: args.faadirpath+'/'+x, os.listdir(args.faadirpath)))
    manager1 = multiprocessing.Manager()
    ECproteins = manager1.dict()
    allProteins = manager1.dict()
    pool = multiprocessing.Pool(processes=20)
    partial_getecdic = partial(getECinfo, ECproteins=ECproteins)
    partial_getprodic = partial(getSeqInfo, allProteins=allProteins)
    pool.map(partial_getecdic, ECPath_list)
    seq_list = pool.map(partial_getprodic, FaaPath_list)
    pool.close()
    pool.join()

    ECproteins = dict(ECproteins)
    allProteins = dict(allProteins)
    pros =list(allProteins.keys())
    seqs = list(allProteins.values())
    print('Saving files !')
    df = pd.DataFrame({'Protein':pros,'Sequence':seqs})
    #df.set_index('Protein')
    df.to_csv(args.outdir+'pro2seq.csv',index=False)
    geno2pro = {}
    with open(args.outdir+'pro2ec_dic.json','w')as f:
        json.dump(ECproteins,f)
    seqs = getRes(ECproteins, allProteins)
    with open(args.outdir+'ec_res_pro.faa', "w") as f:
        f.write(seqs)
    with open(args.outdir+'geno2pro.json', "w") as f:
        for i in seq_list:
            geno2pro.update(i)
        json.dump(geno2pro,f)
    
