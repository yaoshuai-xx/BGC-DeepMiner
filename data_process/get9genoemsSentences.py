import json
import pandas as pd
import pickle

# gene转化成ID
geno2pro_path = '/home/syao/data/validation_data/9genomes/ec_res_pro/geno2pro.json'
pro2id_dic_path = '/home/syao/data/data_process/pro2rep_id.pkl'
map_result_path = '/home/syao/tools/MMseqs2/mapping/9genomes_spup/bestResult.m8'
ec_out_path = '/home/syao/data/validation_data/9genomes/ec_res_pro/pro2ec_dic.json'
outdir = '/home/syao/data/validation_data/9genomes/sentence_out/'
label_path = "/home/syao/data/validation_data/9genomes/genome_bgc_list.pkl"

with open(geno2pro_path, 'r')as f:
    geno2pro_dic = json.load(f)
gene2id_dic = {}
label_dic = {}
for i in geno2pro_dic:
    for j in geno2pro_dic[i]:
        gene2id_dic[j] = 'Unknown'
        label_dic[j] = '0'
print('length of label_dic')
print(len(label_dic))
label_dic_keys1 = set(label_dic.keys())
# label_dic
with open(label_path,'rb')as fl:
    genome_bgc_list = pickle.load(fl)
    for i in genome_bgc_list:
        for j in i:
            label_dic[j] = '1'
print(len(label_dic))
label_dic_keys2 = set(label_dic.keys())
print(label_dic_keys2 - label_dic_keys1)
allgene_dic_keys1 = set(gene2id_dic.keys())
with open(pro2id_dic_path, 'rb')as f, open(map_result_path)as fc, open(ec_out_path)as fe:
    pro2id_dic = pickle.load(f)
    # 读取ClusterID
    for line in fc:
        pro_name = line.split()[0]
        tag_name = line.split()[1]
        try:
            gene2id_dic[pro_name] = pro2id_dic[tag_name]
        except:
            gene2id_dic[pro_name] = 'Unknown'
    print('length of gene2id_dic')
    print(len(gene2id_dic))
    allgene_dic_keys2 = set(gene2id_dic.keys())
    # 读取EC号
    pro2ec_dic = json.load(fe)
    for key, value in pro2ec_dic.items():
        gene2id_dic[key] = value
    print(len(gene2id_dic))
    allgene_dic_keys3 = set(gene2id_dic.keys())
print(allgene_dic_keys2 - allgene_dic_keys1)
print(allgene_dic_keys3 - allgene_dic_keys1)
# 保存gene2id
genes = list(gene2id_dic.keys())
ids = list(gene2id_dic.values())
gene2id_df = pd.DataFrame({'Gene':genes,'id':ids})
gene2id_df.to_csv(outdir+'gene2id.csv', index=False)
# 获得待预测数据
IDsentence_dic = {}
sentence_list = []
sentence_label_list = []
isbgc_list = []
genoem_label_list = []
for i in geno2pro_dic:# 逐个genome处理
    # 按照max_len进行切片,sentence
    gene_list = geno2pro_dic[i]
    for m in range(128,len(gene_list),128):
        tmp_list = gene_list[m-128:m]
        sentence_list.append(' '.join(tmp_list))
        tmp_label_list = []
        for n in tmp_list:
            tmp_label_list.append(label_dic[n])
        sentence_label_list.append(' '.join(tmp_label_list))
        if '1' in tmp_label_list:
            isbgc_list.append('Yes')
        else:
            isbgc_list.append('No')
    tmp_list = gene_list[m:]
    sentence_list.append(' '.join(tmp_list))
    for n in tmp_list:
        tmp_label_list.append(label_dic[n])
    sentence_label_list.append(' '.join(tmp_label_list))
    if '1' in tmp_label_list:
        isbgc_list.append('Yes')
    else:
        isbgc_list.append('No')
    # geneseq转换成IDseq
    new_pro_list = []
    for j in gene_list:
        new_pro_list.append(gene2id_dic[j])
    # 按照max_len进行切片,IDsentence
    count = 0
    for k in range(128,len(new_pro_list),128):
        tmp_ID_list = new_pro_list[k-128:k]
        sentence_name = f'{i[:15]}_{count}'
        IDsentence_dic[sentence_name] = ' '.join(tmp_ID_list)
        count += 1
    # 每个genome最后不够max_len的部分，模型数据处理时会使用padding补足
    tmp_ID_list = new_pro_list[k:]
    sentence_name = f'{i[:15]}_{count}'
    IDsentence_dic[sentence_name] = ' '.join(tmp_ID_list)
df = pd.DataFrame({'GenomeID':list(IDsentence_dic.keys()),'IDSentence':list(IDsentence_dic.values()),'INITsentence':sentence_list,'labels':sentence_label_list,'isBGC':isbgc_list})
df.to_csv(outdir+'9genomes.csv',index=False)

