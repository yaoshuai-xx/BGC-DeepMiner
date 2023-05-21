import json
import pandas as pd
import pickle
import argparse

class getSentences:
    def __init__(self,geno2pro_dic,pro2id_dic) -> None:
        self.geno2pro_dic = geno2pro_dic
        self.pro2id_dic = pro2id_dic
        self.gene2id_dic = {}
        self.IDsentence_dic = {}
        self.sentence_list = []
        self.updateUnknown
    
    def updateUnknown(self):
        for i in self.geno2pro_dic:
            for j in self.geno2pro_dic[i]:
                self.gene2id_dic[j] = 'Unknown'

    def updateEC(self,ec_out_path):
        with open(ec_out_path,'r')as f:
            pro2ec_dic = json.load(f)
        for key, value in pro2ec_dic.items():
            self.gene2id_dic[key] = value
        print(f'Length of gene2id dictionary after updateEC: {len(self.gene2id_dic)}')
    
    def updateClstID(self,map_result_path):
        with open(map_result_path,'r')as f:
            for line in f:
                pro_name = line.split()[0]
                tag_name = line.split()[1]
                try:
                    self.gene2id_dic[pro_name] = self.pro2id_dic[tag_name]
                except:
                    self.gene2id_dic[pro_name] = 'Unknown'
        print(f'Length of gene2id dictionary after updating ClstID: {len(self.gene2id_dic)}')

    def save_gene2id(self,outdir):
        genes = list(self.gene2id_dic.keys())
        ids = list(self.gene2id_dic.values())
        gene2id_df = pd.DataFrame({'Gene':genes,'id':ids})
        gene2id_df.to_csv(outdir+'gene2id.csv', index=False)

    def update(self,ec_out_path,map_result_path):
        self.updateEC(ec_out_path)
        self.updateClstID(map_result_path)
    
    def generateSentences(self,outdir):
        for i in self.geno2pro_dic:
            # 按照max_len进行切片,sentence
            gene_list = self.geno2pro_dic[i]
            for m in range(128,len(gene_list),128):
                tmp_list = gene_list[m-128:m]
                self.sentence_list.append(' '.join(tmp_list))
            tmp_list = gene_list[m:]
            self.sentence_list.append(' '.join(tmp_list))
            # geneseq转换成IDseq
            new_pro_list = []
            for j in gene_list:
                new_pro_list.append(self.gene2id_dic[j])
            # 按照max_len进行切片,IDsentence
            count = 0
            for k in range(128,len(new_pro_list),128):
                tmp_ID_list = new_pro_list[k-128:k]
                sentence_name = f'{i[:15]}_{count}'
                self.IDsentence_dic[sentence_name] = ' '.join(tmp_ID_list)
                count += 1
            # 每个genome最后不够max_len的部分，模型数据处理时会使用padding补足
            tmp_ID_list = new_pro_list[k:]
            sentence_name = f'{i[:15]}_{count}'
            self.IDsentence_dic[sentence_name] = ' '.join(tmp_ID_list)
        df = pd.DataFrame({'GenomeID':list(self.IDsentence_dic.keys()),'TDsentence':list(self.IDsentence_dic.values()),'INITsentence':self.sentence_list})
        df.to_csv(outdir+'Aspergillus.csv',index=False)
        return df


def main(ec_out_path,map_result_path,geno2pro_path,pro2id_dic_path,outdir):
    with open(geno2pro_path, 'r')as fg, open(pro2id_dic_path, 'rb')as fp:
        geno2pro_dic = json.load(fg)
        pro2id_dic = pickle.load(fp)
    sentence = getSentences(geno2pro_dic,pro2id_dic)
    sentence.update(ec_out_path,map_result_path)
    sentence.save_gene2id(outdir)
    sentence.generateSentences(outdir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Integrate deepec and mapping results to generate sentences for model.')
    parser.add_argument('--ec_out_path','-e',required=True,help='path of deepec outfile')
    parser.add_argument('--map_result_path','-m',required=True,help='path of MMseq2 outfile')
    parser.add_argument('--geno2pro_path','-g',required=True,help='path of geno2pro.json')
    parser.add_argument('--pro2id_dic_path','-dic',default='',required=False,help='path of protein to representative ID dictionary')
    parser.add_argument('--outdir','-o',default='./sentence_out/',required=False,help='path of output directory')

    args = parser.parse_args()

    main(args.ec_out_path,args.map_result_path,args.geno2pro_path,args.pro2id_dic_path,args.outdir)