with open('/mnt/usb2/data/corpus/corpus_mibig.txt', 'r') as f:
    corpus = f.readlines()
genome_dic = {}
for i, line in enumerate(corpus[:-2502]):
    genome_dic[i] = line.split()
BGC_dic = {}
for i, line in enumerate(corpus[-2502:]):
    BGC_dic[i] = line.split()
BGC_dic_lengtn = list(BGC_dic.values())
BGC_dic_lengtn = list(map(lambda x: len(x), BGC_dic_lengtn))
genome_length_dic = {}
for i in genome_dic:
    genome_length_dic[i] = len(genome_dic[i])

# 遍历所有的 genome
genome_length_dic_new = {}
for genome_id, genome_genes in genome_dic.items():
    indices_to_remove_set = set()
    # 构建基因计数器，记录BGC基因在 genome_genes 中的出现次数
    gene_counts = {gene: genome_genes.count(gene) for gene in genome_genes}
    # 遍历所有的 BGC
    for bgc_id, bgc_genes in BGC_dic.items():
        # 如果 bgc_genes 中的所有基因都在 genome_genes 中出现
        if all(gene_counts.get(gene, 0) > 0 for gene in bgc_genes):
            # 记录需要删除的基因索引
            indices_to_remove = set([i for i, gene in enumerate(genome_genes) if gene in bgc_genes])
            indices_to_remove_set = indices_to_remove_set.union(indices_to_remove)
            # 删除需要删除的基因
    indices_to_remove_list = list(indices_to_remove_set)
    for index in sorted(indices_to_remove_list, reverse=True):
        del genome_genes[index]
    # 更新 genome_dic 中的基因序列
    genome_length_dic_new[genome_id] = len(genome_genes)
    genome_genes_str_updated = " ".join(genome_genes)
    genome_dic[genome_id] = genome_genes_str_updated
with open('./corpus_mibig_deBGC_all.txt', 'w') as f:
    for i in range(len(genome_dic)):
        f.write(genome_dic[i]+'\n')
remove_length = {}
for i in genome_dic:
    remove_length[i]=genome_length_dic[i]-genome_length_dic_new[i]
print(f'去掉的长度：{remove_length}')