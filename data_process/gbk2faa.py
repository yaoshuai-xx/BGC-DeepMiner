import os
import re
import argparse
def process_gbk(out_path,file):
    file_name = file.split('/')[-1]
    print("处理文件："+file)
    with open(file, 'r') as f:
    # 读取DNA序列
        text = f.read() # string格式
        # 获取DNA序列
        DNA_len = int(re.search(r'[0-9]+ bp', text, flags = 0).group()[:-2])
        pattern = re.compile(r'ORIGIN[\s\S]+')
        DNA_seq_num = re.search(pattern, text, flags = 0).group()[7:]
        DNA_seq = re.sub(r'[0-9 \n/]*', '', DNA_seq_num)
        if(DNA_len != len(DNA_seq)):
            print("文件"+file+"出错：")
            print("DNA序列读取数目错误")

#         # 定义写入文件
        amino_acid_filename = (file_name).replace('.gbff', '.faa')
        output_dir_aa = out_path
        write_file_aa = open(output_dir_aa + '/' + amino_acid_filename, 'w')
        count = 0
        
        CDS_pattern = re.compile(r'     CDS             ')
        CDS_set_start = [substr.start() for substr in re.finditer(CDS_pattern, text)] + [text.find('ORIGIN')]
        for i in range(len(CDS_set_start)-1):
            count += 1
            CDS_block = text[CDS_set_start[i]:CDS_set_start[i+1]]
            translation_pattren = re.compile(r'/translation="([A-Z\s]+)"')
            try:
                translation = re.search(translation_pattren, CDS_block).group(1)
            except:
                print(count,'\n',CDS_block)
                continue
            translation = translation.replace(' ', '').replace('\n', '')
            locus_tag_pattern = re.compile(r'/locus_tag="(\w+)"')
            locus_tag = re.search(locus_tag_pattern, CDS_block).group(1)
            #写入文件
            write_file_aa.write(f'>{locus_tag}\n{translation}\n')
        print("已处理", count, "个基因")
        write_file_aa.close()
parser = argparse.ArgumentParser(description='Get faa file from gbk file.')
parser.add_argument('--gbk_path','-g', type=str, default='./gbk/', help='gbk file path')
parser.add_argument('--faa_path','-f', type=str, default='./faa/', help='faa file path')
args = parser.parse_args()
files_path = args.gbk_path

files_list = os.listdir(files_path)
print(files_list)
for file in files_list:
    file_path = files_path+file
    process_gbk(args.faa_path,file_path)
