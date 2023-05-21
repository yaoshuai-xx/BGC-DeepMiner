import argparse
import os
from collections import defaultdict
import re
import multiprocessing

class DeduplicateEC:

    # @classmethod
    # def backup(cls, filepath):
    #     if os.path.exists(filepath+'.bak'):
    #         DeduplicateEC.backup(filepath+'.bak')
    #     os.rename(filepath, filepath+'.bak')
    
    def __init__(self, inputFolder, dstfolder):
        self.inputFolder = inputFolder
        self.name = inputFolder.split('/')[-2]
        self.dstfolder = dstfolder
    
    def __getECResults(self):
        with open(self.inputFolder+'/DeepEC_Result.txt', 'r') as f:
            text = f.readlines()[1:]
            proteins = {}
            duplicatePros = set()
            for line in text:
                protein = line.split()[0]
                ECNum = line.split()[1]
                if protein not in proteins:
                    proteins[protein] = ECNum
                else:
                    duplicatePros.add(protein)
        self.proteins = proteins
        self.duplicatePros = duplicatePros
    
    def deduplicate(self):
        self.__getECResults()
        duplProc = defaultdict(list)
        with open(self.inputFolder+'/log_files/4digit_EC_prediction.txt', 'r') as f:
            text = f.readlines()[1:]
            for line in text:
                sptlist = line.strip().split('\t')
                query = sptlist[0].strip()
                if '_SEPARATED_SEQUENCE_' in query:
                    query = query.split('_SEPARATED_SEQUENCE_')[0].strip()
                predicted_ec = sptlist[1].strip()
                dnn_activity = sptlist[2].strip()
                if predicted_ec != 'EC number not predicted':
                    # 在去重列表中的预测EC号和置信度都要保存下来以便取最高
                    if query in self.duplicatePros:
                        duplProc[query].append((predicted_ec, dnn_activity))
        for key, value in duplProc.items():
            duplProc[key] = sorted(value, key=lambda a: a[-1], reverse=True)
            new_key = re.sub(r'_1$','',key)
            self.proteins[new_key] = duplProc[key][0][0]
            self.proteins.pop(key)
        self.duplProc = duplProc    
    
    def getFile(self):
        self.deduplicate()
#         os.rename(self.inputFolder+'DeepEC_Result.txt', self.inputFolder+'DeepEC_Result.txt.bak')
        # DeduplicateEC.backup(self.inputFolder+'DeepEC_Result.txt')
        if not os.path.exists(self.dstfolder):
            os.makedirs(self.dstfolder, exist_ok=True)
        with open(self.dstfolder+self.name+'_DeepEC_Result.txt', 'w') as fp:
            fp.write('Query ID	Predicted EC number\n')
            for each_query in self.proteins:
                fp.write('%s\t%s\n'%(each_query, self.proteins[each_query]))

def process(fileFolderPath ,dstfolder):
    if os.path.exists(fileFolderPath):
        try:
            print(f"Process file:{fileFolderPath}!")
            a = DeduplicateEC(fileFolderPath, dstfolder)
            a.getFile()
        except Exception as e:
            print(f'Error!\n{fileFolderPath}\t{e}')
    else:
        print("File folder not exists:")
        print(fileFolderPath)
        return fileFolderPath

def main():
    parser = argparse.ArgumentParser(
        prog='Deduplicate',
        description='Deduplicate EC Number in DeepEC_Result.txt',
        epilog='Only for research!'
    )
    parser.add_argument('inputFolder', help='The folder you want to process.')
    parser.add_argument('--many', '-m', action='store_true', required=False, help=r"Process many files' folder or only one.")
    parser.add_argument('--keyword', '-k', default='', required=False, help='Specify key-word in folder name.')
    parser.add_argument('--dstfolder', '-d', default='./deduplicate_output/', required=False, help='Destination directory')
    
    args = parser.parse_args()

    sum = 0
    if args.many:
        pool = multiprocessing.Pool(processes=20)
        rootFolder = args.inputFolder
        for fileFolder in os.listdir(rootFolder):
            if args.keyword in fileFolder:
                fileFolderPath = rootFolder+fileFolder+'/'
                error_file = pool.apply_async(process, (fileFolderPath,args.dstfolder,))
        pool.close()
        pool.join()
        print("多线程执行完毕！")
        error_file = error_file.get()
        try:
            print(f'File folder not exists:{error_file}')
        except:
            pass
    else:
        a = DeduplicateEC(args.inputFolder, args.dstfolder)
        a.getFile()
        sum += 1
    print('Deduplicate done!')

if __name__=='__main__':
    main()

