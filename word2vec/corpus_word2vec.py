#!/usr/bin/env python3
import datetime
from gensim.models import word2vec
import logging
from gensim.models.callbacks import CallbackAny2Vec
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

loss_list=[]
class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self):
        self.epoch = 1

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_list.append(loss)
        print('Loss after epoch {}: {:.6f}'.format(self.epoch, loss))
        self.epoch += 1

def read_file(ifn):
  with open(ifn, 'r', encoding='utf-8') as f:
    l = f.readlines()
    x = [line.strip().split(' ') for line in l]
  return(x)

def train_word2vec(x):
  return(word2vec.Word2Vec(sentences=x, vector_size = 200, window=5, min_count=3, sg=1, epochs=10, alpha=0.001,seed=1001, workers=55, hs=0, negative=20, compute_loss=True, callbacks=[callback()]))

if(__name__ == '__main__'):
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  starttime = datetime.datetime.now()
  print('#loading data...')
  ifn = '/home/yaoshuai/data/corpus/corpus_mibig.txt'
  x = read_file(ifn)
  print('#training word2vec and transforming to vectors by skip-gram...')
  print('input data, for example:'+'\n', x[:1])
  model = train_word2vec(x)
  #saving in two format
  model.save('corpus_word2vec.sav')
  model.wv.save_word2vec_format('corpus_word2vec.vector')

  #detail loss info
  print('total loss: ', loss_list)#计算的是累积的loss
  loss_list.insert(0,0)#这一轮的累加和减去上一次的累加和就是该轮的loss，在list第一个插入0，方便计算第一轮loss
  y=[y-x for x, y in zip(loss_list, loss_list[1:])]
  print('Training Losses:')
  for i, loss in enumerate(y):
    print(f'Epoch {i+1}: {loss}')
  #loss figure
  mpl.rcParams["figure.figsize"] = (12,7)
  #用来正常显示中文标签
  #plt.rcParams["font.sans-serif"] = ['SimHei']

  x = np.arange(1,len(y)+1)
  plt.figure(figsize=(6,4))
  plt.plot(x,y,color="red",linewidth=1 )
  plt.xlabel(u'epoch')
  plt.ylabel(u'loss')
  plt.title("loss curve ")
  #plt.ylim(0,300000)# xlim、ylim：分别设置X、Y轴的显示范围。
  plt.savefig('loss figure.jpg',dpi=120,bbox_inches='tight')

  #select words to check
  try:
    select_words = ['EC:1.8.1.9','EC:6.1.1.17', 'EC:3.2.1.37', 'EC:5.4.99.25', 'EC:3.6.3.4', 'EC:3.6.3.25', 'EC:2.3.1.74', 'EC:6.2.1.3', 'Cluster_1169711', 'Cluster_0198348', 'Cluster_1903238' ,'Cluster_0525100']
    for select_word in select_words:
      print('Check word:'+select_word)  #获得词频
      print(model.wv.most_similar(select_word,topn = 20))
      print('------------------------------------------------')
  except KeyError as e:
    print(e)
  endtime = datetime.datetime.now()
  print(f'总时间：{endtime-starttime}')
  print('#training done!')
  #rmodel = word2vec.Word2Vec.load('corpus_word2vec.sav')
