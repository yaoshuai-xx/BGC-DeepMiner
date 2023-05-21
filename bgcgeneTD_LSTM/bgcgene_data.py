#!/usr/bin/env python3
import torch as pt
import numpy as np

from gensim.models import Word2Vec

class DataPreprocess:
    def __init__(self, sentences, labels, max_len, w2vmodel='./word2vec.sav'):
        self.sentences = sentences
        self.labels = labels
        self.sen_len = max_len
        self.w2v_path = w2vmodel
        self.index2word = []
        self.word2index = {}
        self.embedding_matrix = []
        self.word2vecModel = Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.word2vecModel.wv.vector_size

    def add_embedding(self):
        vector = pt.empty(1, self.embedding_dim)
        pt.nn.init.uniform_(vector)
        #self.word2index[word] = len(self.word2index)
        #self.index2word.append(word)
        #self.embedding_matrix =   pt.cat([self.embedding_matrix, vector], 0)

    def make_embedding(self):
        for i,word in enumerate(self.word2vecModel.wv.key_to_index):
            self.word2index[word] = len(self.word2index)
            self.index2word.append(word)
            self.embedding_matrix.append(self.word2vecModel.wv[word])
        self.embedding_matrix = np.array(self.embedding_matrix)
        self.embedding_matrix = pt.tensor(self.embedding_matrix)
        self.add_embedding("<PAD>")
        return(self.embedding_matrix)

    def pad_sequence(self, sentence, padding):
        if(len(sentence) > self.sen_len):
            sentence = sentence[:self.sen_len]
        else:
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(padding)
        assert(len(sentence) == self.sen_len)
        return(sentence)

    def sentences2embedding(self):
        padding = np.zeros(self.embedding_dim)
        sentences_embedding = []
        for i,item in enumerate(self.sentences):
            sentence_embedding = []
            for j,word in enumerate(item):
                if word in self.word2vecModel.wv:
                    sentence_embedding.append(self.word2vecModel.wv[word])
                else:
                    sentence_embedding.append(padding)
                    try:
                        self.labels[i][j] = 0
                    except:
                        pass
            sentence_embedding = self.pad_sequence(sentence_embedding, padding)
            sentence_embedding = np.array(sentence_embedding)
            sentences_embedding.append(sentence_embedding)
        sentences_embedding = np.array(sentences_embedding)
        return(pt.tensor(sentences_embedding, dtype=pt.float32))

    def labels2tensor(self):
        labels = np.array(self.labels)
        y = pt.tensor(labels, dtype=pt.float32)
        return(y)
    
    #def sentence_word2idx(self):
    #    sentence_list = []
    #    for i,item in enumerate(self.sentences):
    #        sentence_index = []
    #        for j,word in enumerate(item):
    #            if(word in self.word2index.keys()):
    #                sentence_index.append(self.word2index[word])
    #            else:
    #                sentence_index.append(self.word2index["<PAD>"])
    #                self.labels[i][j] = 0
    #        #sentence_index = self.pad_sequence(sentence_index)
    #        sentence_list.append(sentence_index)
    #    sentence_list = np.array(sentence_list)
    #    return(pt.LongTensor(sentence_list))

    #def add_label(self, labels):
    #    sentences_length = list(map(len, self.sentences))
    #    new_labels = []
    #    labels = list(map(int, labels))
    #    for i in range(labels):
    #        label = []
    #        if labels[i]==1:
    #            neg = [0 for j in range(self.sen_len-sentences_length[i])]
    #            pos = [1 for j in range(sentences_length[i])]
    #            label.append(neg)
    #            label.append(pos)
    #            label.append(1)
    #        elif labels[i]==0:
    #            label = [0 for j in range(self.sen_len)]
    #            label.append(0)
    #        new_labels.append(label)
    #    return new_labels



class BGCDataset(pt.utils.data.Dataset):
    def __init__(self, x, y=None):
        self.data = x
        self.label = y

    def __getitem__(self, index):
        if(self.label is None):
            return(self.data[index])
        return(self.data[index], self.label[index])

    def __len__(self):
        return(len(self.data))


