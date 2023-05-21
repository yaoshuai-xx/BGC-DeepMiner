#!/usr/bin/env python3
import torch as pt
from batch3Linear import batch3Linear

class LSTMNET(pt.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout=0.5, max_len=128):
        super(LSTMNET, self).__init__()
        #self.embedding = pt.nn.Embedding(embedding.size(0), embedding.size(1))
        #self.embedding.weight = pt.nn.Parameter(embedding, requires_grad=requires_grad)
        self.lstm = pt.nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.TD = TimeDistributed(FeatLayer(hidden_dim*2, 1, max_len), batch_first=True)
        # (batch_size, max_len, 2*hidden_dim)
        self.comp = pt.nn.Sequential(
            pt.nn.Dropout(dropout),
            pt.nn.Linear(2*hidden_dim, 64), # (batch_size, max_len, 64)
            pt.nn.Tanh(),
            pt.nn.Linear(64, 1),   # (batch_size, max_len, 1)
            pt.nn.Sigmoid()
        )
        self.reduce = pt.nn.Sequential(
            pt.nn.Linear(max_len, 32),
            pt.nn.Tanh(),
            pt.nn.Linear(32, 1),
            pt.nn.Sigmoid()
        )
    # LSTM初始化
    def _init_lstm_weight(self, weight):
        for w in weight.chunk(4, 0):
            pt.nn.init.xavier_uniform_(w)
    def _init_lstm(self):
        self._init_lstm_weight(self.lstm.weight_ih_l0)
        self._init_lstm_weight(self.lstm.weight_hh_l0)
        self.lstm.bias_ih_l0.data.zero_()
        self.lstm.bias_hh_l0.data.zero_()

    def forward(self, inputs):
        #inputs = self.embedding(inputs) #(batch_size, lenOfBGC, emb_dimen)
        x, _ = self.lstm(inputs, None)#取所有时间步的输出张量，忽略隐藏状态张量（batch_size, max_len, 2*hidden_dim）
        y = self.TD(x)
        x = self.comp(x)
        x = self.reduce(x.squeeze(-1))
        return(x, y)

class FeatLayer(pt.nn.Module):
    def __init__(self, dim_in, dim_out, max_len):
        super(FeatLayer, self).__init__()
        sizes = []
        factor = 8
        while dim_in > dim_out:
             sizes.append(dim_in)
             dim_in//=factor
        self.feat = pt.nn.Sequential(
                pt.nn.LayerNorm(sizes[0]),
                *[pt.nn.Sequential(batch3Linear(batch_size=max_len, in_features=sizes[i], out_features=sizes[i+1]), pt.nn.Sigmoid()) for i in range(len(sizes)-1)],
                batch3Linear(batch_size=max_len, in_features=sizes[-1], out_features=dim_out), #(batch_size, max_len, 1)
                pt.nn.Sigmoid()
                )

    def forward(self, x):
        return(self.feat(x))

class TimeDistributed(pt.nn.Module):
    def __init__(self, module, batch_first=False):
            super(TimeDistributed, self).__init__()
            self.module = module
            self.batch_first = batch_first
    def forward(self, x):
            if len(x.size()) <= 2:
                    return self.module(x)
            # 输入(max_len, batch_size, hidden_dim*2)
            x_reshaped = x.reshape(x.shape[1], x.shape[0], -1)
            y = self.module(x_reshaped)
            # 输出(max_len, batch_size, 1)
            # 还原原始形状
            y = y.reshape(x.shape[0], x.shape[1], -1)
            y = y.squeeze(-1)
            # (batch_size, max_len)
            return y

