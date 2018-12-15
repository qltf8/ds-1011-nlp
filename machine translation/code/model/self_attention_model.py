#the code is adapted from http://nlp.seas.harvard.edu/2018/04/03/attention.html

import numpy as np
import torch, torchtext
import torch.nn as nn
import torch.nn.functional as F
import math, copy, sys, os
from torch.autograd import Variable

class SingleHeadedAttention(torch.nn.Module):
    def __init__(self, d_k, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super().__init__()
        
        # We assume d_v always equals d_k
        self.d_k = d_k
        self.linears = clones(torch.nn.Linear(d_model, d_k), 2)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):

        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key = [l(x) for l, x in zip(self.linears, (query, key))]
        #query, key = batch, seq, d_k
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        #x: batch, seq_query, embedding_size
        
        
        return x
    
def attention(query, key, value, mask=None, dropout=None):
    '''
    query: batch, seq1, d_k
    key: batch, seq2, d_k
    value: batch, seq2, embedding_size
    mask: batch, 1, seq_2
    '''
    
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

    
class EncoderDecoder(torch.nn.Module):
    
    def __init__(self, d_model, d_k, num_encoder, num_decoder, src_vocab, tgt_vocab, dropout):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.dropout = dropout
        self.num_encoder = num_encoder
        self.num_decoder = num_decoder
        self.src_embedding = torch.nn.Embedding(src_vocab, d_model)
        self.tgt_embedding = torch.nn.Embedding(tgt_vocab, d_model)
        self.position_src = PositionalEncoding(d_model, dropout)
        self.position_tgt = PositionalEncoding(d_model, dropout)
        self.generator = torch.nn.Sequential(torch.nn.Linear(d_model, tgt_vocab), torch.nn.LogSoftmax(dim=-1))
    
    def forward(self, src, tgt, src_mask, tgt_mask):

        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.position_src(self.src_embedding(src)),src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):

        return self.decoder(self.position_tgt(self.tgt_embedding(tgt)), memory, src_mask, tgt_mask)
    
    def init_model(self):
        self.encoder = Encoder(self.num_encoder,  self.d_model, self.d_k, self.dropout)
        self.decoder = Decoder(self.num_decoder, self.d_model, self.d_k, self.dropout)
    

class PositionalEncoding(torch.nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        
        x = x + self.pe[:, :x.size(1)]
        
        return self.dropout(x)
    
    
class PositionwiseFeedForward(nn.Module):
    
    def __init__(self, d_model, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(F.relu(self.w_1(x)))
    
    

class Encoder(torch.nn.Module):
    def __init__(self, N, d_model, d_k, dropout):
        super().__init__()
        self.layers = clones(EncoderLayer(d_model, SingleHeadedAttention(d_k, d_model), PositionwiseFeedForward(d_model, dropout)), N)
    def forward(self, x, mask):
        
        for layer in self.layers:
            x = layer(x, mask)
            
        return x
        
        
class EncoderLayer(nn.Module):
    '''
    self_attn: multiheadattention
    feed_forward: PositionwiseFeedForward
    '''
    def __init__(self, d_model, self_attn, feed_forward):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm_1 = torch.nn.LayerNorm(d_model)
        self.norm_2 = torch.nn.LayerNorm(d_model)

    def forward(self, x, mask):
        
        x = self.self_attn(x, x, x, mask) + x
        x = self.norm_1(x)
        x = self.feed_forward(x) + x
        
        return self.norm_2(x)
    
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, d_model, self_attn, src_attn, feed_forward):
        super(DecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm_1 = torch.nn.LayerNorm(d_model)
        self.norm_2 = torch.nn.LayerNorm(d_model)
        self.norm_3 = torch.nn.LayerNorm(d_model)
        
 
    def forward(self, x, memory, src_mask, tgt_mask):
        
        m = memory
        x = self.self_attn(x, x, x, tgt_mask) + x
        x = self.norm_1(x)
        x = self.src_attn(x, m, m, src_mask) + x
        x = self.norm_2(x)
        x = self.feed_forward(x) + x
        
        return self.norm_3(x)
    
    
class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self,  N, d_model, d_k, dropout):
        super().__init__()
        self.layers = clones(DecoderLayer(d_model, SingleHeadedAttention(d_k, d_model),SingleHeadedAttention(d_k, d_model), 
                                          PositionwiseFeedForward(d_model, dropout)), N)
        
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x
    
    
def clones(module, N):
    "Produce N identical layers."
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])