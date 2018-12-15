import numpy as np
import torch, torchtext
import torch.nn as nn
import torch.nn.functional as F
import math, copy, sys, os
from torch.autograd import Variable
import pickle
import csv
import re, random, string, subprocess, time



class GRU_Decoder_With_Attention(torch.nn.Module):
    
    def __init__(self, num_vocab, input_size, hidden_size, dropout=0.1):
        super().__init__()
        self.num_vocab = num_vocab
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.dropout = dropout
        
        self.embedding_layer = torch.nn.Embedding(self.num_vocab, self.input_size)
        self.gru = torch.nn.GRU(hidden_size= self.hidden_size, input_size= self.input_size + 1 * self.hidden_size, 
                                  num_layers= self.num_layers)
        
        self.calcu_weight_1  = torch.nn.Linear(2*self.hidden_size, hidden_size)
        self.calcu_weight_2  = torch.nn.Linear(self.hidden_size, 1)
        self.init_weight = torch.nn.Linear(self.hidden_size, self.hidden_size)
        
        self.linear_vob = torch.nn.Linear(self.hidden_size, self.num_vocab)
        
    def forward(self, input_word_index, hidden_vector, encoder_memory, is_init = False):
        #input_word_index: [num]
        #hidden_vector: 1, 1, hidden_size
        #encoder_memory: source_sen_len , 1 * hidden_size
        
        if hidden_vector.shape[0] != self.num_layers or hidden_vector.shape[2] != self.hidden_size:
            raise ValueError('The size of hidden_vector is not correct, expect '+str((self.num_layers, self.hidden_size))\
                            + ', actually get ' + str(hidden_vector.shape))
        
        if is_init:
            hidden_vector = torch.tanh(self.init_weight(hidden_vector))
        
        
        n_hidden_vector = torch.stack([hidden_vector.squeeze()]*encoder_memory.shape[0],dim=0)
        com_n_h_memory = torch.cat([n_hidden_vector, encoder_memory], dim =1)
        com_n_h_temp = torch.tanh(self.calcu_weight_1(com_n_h_memory))
        
        
        weight_vector = self.calcu_weight_2(com_n_h_temp)
        weight_vector =  torch.nn.functional.softmax(weight_vector, dim=0)
        #weight_vector: source_sen_len * 1
        
        
        convect_vector = torch.mm(weight_vector.transpose(1,0), encoder_memory)
        #convect_vector: 1 , 2 * hidden_size
        
        
        input_vector = self.embedding_layer(input_word_index).view(1,1,-1)
        
        
        input_vector = torch.cat([convect_vector.unsqueeze(0), input_vector], dim=2)
        
        
        output, h_t = self.gru(input_vector,hidden_vector)
        output = output.view(1, self.hidden_size)
        
        
        prob = self.linear_vob(output)
        #prob 1, vob_size
        
        prob = torch.nn.functional.log_softmax(prob, dim=1)
        
        
        return prob, h_t
