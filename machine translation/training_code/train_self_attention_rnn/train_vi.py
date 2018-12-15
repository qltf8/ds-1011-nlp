
import numpy as np
import torch, torchtext
import torch.nn as nn
import torch.nn.functional as F
import math, copy, sys, os
from torch.autograd import Variable
import pickle
import csv
import re, random, string, subprocess, time


TEXT_vi = torchtext.data.ReversibleField(sequential=True, use_vocab=True, batch_first = True, tokenize= lambda t:t.split(),
                                        include_lengths=True)
TEXT_en = torchtext.data.ReversibleField(sequential=True, use_vocab=True, batch_first = False, tokenize= lambda t:t.split(),
                              lower=True, init_token='<sos>', eos_token='<eos>',include_lengths=True)
train_vi_en = torchtext.data.TabularDataset('/home/ql819/text_data/train_vi_en.csv', format='csv', 
                             fields=[('source',TEXT_vi),('target',TEXT_en)])
validation_vi_en = torchtext.data.TabularDataset('/home/ql819/text_data/dev_vi_en.csv', format='csv', 
                             fields=[('source',TEXT_vi),('target',TEXT_en)])


TEXT_vi.build_vocab(train_vi_en, min_freq=3)
TEXT_en.build_vocab(train_vi_en, min_freq=3)

train_vi_en_iter = torchtext.data.BucketIterator(train_vi_en, batch_size=1, sort_key= lambda e: len(e.source),
                             repeat = False, sort_within_batch=True, shuffle=True, device=torch.device(0))
validation_vi_en_iter = torchtext.data.BucketIterator(validation_vi_en, batch_size=1, sort_key= lambda e: len(e.source),
                             repeat = False, sort_within_batch=True, shuffle=True, device=torch.device(0))


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


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, src_embed, N):
        super(Encoder, self).__init__()
        self.src_embed = src_embed
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x):
        "Pass the input (and mask) through each layer in turn."
        x = self.src_embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
    
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer[1](x, self.feed_forward)
    

    
def attention(query, key, value, dropout=None):
    '''
    query: batch, seq1, d_k
    key: batch, seq2, d_k
    value: batch, seq2, embedding_size
    mask: batch, 1, seq_2
    '''
    
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, d_k, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        
        # We assume d_v always equals d_k
        self.d_k = d_k
        self.linears = clones(nn.Linear(d_model, d_k), 2)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value):
        "Implements Figure 2"
        
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key = [l(x) for l, x in zip(self.linears, (query, key))]
        #query, key = batch, seq, d_k
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value,
                                 dropout=self.dropout)
        #x: batch, seq_query, embedding_size
        
        
        return x
    
    
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(F.relu(self.w_1(x)))
    
    
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
    
    
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
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
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)
    

    
def make_model(src_vocab, tgt_vocab, N=6, 
               d_model=512, d_k=64, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(d_k, d_model)
    ff = PositionwiseFeedForward(d_model, dropout)
    position = PositionalEncoding(d_model, dropout)
    encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), nn.Sequential(Embeddings(d_model, src_vocab), c(position)), N)
    decoder = GRU_Decoder_With_Attention(num_vocab = tgt_vocab, input_size = d_model, hidden_size = d_model)
    for p in encoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return encoder, decoder



def train(encoder, decoder, optimizer, data_iter, teacher_forcing_ratio, batch_size = 64):

    encoder.train()
    decoder.train()
    
    count = 0
    loss = 0
    
    
    for batch in data_iter:
        
        
        source, target = batch.source, batch.target
        

        source_data,source_len = source[0], source[1]
        target_data,target_len = target[0], target[1]
        
        all_output = encoder(source_data)
        #all_output: 1, source_len, embedding_size

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        

        output = all_output[0,:]
        target_word_list = target_data.squeeze()
        target_word = torch.tensor([TEXT_en.vocab.stoi['<sos>']]).cuda(0)

        h_t = output[0,:]
        h_t = h_t.view([1,1,-1])

        is_init = True

        for word_index in range(1, target_len[0].item()):
            prob, h_t = decoder(target_word, h_t, output, is_init)
            is_init = False
            if use_teacher_forcing:
                target_word = target_word_list[[word_index]]
                loss += torch.nn.functional.nll_loss(prob, target_word)
            else:
                right_target_word = target_word_list[[word_index]]
                loss += torch.nn.functional.nll_loss(prob, right_target_word)
                predict_target_word_index = prob.topk(1)[1].item()

                if TEXT_en.vocab.stoi['<eos>'] == predict_target_word_index:
                    break
                else:
                    target_word = torch.tensor([predict_target_word_index]).cuda(0)
                    
        count += 1
        if count % batch_size == 0:
            
            loss = loss/batch_size
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            count = 0
            loss = 0
        
        
    if count % batch_size != 0:
        loss = loss/count
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        
        
class Bean_Search_Status_Record:
    
    def __init__(self, h_t, predict_word_index_list, sum_log_prob):
        self.h_t = h_t
        self.predict_word_index_list = predict_word_index_list
        self.sum_log_prob = sum_log_prob
        self.avg_log_prob = 0
        
    

def test(encoder, decoder, data_iter, k=10):
    encoder.eval()
    decoder.eval()

    path_name = '../eval/'+str(time.time()).replace('.','_')+'/'
    os.mkdir(path_name)

    predict_file_name = path_name + 'predict.txt'
    target_file_name = path_name + 'target_file_name.txt'

    predict_file = open(predict_file_name, 'w')
    target_file = open(target_file_name, 'w')


    for batch in data_iter:
        
        
        
        source, target = batch.source, batch.target
        

        source_data,source_len = source[0], source[1]
        target_data,target_len = target[0], target[1]
        
        all_output = encoder(source_data)
        output = all_output[0,:]
        
        target_word = torch.tensor([TEXT_en.vocab.stoi['<sos>']]).cuda(0)

        h_t = output[0,:]
        h_t = h_t.view([1,1,-1])

        is_init = True


        right_whole_sentence_word_index = target_data[1: target_len[0].item()-1,0]
        right_whole_sentence_word_index = list(right_whole_sentence_word_index.cpu().numpy())
        
        
        sequences = [Bean_Search_Status_Record(h_t, predict_word_index_list = [target_word], 
                                               sum_log_prob = 0.0)]
        
        t = 0
        while (t < 100):
            all_candidates = []
            for i in range(len(sequences)):
                record = sequences[i]
                h_t = record.h_t
                predict_word_index_list = record.predict_word_index_list
                sum_log_prob = record.sum_log_prob
                target_word = predict_word_index_list[-1]
                
                if TEXT_en.vocab.stoi['<eos>'] != target_word:
                
                    prob, h_t = decoder(torch.tensor([target_word]).cuda(0), h_t, output, is_init)

                    k_prob_value_list, k_word_index_list = prob.topk(k,dim=1)
                    k_prob_value_list = k_prob_value_list.cpu().detach().squeeze().numpy()
                    k_word_index_list = k_word_index_list.cpu().squeeze().numpy()
                    
                    
                    for prob_value, word_index in zip(k_prob_value_list, k_word_index_list):
                        prob_value = float(prob_value)
                        word_index = int(word_index)
                        new_record = Bean_Search_Status_Record(h_t, predict_word_index_list+[word_index], sum_log_prob+prob_value)
                        new_record.avg_log_prob = new_record.sum_log_prob/(len(new_record.predict_word_index_list) - 1)
                        all_candidates.append(new_record)
                else:
                    all_candidates.append(record)
            is_init = False
                        
            ordered = sorted(all_candidates, key = lambda r: r.sum_log_prob, reverse = True)
            sequences = ordered[:k]
            
            t += 1
        final_record = sequences[0]
        
        predict_whole_sentence_word_index = [TEXT_en.vocab.itos[temp_index] for temp_index in final_record.predict_word_index_list[1:-1]]
        right_whole_sentence_word_index = [TEXT_en.vocab.itos[temp_index] for temp_index in right_whole_sentence_word_index]

        predict_whole_sentence = ' '.join(predict_whole_sentence_word_index)
        right_whole_sentence = ' '.join(right_whole_sentence_word_index)

        predict_file.write(predict_whole_sentence.strip() + '\n')
        target_file.write(right_whole_sentence.strip() + '\n')


    predict_file.close()
    target_file.close()

    result = subprocess.run('cat {} | sacrebleu {}'.format(predict_file_name,target_file_name),shell=True,stdout=subprocess.PIPE)
    result = str(result)
    print(result)
    sys.stdout.flush()
    
    
    return get_blue_score(result)


def get_blue_score(s):
    a = re.search(r'13a\+version\.1\.2\.12 = ([0-9.]+)',s)
    return float(a.group(1))



def parameters_list_change_grad(encoder, decoder):
    para_list = []
    for name, data in list(encoder.named_parameters()):
        if 'src_embed' in name:
            data.requires_grad = False
        else:
            para_list.append(data)
            
    for name, data in list(decoder.named_parameters()):
        if 'embedding' in name:
            data.requires_grad = False
        else:
            para_list.append(data)
    return para_list        




encoder,decoder = make_model(src_vocab=len(TEXT_vi.vocab.stoi), tgt_vocab=len(TEXT_en.vocab.stoi), N=3, 
               d_model=512, d_k=64, dropout=0.1)

encoder = encoder.cuda(0)
decoder = decoder.cuda(0)

early_stop = 3
best_blue_score = -1
best_index = -1

save_model_dir_name = '../save_model/vi_to_en_'
teacher_forcing_ratio = 0.9

optimizer = torch.optim.Adam([{'params': encoder.parameters(), 'lr': 0.001},
                              {'params': decoder.parameters(), 'lr': 0.001}])


for index_unique in range(100):
    train(encoder, decoder, optimizer, train_vi_en_iter, teacher_forcing_ratio)
    blue_score = test(encoder, decoder, validation_vi_en_iter)
    print('epoch: ',index_unique, ' the blue score on validation dataset is : ', blue_score)
    sys.stdout.flush()
    if best_blue_score < blue_score:
        
        best_index = index_unique
        best_blue_score = blue_score
        torch.save(encoder, save_model_dir_name+'cnn_encode')
        torch.save(decoder, save_model_dir_name+'rnn_decoder')
        
    if index_unique - best_index >= early_stop:
        break




print('--------------------------------------')
sys.stdout.flush()


encoder = torch.load(save_model_dir_name+'cnn_encode')
decoder = torch.load(save_model_dir_name+'rnn_decoder')
        
        

para_list = parameters_list_change_grad(encoder, decoder)     
optimizer = torch.optim.Adam(para_list, lr = 0.001)  
save_model_dir_name = '../save_model/refined_vi_to_en_'

early_stop = 3
best_blue_score = -1
best_index = -1

for index_unique in range(100):
    train(encoder, decoder, optimizer, train_vi_en_iter, teacher_forcing_ratio)
    blue_score = test(encoder, decoder, validation_vi_en_iter)
    print('epoch: ',index_unique, ' the blue score on validation dataset is : ', blue_score)
    sys.stdout.flush()
    
    if best_blue_score < blue_score:
        
        best_index = index_unique
        best_blue_score = blue_score
        torch.save(encoder, save_model_dir_name+'cnn_encode_'+str(index_unique))
        torch.save(decoder, save_model_dir_name+'rnn_decoder_'+str(index_unique))
    if index_unique - best_index >= early_stop:
        break