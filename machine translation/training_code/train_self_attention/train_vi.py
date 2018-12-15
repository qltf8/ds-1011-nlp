
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
TEXT_en = torchtext.data.ReversibleField(sequential=True, use_vocab=True, batch_first = True, tokenize= lambda t:t.split(),
                              lower=True, init_token='<sos>', eos_token='<eos>',include_lengths=True)


train_vi_en = torchtext.data.TabularDataset('../data/processed_data/train_vi_en.csv', format='csv', 
                             fields=[('source',TEXT_vi),('target',TEXT_en)])
validation_vi_en = torchtext.data.TabularDataset('../data/processed_data/dev_vi_en.csv', format='csv', 
                             fields=[('source',TEXT_vi),('target',TEXT_en)])


TEXT_vi.build_vocab(train_vi_en, min_freq=3)
TEXT_en.build_vocab(train_vi_en, min_freq=3)


train_vi_en_iter = torchtext.data.BucketIterator(train_vi_en, batch_size=4, sort_key= lambda e: len(e.source) + len(e.target),
                             repeat = False, sort_within_batch=True, shuffle=True, device=torch.device(0))
validation_vi_en_iter = torchtext.data.BucketIterator(validation_vi_en, batch_size=1, sort_key= lambda e: len(e.source) + len(e.target),
                             repeat = False, sort_within_batch=True, shuffle=True, device=torch.device(0))



class SingleHeadedAttention(torch.nn.Module):
    def __init__(self, d_k, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super().__init__()
        
        self.d_k = d_k
        self.linears = clones(torch.nn.Linear(d_model, d_k), 2)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):

        nbatches = query.size(0)
        
        query, key = [l(x) for l, x in zip(self.linears, (query, key))]
        #query, key = batch, seq, d_k
        
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

    
def clones(module, N):
    "Produce N identical layers."
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


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


    
    
class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.sum() > 0 and len(mask) > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))
    
    
class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion
        
    def __call__(self, x, y):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1))
        return loss
    
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
    



class Bean_Search_Status_Record:
    
    def __init__(self, predict_word_index_list, sum_log_prob):
        self.predict_word_index_list = predict_word_index_list
        self.sum_log_prob = sum_log_prob
        self.avg_log_prob = 0
        
    

def test(model, data_iter, k=5):
    model.eval()

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
        
        source_mask = (source_data != TEXT_vi.vocab.stoi['<pad>']).unsqueeze(1)
        
        memory = model.encode(source_data, source_mask)
        
        target_word = TEXT_en.vocab.stoi['<sos>']


        right_whole_sentence_word_index = target_data[0, 1: target_len[0].item()-1]
        right_whole_sentence_word_index = list(right_whole_sentence_word_index.cpu().numpy())
        
        
        sequences = [Bean_Search_Status_Record(predict_word_index_list = [target_word], 
                                               sum_log_prob = 0.0)]
        
        t = 0
        while (t < 100):
            all_candidates = []
            for i in range(len(sequences)):
                record = sequences[i]
                predict_word_index_list = record.predict_word_index_list
                predict_word_index_list_tensor = torch.tensor(predict_word_index_list).view(1,-1).type_as(source_data)
                sum_log_prob = record.sum_log_prob
                last_word_index = predict_word_index_list[-1]
                
                if TEXT_en.vocab.stoi['<eos>'] != last_word_index:
                
                    out = model.decode(memory, source_mask, 
                                       Variable(predict_word_index_list_tensor), 
                                       Variable(subsequent_mask(predict_word_index_list_tensor.size(1))
                                                .type_as(source_data)))
                    prob = model.generator(out[:, -1])
        
                    k_prob_value_list, k_word_index_list = prob.topk(k,dim=1)
                    k_prob_value_list = k_prob_value_list.cpu().detach().squeeze().numpy()
                    k_word_index_list = k_word_index_list.cpu().squeeze().numpy()
                    
                    
                    for prob_value, word_index in zip(k_prob_value_list, k_word_index_list):
                        prob_value = float(prob_value)
                        word_index = int(word_index)
                        new_record = Bean_Search_Status_Record( predict_word_index_list+[word_index], sum_log_prob+prob_value)
                        new_record.avg_log_prob = new_record.sum_log_prob/(len(new_record.predict_word_index_list) - 1)
                        all_candidates.append(new_record)
                else:
                    all_candidates.append(record)
                        
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


def run_epoch(data_iter, model, loss_compute, virtual_batch_size=128, optimizer=None):
    total_loss = 0
    processed_text = 0
    processed_loss = 0
    for batch in data_iter:
        source, target = batch.source, batch.target
        source_data, source_len = source[0], source[1]
        target_data, target_len = target[0], target[1]

        source_mask = (source_data != TEXT_vi.vocab.stoi['<pad>']).unsqueeze(1)
        #source_mask: batch, 1, source_sen_len

        target_true_word_index = target_data[:,1:]
        target_data = target_data[:,:-1]

        target_mask = ((target_data != TEXT_en.vocab.stoi['<pad>']) & (target_data != TEXT_en.vocab.stoi['<eos>'])).unsqueeze(1)
        #target_mask: batch, 1, source_sen_len

        target_mask = target_mask & (subsequent_mask(target_mask.shape[-1])).type_as(target_mask)

        

        out = model.forward(source_data, target_data, 
                            source_mask, target_mask)
        
        processed_loss += loss_compute(out, target_true_word_index)
        
        processed_text += source_data.shape[0]
        
        if processed_text >= virtual_batch_size:
            if optimizer:
                sys.stdout.flush()
                (processed_loss/processed_text).backward()
                optimizer.step()
                optimizer.zero_grad()
            total_loss += processed_loss.cpu().detach().item()
            processed_loss = 0
            processed_text = 0
            
    return total_loss




def parameters_list_change_grad(model):
    para_list = []
    for name, data in list(model.named_parameters()):
        if 'src_embed' in name or 'tgt_embed' in name:
            data.requires_grad = False
        else:
            para_list.append(data)

    return para_list

save_model_dir_name = '../save_model/self_attention_vi_to_en_'
model = EncoderDecoder(512, 64, 4, 4, len(TEXT_vi.vocab.stoi), len(TEXT_en.vocab.stoi), 0.1).cuda()
model.init_model()
criterion = LabelSmoothing(size=len(TEXT_en.vocab.stoi), padding_idx=TEXT_en.vocab.stoi['<pad>'], smoothing=0.05)
     
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)  


early_stop = 3
best_blue_score = -1
best_index = 0


for index_unique in range(100):
    model.train()
    run_epoch(data_iter=train_vi_en_iter, model=model, loss_compute=SimpleLossCompute(model.generator, criterion), optimizer = optimizer)
    model.eval()
    blue_score = test(model, validation_vi_en_iter)
    print('epoch: ',index_unique, ' the blue score on validation dataset is : ', blue_score)
    sys.stdout.flush()
    if best_blue_score < blue_score:
        
        best_index = index_unique
        best_blue_score = blue_score

        torch.save(model, save_model_dir_name+'model')
        
    if index_unique - best_index >= early_stop:
        break


    
    
print('--------------------------------------')
sys.stdout.flush()



model = torch.load(save_model_dir_name+'model')
model = EncoderDecoder(512, 64, 2, 2, len(TEXT_vi.vocab.stoi), len(TEXT_en.vocab.stoi), 0.1)
model.init_model()
criterion = LabelSmoothing(size=len(TEXT_en.vocab.stoi), padding_idx=TEXT_en.vocab.stoi['<pad>'], smoothing=0.05)

para_list = parameters_list_change_grad(model)     
optimizer = torch.optim.Adam(para_list, lr = 0.001)  
save_model_dir_name = '../save_model/refined_self_attention_vi_to_en_'

early_stop = 3
best_blue_score = -1
best_index = 0

for index_unique in range(100):
    model.train()
    run_epoch(data_iter=train_vi_en_iter, model=model, loss_compute=SimpleLossCompute(model.generator, criterion), optimizer = optimizer)
    model.eval()
    blue_score = test(model, validation_vi_en_iter)
    print('epoch: ',index_unique, ' the blue score on validation dataset is : ', blue_score)
    sys.stdout.flush()
    if best_blue_score < blue_score:
        
        best_index = index_unique
        best_blue_score = blue_score

        torch.save(model, save_model_dir_name+'model')
        
    if index_unique - best_index >= early_stop:
        break


