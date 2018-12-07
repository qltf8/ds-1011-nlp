
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



class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    
    
class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
    
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
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

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
    
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
    
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


class MultiHeadedAttention(nn.Module):
    def __init__(self, d_k, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        
        # We assume d_v always equals d_k
        self.d_k = d_k
        self.linears = clones(nn.Linear(d_model, d_k), 2)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key = [l(x) for l, x in zip(self.linears, (query, key))]
        #query, key = batch, seq, d_k
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
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
    
    
def make_model(src_vocab, tgt_vocab, N=6, 
               d_model=512, d_k=64, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(d_k, d_model)
    ff = PositionwiseFeedForward(d_model, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model



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
model = make_model(src_vocab=len(TEXT_vi.vocab.stoi), tgt_vocab=len(TEXT_en.vocab.stoi), N=4, d_model=512, d_k=64, dropout=0.05).cuda()
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


