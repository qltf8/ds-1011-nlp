import pandas as pd
import numpy as np
import torch
import torchtext
import pickle
import csv
import unicodedata
import re, random, time, string, subprocess
import os, sys, copy





TEXT_zh = torchtext.data.ReversibleField(sequential=True, use_vocab=True, batch_first = False, tokenize= lambda t:t.split(),
                                        include_lengths=True)
TEXT_en = torchtext.data.ReversibleField(sequential=True, use_vocab=True, batch_first = False, tokenize= lambda t:t.split(),
                              lower=True, init_token='<sos>', eos_token='<eos>',include_lengths=True)
train_zh_en = torchtext.data.TabularDataset('/home/ql819/text_data/train_zh_en.csv', format='csv', 
                             fields=[('source',TEXT_zh),('target',TEXT_en)])
validation_zh_en = torchtext.data.TabularDataset('/home/ql819/text_data/dev_zh_en.csv', format='csv', 
                             fields=[('source',TEXT_zh),('target',TEXT_en)])


TEXT_zh.build_vocab(train_zh_en, min_freq=3)
TEXT_en.build_vocab(train_zh_en, min_freq=3)


train_zh_en_iter = torchtext.data.BucketIterator(train_zh_en, batch_size=1, sort_key= lambda e: len(e.source),
                             repeat = False, sort_within_batch=True, shuffle=True, device=torch.device(0))
validation_zh_en_iter = torchtext.data.BucketIterator(validation_zh_en, batch_size=1, sort_key= lambda e: len(e.source),
                             repeat = False, sort_within_batch=True, shuffle=True, device=torch.device(0))



class Bi_Multi_Layer_LSTM_Encoder(torch.nn.Module):
    
    def __init__(self, num_vocab, input_size = 512, hidden_size = 512, dropout = 0.15):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.dropout = dropout
        
        self.dropout_layer = torch.nn.Dropout(self.dropout)
        
        self.bidirectional = True
        self.embedding_layer = torch.nn.Embedding(num_vocab, self.input_size)
        self.lstm = torch.nn.LSTM(input_size= self.input_size, hidden_size = self.hidden_size, batch_first = False,
                                 bidirectional = self.bidirectional, num_layers = self.num_layers)
        
        h_0 = torch.zeros(1, self.hidden_size)
        torch.nn.init.normal_(h_0, mean=0, std=0.0001)
        self.h_0 = torch.nn.Parameter(h_0,requires_grad=True)
        
        
        c_0 = torch.zeros(1, self.hidden_size)
        torch.nn.init.normal_(c_0, mean=0, std=0.0001)
        self.c_0 = torch.nn.Parameter(c_0,requires_grad=True)
        
        
        
        if self.bidirectional:
            h_1 = torch.zeros(1, self.hidden_size)
            torch.nn.init.normal_(h_1, mean=0, std=0.0001)
            self.h_1 = torch.nn.Parameter(h_1,requires_grad=True)
            
            
            c_1 = torch.zeros(1, self.hidden_size)
            torch.nn.init.normal_(c_1, mean=0, std=0.0001)
            self.c_1 = torch.nn.Parameter(c_1,requires_grad=True)
            
        
    def forward(self, X):
        
        X_data,X_len = X
        #X_data: source_len, 1, input_size    X_len:1,1
        
        X_data = self.embedding_layer(X_data)
        
        h_0 = torch.cat([self.h_0]*len(X_len), dim=0).unsqueeze(1)
        c_0 = torch.cat([self.c_0]*len(X_len), dim=0).unsqueeze(1)
        
        
        if self.bidirectional:
            h_1 = torch.cat([self.h_1]*len(X_len), dim=0).unsqueeze(1)
            c_1 = torch.cat([self.c_1]*len(X_len), dim=0).unsqueeze(1)
            
            h = torch.cat([h_0,h_1], dim=0)
            c = torch.cat([c_0,c_1], dim=0)   

        output, (h_n, c_n) = self.lstm(X_data, (h, c))
        #output: source_len, 1, 2*hidden_size
        h_n = h_n.view(self.num_layers, 2, len(X_len), self.hidden_size)
        c_n = c_n.view(self.num_layers, 2, len(X_len), self.hidden_size)
        
        
        return output, h_n, c_n
    
    def init_parameters(self):
        
        for name, matrix in self.lstm.named_parameters():
            if 'weight_hh_' in name:
                for i in range(0, matrix.size(0), self.hidden_size):
                    torch.nn.init.orthogonal_(matrix[i:i+self.hidden_size], gain=0.01)
            elif 'bias_' in name:
                l = len(matrix)
                matrix[l // 4: l //2].data.fill_(1.0)
                
                
class LSTM_Decoder_With_Attention(torch.nn.Module):
    
    def __init__(self, num_vocab, input_size = 512, hidden_size = 512, dropout=0.15):
        super().__init__()
        self.num_vocab = num_vocab
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.dropout = dropout
        self.dropout_layer = torch.nn.Dropout(self.dropout)
        
        self.embedding_layer = torch.nn.Embedding(self.num_vocab, self.input_size)
        self.lstm = torch.nn.LSTM(hidden_size= self.hidden_size, input_size= self.input_size + 2 * self.hidden_size, 
                                  num_layers= self.num_layers)
        
        self.calcu_weight_1  = torch.nn.Linear(3*self.hidden_size, hidden_size)
        self.calcu_weight_2  = torch.nn.Linear(self.hidden_size, 1)
        
        self.init_weight_1 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.init_weight_2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        
        self.linear_vob = torch.nn.Linear(self.hidden_size, self.num_vocab)
        
        
    def forward(self, input_word_index, hidden_vector, cell_vector, encoder_memory, is_init = False):
        #input_word_index: [num]
        #hidden_vector: 1, 1, hidden_size
        #cell_vector: 1, 1, hidden_size
        #encoder_memory: source_sen_len , 2 * hidden_size
        
        if hidden_vector.shape[0] != self.num_layers or hidden_vector.shape[2] != self.hidden_size:
            raise ValueError('The size of hidden_vector is not correct, expect '+str((self.num_layers, self.hidden_size))\
                            + ', actually get ' + str(hidden_vector.shape))
        
        if is_init:
            hidden_vector = torch.tanh(self.init_weight_1(hidden_vector))
            cell_vector = torch.tanh(self.init_weight_2(cell_vector))
            
        
        
        n_hidden_vector = torch.stack([hidden_vector.squeeze()]*encoder_memory.shape[0],dim=0)
        com_n_h_memory = torch.cat([n_hidden_vector, encoder_memory], dim =1)
        com_n_h_temp = torch.tanh(self.calcu_weight_1(com_n_h_memory))
        
        
        weight_vector = self.calcu_weight_2(com_n_h_temp)
        weight_vector =  torch.nn.functional.softmax(weight_vector, dim=0)
        #weight_vector: source_sen_len * 1
        
        
        convect_vector = torch.mm(weight_vector.transpose(1,0), encoder_memory)
        #convect_vector: 1 , 2 * hidden_size
        
        
        input_vector = self.embedding_layer(input_word_index).view(1,1,-1)
        input_vector = self.dropout_layer(input_vector)
        
        
        input_vector = torch.cat([convect_vector.unsqueeze(0), input_vector], dim=2)
        
        
        output, (h_t, c_t) = self.lstm(input_vector,(hidden_vector, cell_vector))
        output = output.view(1, self.hidden_size)
        
        
        prob = self.linear_vob(output)
        #prob 1, vob_size
        
        prob = torch.nn.functional.log_softmax(prob, dim=1)
        
        
        return prob, h_t, c_t
    
    def init_parameters(self):
        
        for name, matrix in self.lstm.named_parameters():
            if 'weight_hh_' in name:
                for i in range(0, matrix.size(0), self.hidden_size):
                    torch.nn.init.orthogonal_(matrix[i:i+self.hidden_size], gain=0.01)
            elif 'bias_' in name:
                l = len(matrix)
                matrix[l // 4: l //2].data.fill_(1.0)
    
                    
                
def train(encoder, decoder, optimizer, data_iter, teacher_forcing_ratio, batch_size = 64):

    encoder.train()
    decoder.train()
    
    count = 0
    loss = 0
    
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    
    
    for batch in data_iter:
        
        
        source, target = batch.source, batch.target
        

        target_data,target_len = target[0], target[1]
        
        all_output, h_n, c_n = encoder(source)
        

        output = all_output[:,0]
        target_word_list = target_data.squeeze()
        target_word = torch.tensor([TEXT_en.vocab.stoi['<sos>']]).cuda(0)

        h_t = h_n[:,1,:]
        c_t = c_n[:,1,:]

        is_init = True

        for word_index in range(1, target_len[0].item()):
            prob, h_t, c_t = decoder(target_word, h_t, c_t, output, is_init)
            
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
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        
        
    if count % batch_size != 0:
        loss = loss/count
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

                
            
        
class Bean_Search_Status_Record:
    
    def __init__(self, h_t, c_t, predict_word_index_list, sum_log_prob):
        self.h_t = h_t
        self.c_t = c_t
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
        
        all_output, h_n, c_n = encoder(source)
        output = all_output[:,0]

        target_word = TEXT_en.vocab.stoi['<sos>']

        h_t = h_n[:,1,:]
        c_t = c_n[:,1,:]

        is_init = True


        right_whole_sentence_word_index = target_data[1: target_len[0].item()-1,0]
        right_whole_sentence_word_index = list(right_whole_sentence_word_index.cpu().numpy())
        
        
        sequences = [Bean_Search_Status_Record(h_t, c_t, predict_word_index_list = [target_word], 
                                               sum_log_prob = 0.0)]
        
        t = 0
        while (t < 100):
            all_candidates = []
            for i in range(len(sequences)):
                record = sequences[i]
                h_t = record.h_t
                c_t = record.c_t
                predict_word_index_list = record.predict_word_index_list
                sum_log_prob = record.sum_log_prob
                target_word = predict_word_index_list[-1]
                
                if TEXT_en.vocab.stoi['<eos>'] != target_word:
                
                    prob, h_t, c_t = decoder(torch.tensor([target_word]).cuda(0), h_t, c_t, output, is_init)

                    k_prob_value_list, k_word_index_list = prob.topk(k,dim=1)
                    k_prob_value_list = k_prob_value_list.cpu().detach().squeeze().numpy()
                    k_word_index_list = k_word_index_list.cpu().squeeze().numpy()
                    
                    
                    for prob_value, word_index in zip(k_prob_value_list, k_word_index_list):
                        prob_value = float(prob_value)
                        word_index = int(word_index)
                        new_record = Bean_Search_Status_Record(h_t, c_t, predict_word_index_list+[word_index], sum_log_prob+prob_value)
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


def parameters_list(encoder, decoder):
    para_list_1 = []
    para_list_2 = []
    for name, data in list(encoder.named_parameters()):
        if 'embedding' in name:
            para_list_1.append(data)
        else:
            para_list_2.append(data)

    for name, data in list(decoder.named_parameters()):
        if 'embedding' in name:
            para_list_1.append(data)
        else:
            para_list_2.append(data)
    return para_list_1, para_list_2


def parameters_list_change_grad(encoder, decoder):
    para_list = []
    for name, data in list(encoder.named_parameters()):
        if 'embedding' in name:
            data.requires_grad = False
        else:
            para_list.append(data)

    for name, data in list(decoder.named_parameters()):
        if 'embedding' in name:
            data.requires_grad = False
        else:
            para_list.append(data)
    return para_list



encoder = Bi_Multi_Layer_LSTM_Encoder(num_vocab=len(TEXT_zh.vocab.stoi))
decoder = LSTM_Decoder_With_Attention(num_vocab = len(TEXT_en.vocab.stoi))
encoder.init_parameters()
decoder.init_parameters()
encoder = encoder.cuda(0)
decoder = decoder.cuda(0)

early_stop = 4
best_blue_score = -1
best_index = -1

save_model_dir_name = '../save_model/teacher_zh_to_en_'

para_list_1, para_list_2 = parameters_list(encoder, decoder)

optimizer = torch.optim.Adam([{'params': para_list_1, 'lr': 0.001},
                              {'params': para_list_2, 'lr': 0.001}])

teacher_forcing_ratio = 0.95

for index_unique in range(100):
    train(encoder, decoder, optimizer, train_zh_en_iter, teacher_forcing_ratio)
    blue_score = test(encoder, decoder, validation_zh_en_iter)
    print('epoch: ',index_unique, ' the blue score on validation dataset is : ', blue_score)
    sys.stdout.flush()
    if best_blue_score < blue_score:
        
        best_index = index_unique
        best_blue_score = blue_score
        best_encoder = copy.deepcopy(encoder)
        best_decoder = copy.deepcopy(decoder)
        torch.save(encoder, save_model_dir_name+'encode_'+str(index_unique))
        torch.save(decoder, save_model_dir_name+'decoder_'+str(index_unique))
        
    if index_unique - best_index >= early_stop:
        break
        
        
        
        
        
print('--------------------------------------')
sys.stdout.flush()
        
        


encoder = best_encoder
decoder = best_decoder

para_list = parameters_list_change_grad(encoder, decoder)     
optimizer = torch.optim.Adam(para_list, lr = 0.001)  
save_model_dir_name = '../save_model/teacher_refined_zh_to_en_'

early_stop = 3
best_blue_score = -1
best_index = -1

for index_unique in range(100):
    train(encoder, decoder, optimizer, train_zh_en_iter, teacher_forcing_ratio)
    blue_score = test(encoder, decoder, validation_zh_en_iter)
    print('epoch: ',index_unique, ' the blue score on validation dataset is : ', blue_score)
    sys.stdout.flush()
    
    if best_blue_score < blue_score:
        
        best_index = index_unique
        best_blue_score = blue_score
        torch.save(encoder, save_model_dir_name+'encode_'+str(index_unique))
        torch.save(decoder, save_model_dir_name+'decoder_'+str(index_unique))
    if index_unique - best_index >= early_stop:
        break