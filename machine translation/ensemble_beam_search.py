class Bean_Search_Status_Record:
    
    def __init__(self, h_t_list, c_t_list, predict_word_index_list, sum_log_prob):
        self.h_t_list = h_t_list
        self.c_t_list = c_t_list
        self.predict_word_index_list = predict_word_index_list
        self.sum_log_prob = sum_log_prob
        self.avg_log_prob = 0

def beam_search_ensembel_test(encoder_list, decoder_list, data_iter, k=10):
    
    assert len(encoder_list) == len(decoder_list), 'the num of encoders should be equal to the num of decoders'
    _ = [model.eval() for model in encoder_list]
    _ = [model.eval() for model in decoder_list]

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
        
        h_t_list = []
        c_t_list = []
        output_list = []
        
        for encoder in encoder_list:
            all_output, h_n, c_n = encoder(source)
            output = all_output[:,0]
            h_t = h_n[:,1,:]
            c_t = c_n[:,1,:]
            h_t_list.append(h_t)
            c_t_list.append(c_t)
            output_list.append(output)
            

        target_word = TEXT_en.vocab.stoi['<sos>']


        is_init = False


        right_whole_sentence_word_index = target_data[1: target_len[0].item()-1,0]
        right_whole_sentence_word_index = list(right_whole_sentence_word_index.cpu().numpy())
        
        
        sequences = [Bean_Search_Status_Record(h_t_list, c_t_list, predict_word_index_list = [target_word], 
                                               sum_log_prob = 0.0)]
        
        t = 0
        while (t < 60):
            all_candidates = []
            for i in range(len(sequences)):
                record = sequences[i]
                
                h_t_list = record.h_t_list
                c_t_list = record.c_t_list
                predict_word_index_list = record.predict_word_index_list
                sum_log_prob = record.sum_log_prob
                target_word = predict_word_index_list[-1]
                
                temp_h_t_list = []
                temp_c_t_list = []
                temp_prob = None
                
                if TEXT_en.vocab.stoi['<eos>'] != target_word:
                    for num_model in range(len(encoder_list)):
                        
                        decoder = decoder_list[num_model]
                        h_t = h_t_list[num_model]
                        c_t = c_t_list[num_model]
                        output = output_list[num_model]
                
                        prob, h_t, c_t = decoder(torch.tensor([target_word]).cuda(0), h_t, c_t, output, is_init)
                    
                        temp_h_t_list.append(h_t)
                        temp_c_t_list.append(c_t)
                        
                        if temp_prob is None:
                            temp_prob = prob
                        else:
                            temp_prob = torch.cat([temp_prob, prob], dim=0)
                            
                    
                            
                            
                    prob = temp_prob.mean(dim=0, keepdim=True)
                    k_prob_value_list, k_word_index_list = prob.topk(k,dim=1)
                    k_prob_value_list = k_prob_value_list.cpu().detach().squeeze().numpy()
                    k_word_index_list = k_word_index_list.cpu().squeeze().numpy()


                    for prob_value, word_index in zip(k_prob_value_list, k_word_index_list):
                        prob_value = float(prob_value)
                        word_index = int(word_index)
                        new_record = Bean_Search_Status_Record(temp_h_t_list, temp_c_t_list, predict_word_index_list+[word_index], sum_log_prob+prob_value)
                        new_record.avg_log_prob = new_record.sum_log_prob/((4+len(new_record.predict_word_index_list))**0.6/(6)**0.6)
                        all_candidates.append(new_record)
                else:
                    all_candidates.append(record)
            is_init = False
                        
            ordered = sorted(all_candidates, key = lambda r: r.avg_log_prob, reverse = True)
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