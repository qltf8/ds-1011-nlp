import torch

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