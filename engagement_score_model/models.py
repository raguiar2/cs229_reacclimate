import torch
import torch.nn as nn
import pdb
class LTSM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.rnn = nn.LSTM( embedding_dim, 
                            hidden_dim, 
                            num_layers=n_layers, 
                            bidirectional=bidirectional, 
                            dropout=dropout)
       
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear( hidden_dim*2+1, 32 )
        self.linear2 = nn.Linear( 32, 16 )
        self.linear3 = nn.Linear( 16, output_dim )

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths, followerCnt):
        #text = [sent len, batch size]
        embedded = self.embedding(text)
        
        #pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        
        #unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        #and apply dropout
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:], followerCnt.reshape(-1,1)), dim = 1))

        fc = self.dropout( self.relu(self.linear1( hidden.squeeze(0) )))
        fc = self.dropout( self.relu(self.linear2( fc )))
        fc = self.linear3( fc )
            
        return fc
