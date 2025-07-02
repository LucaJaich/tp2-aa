from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers=1, bidirectional=False, dropout=0.9):
        super().__init__()
        
        self.lstm = nn.RNN(embedding_dim, hidden_dim, num_layers, 
                    batch_first=True, bidirectional=bidirectional, dropout=dropout)
        
        hidden_dim_linear = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.fc_punct_inic = nn.Sequential(
            nn.Linear(hidden_dim_linear, hidden_dim),  
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )
        self.fc_punct_final = nn.Sequential(
            nn.Linear(hidden_dim_linear, hidden_dim),  
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4)
        )
        self.fc_caps = nn.Sequential(
            nn.Linear(hidden_dim_linear, hidden_dim),  
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x, lengths):
        # x: (batch_size, seq_len, embedding_dim)
        # lengths: (batch_size,) longitudes reales (sin padding)

        # Empaquetar la secuencia para que la RNN ignore el padding
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

        packed_output, _ = self.lstm(packed)

        # Desempaquetar para obtener tensor padded
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_punct_inic = self.fc_punct_inic(output)
        out_punct_final = self.fc_punct_final(output)
        out_caps = self.fc_caps(output)

        return out_punct_inic, out_punct_final, out_caps   
    # def forward(self, x):
    #     output, _ = self.lstm(x)

    #     print(x.shape)
        
    #     # output = output.squeeze(1)
    #     out_punct_inic = self.fc_punct_inic(output)
    #     out_punct_final = self.fc_punct_final(output)
    #     out_caps = self.fc_caps(output)
        
    #     return out_punct_inic, out_punct_final, out_caps
    