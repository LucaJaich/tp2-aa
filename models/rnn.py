from torch import nn
    
class RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers=1, bidirectional=False, dropout=0.5):
        super().__init__()
        
        hidden_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                    batch_first=True, bidirectional=False, dropout=dropout)
        
        self.fc_punct_inic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )
        self.fc_punct_final = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4)
        )
        self.fc_caps = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )
        
    def forward(self, x):
        output, _ = self.lstm(x)
        
        output = output.squeeze(1)
        out_punct_inic = self.fc_punct_inic(output)
        out_punct_final = self.fc_punct_final(output)
        out_caps = self.fc_caps(output)
        
        return out_punct_inic, out_punct_final, out_caps