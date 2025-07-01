from torch import nn

class BiRNN(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                            batch_first=True, bidirectional=True)

        self.fc_punct_inic = nn.Linear(hidden_dim * 2, 2)  # ¿ o nada
        self.fc_punct_final = nn.Linear(hidden_dim * 2, 4) # ¿ , . o nada
        self.fc_caps = nn.Linear(hidden_dim * 2, 4)        # tipos de capitalización
        #sin softmax porque se usa CrossEntropyLoss que ya lo aplica internamente


    def forward(self, embeddings):
        """
            Receives a sequence of BERT embeddings and outputs, for each token:
            logits de puntuacion inicial entre ¿ o nada
            logits de puntuacion final entre ? , . o nada
            logits de capitalizacion entre todo minuscula, primera mayuscula, alguna mayuscula intermedia, todo mayuscula
        """
        lstm_out, _ = self.lstm(embeddings)  # (batch_size, seq_len, hidden_dim)

        out_punct_inic = self.fc_punct_inic(lstm_out)
        out_punct_final = self.fc_punct_final(lstm_out)
        out_caps = self.fc_caps(lstm_out)

        return out_punct_inic, out_punct_final, out_caps