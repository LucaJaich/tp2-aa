from torch import nn

class RNNEndModel(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size=hidden_dim, batch_first=True)
        sigmoid = nn.Sigmoid()
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 1), sigmoid)

    def forward(self, x):
        output, _ = self.rnn(x)
        output = self.fc(output.squeeze(1))
        return output
