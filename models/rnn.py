from torch import nn

class RNNEndModel(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size=hidden_dim, batch_first=True)
        softmax = nn.Softmax()
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 4), softmax)

    def forward(self, x):
        output, _ = self.rnn(x)
        output = self.fc(output.squeeze(1))
        return output
