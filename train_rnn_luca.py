import json
from datasets.rnn import RNNTextDataset
from models.rnn import RNN
import torch
import torch.nn as nn
from models.utils import fit

print("Loading dataset...")
with open('datasets/data/cuentos/cuentos_cleaned_train.json', 'r') as file:
    data = json.load(file)
print("Dataset loaded.")

dataset = RNNTextDataset(data[:3000])
print("Dataset created.")

BIDIRECTIONAL = True
INPUT_SIZE = dataset[0]["X"].shape[0] # embedding size
HIDDEN_DIM = 512  # hidden dimension size
NUM_LAYERS = 3  # number of LSTM layers
LEARNING_RATE = 0.01
BATCH_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

model = RNN(INPUT_SIZE, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, bidirectional=BIDIRECTIONAL)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

total_size = len(dataset)
train_size = int(0.8 * total_size)
test_size = total_size - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print("Train size:", len(train_dataset))
print("Validation size:", len(test_dataset))

train_losses, val_losses = fit(model, train_dataloader, val_dataloader, criterion, optimizer, NUM_EPOCHS=1)

# chart losses
import matplotlib.pyplot as plt
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.show()

torch.save(model.state_dict(), "./models/checkpoints/exported_program_brnn.pt2")