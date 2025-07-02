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

dataset = RNNTextDataset(data[:5000])
print("Dataset size:", len(dataset))
print("PUNT FINAL:", dataset.inputs["punt_final"].value_counts(normalize=True))
print("PUNT INIC:", dataset.inputs["punt_inicial"].value_counts(normalize=True))
print("CAPS:", dataset.inputs["cap"].value_counts(normalize=True))
print("Dataset created.")

BIDIRECTIONAL = True
INPUT_SIZE = dataset[0]["X"].shape[0] # embedding size
HIDDEN_DIM = 64  # hidden dimension size
NUM_LAYERS = 1  # number of LSTM layers
LEARNING_RATE = 0.001
BATCH_SIZE = 20000
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

model = RNN(INPUT_SIZE, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, bidirectional=BIDIRECTIONAL)
model = model.to(DEVICE)

weights_punct_inic = torch.tensor([1.0, 1.0]).to(DEVICE)  # Adjust weights for initial punctuation
weights_punct_final = torch.tensor([1.0, 1.0, 1.0, 1.0]).to(DEVICE)  # Adjust weights for final punctuation
weights_caps = torch.tensor([1.0, 1.0]).to(DEVICE)  # Adjust weights for capitalization

loss_punt_inic = nn.CrossEntropyLoss(weight=weights_punct_inic.to(DEVICE))
loss_punt_final = nn.CrossEntropyLoss(weight=weights_punct_final.to(DEVICE))
loss_caps = nn.CrossEntropyLoss(weight=weights_caps.to(DEVICE))

optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

total_size = len(dataset)
train_size = int(0.8 * total_size)
test_size = total_size - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print("Train size:", len(train_dataset))
print("Validation size:", len(test_dataset))

train_losses, val_losses = fit(model, train_dataloader, val_dataloader, loss_punt_inic, loss_punt_final, loss_caps, optimizer, NUM_EPOCHS=EPOCHS)

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