from datasets.rnn import RNNEndTextDataset
from models.rnn import RNNEndModel
from models.utils import fit

from torch import nn
import torch
import json

print("Loading dataset...")
with open('datasets/data/film_reviews.json', 'r') as file:
    data = json.load(file)
print("Dataset loaded.")

print("Creating dataset...")
dataset = RNNEndTextDataset(data[:100])
print("Dataset created.")

print("Creating model...")
INPUT_SIZE = dataset[0]["X"].shape[0]
model = RNNEndModel(input_size=INPUT_SIZE, hidden_dim=64)

# DEBERIA USAR PAD_TOKEN?
# criterion = nn.CrossEntropyLoss(ignore_index=PAD_token) 

LEARNING_RATE = 0.01

criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

total_size = len(dataset)
train_size = int(0.8 * total_size)
test_size = total_size - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
print(train_dataset[0])

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

print("Fitting model...")
train_losses, val_losses = fit(model, train_dataloader, val_dataloader, criterion, optimizer, NUM_EPOCHS=1)
len(train_losses)