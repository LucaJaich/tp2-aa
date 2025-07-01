from models.rnn import RNN
from datasets.rnn import RNNTextDataset, collate_fn
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import json
import torch
import tqdm

# Parámetros
BATCH_SIZE = 8
EPOCHS = 1
HIDDEN_DIM = 128
EMBEDDING_DIM = 768  # por bert case
NUM_LAYERS = 1
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#ejemplo con un solo archivo
with open('datasets/data/gutenberg/text_2.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

dataset = RNNTextDataset(data)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

model = RNN(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    print(f"Epoch {epoch + 1}...")

    for i, batch in enumerate(dataloader):
        X = batch['X'].to(DEVICE)
        y_inic = batch['p_inicial'].to(DEVICE)
        y_final = batch['p_final'].to(DEVICE)
        y_cap = batch['cap'].to(DEVICE)
        lengths = batch['lengths'].to(DEVICE)

        optimizer.zero_grad()

        out_inic, out_final, out_cap = model(X, lengths)

        B, S, _ = out_inic.shape
        loss1 = loss_fn(out_inic.view(B * S, -1), y_inic.view(-1))
        loss2 = loss_fn(out_final.view(B * S, -1), y_final.view(-1))
        loss3 = loss_fn(out_cap.view(B * S, -1), y_cap.view(-1))

        loss = loss1 + loss2 + loss3
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Opción: imprimir cada N batches si querés progreso
        if (i + 1) % 10 == 0:
            print(f"  Batch {i+1} - Loss: {loss.item():.4f}")

    print(f"Epoch {epoch + 1} finished - Total Loss: {total_loss:.4f}")
