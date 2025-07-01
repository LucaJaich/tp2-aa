from models.rnn import RNN
from datasets.rnn import RNNTextDataset, collate_fn
from torch.utils.data import DataLoader
import torch.nn as nn
import json
import torch
from sklearn.metrics import f1_score

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
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn) #la collate_fn agrega el padding

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
        lengths = batch['lengths'].to(DEVICE) #batch['lengths'] tiene las longitudes originales de las secuencias, se aplica padding en collate_fn y se guarda para poder reconstruir

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
        if (i + 1) % 100 == 0:
            print(f"  Batch {i+1} - Loss: {loss.item():.4f}")

    print(f"Epoch {epoch + 1} finished - Total Loss: {total_loss:.4f}")


##test
model.eval()

with open('datasets/data/gutenberg/text_3.json', 'r', encoding='utf-8') as file:
    test_data = json.load(file)

test_dataset = RNNTextDataset(test_data)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

loss_fn = nn.CrossEntropyLoss()

total_loss = 0

# Para F1-score
all_preds_inic = []
all_true_inic = []

all_preds_final = []
all_true_final = []

all_preds_cap = []
all_true_cap = []

with torch.no_grad():
    for batch in test_loader:
        X = batch['X'].to(DEVICE)
        y_inic = batch['p_inicial'].to(DEVICE)
        y_final = batch['p_final'].to(DEVICE)
        y_cap = batch['cap'].to(DEVICE)
        lengths = batch['lengths'].to(DEVICE)

        out_inic, out_final, out_cap = model(X, lengths)

        B, S, _ = out_inic.shape

        loss1 = loss_fn(out_inic.view(B * S, -1), y_inic.view(-1))
        loss2 = loss_fn(out_final.view(B * S, -1), y_final.view(-1))
        loss3 = loss_fn(out_cap.view(B * S, -1), y_cap.view(-1))
        loss = loss1 + loss2 + loss3
        total_loss += loss.item()

        # Predicciones
        preds_inic = out_inic.argmax(dim=-1)
        preds_final = out_final.argmax(dim=-1)
        preds_cap = out_cap.argmax(dim=-1)

        # Máscaras (donde no hay padding)
        mask = y_inic != -100

        # Flatten y filtrar por máscara
        all_preds_inic.extend(preds_inic[mask].cpu().tolist())
        all_true_inic.extend(y_inic[mask].cpu().tolist())

        all_preds_final.extend(preds_final[mask].cpu().tolist())
        all_true_final.extend(y_final[mask].cpu().tolist())

        all_preds_cap.extend(preds_cap[mask].cpu().tolist())
        all_true_cap.extend(y_cap[mask].cpu().tolist())

# Calcular F1 macro
f1_inic = f1_score(all_true_inic, all_preds_inic, average='macro')
f1_final = f1_score(all_true_final, all_preds_final, average='macro')
f1_cap = f1_score(all_true_cap, all_preds_cap, average='macro')

# Reporte
print(f"\n🔍 Evaluación final:")
print(f"Test Loss: {total_loss:.4f}")
print(f"F1-macro puntuación inicial: {f1_inic:.4f}")
print(f"F1-macro puntuación final:   {f1_final:.4f}")
print(f"F1-macro capitalización:     {f1_cap:.4f}")