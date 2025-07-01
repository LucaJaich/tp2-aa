from models import RNN
from datasets import RNNTextDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

dataset = RNNTextDataset("datasets/data/gutenberg", tokenizer, embedder, labeler)
#dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

model = RNN(embedding_dim=768, hidden_dim=256)
criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = optim.Adam(model.parameters())

model.train()
for epoch in range(5):
    for batch in dataloader:
        inputs, (y_inic, y_final, y_caps) = batch  # inputs: (B, L, D), y_*: (B, L)
        
        optimizer.zero_grad()
        out_inic, out_final, out_caps = model(inputs)

        loss_inic = criterion(out_inic.view(-1, 2), y_inic.view(-1))
        loss_final = criterion(out_final.view(-1, 4), y_final.view(-1))
        loss_caps = criterion(out_caps.view(-1, 4), y_caps.view(-1))

        loss = loss_inic + loss_final + loss_caps
        loss.backward()
        optimizer.step()
