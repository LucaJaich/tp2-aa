from torch.utils.data import IterableDataset
import os
import json
import torch

class StreamingTextDataset(IterableDataset):
    def __init__(self, folder_path, tokenizer="bert"):
        self.folder_path = folder_path
        self.tokenizer = tokenizer
        self.filepaths = [os.path.join(folder_path, f) 
                          for f in os.listdir(folder_path) if f.endswith(".json")]

    def __iter__(self):
        for filepath in self.filepaths:
            with open(filepath, "r", encoding="utf-8") as f:
                texts = json.load(f)
                for text in texts:
                    text_obj = Text(text, tokenizer=self.tokenizer)
                    df = format_input(text_obj)

                    for _, row in df.iterrows():
                        # Convertir cada fila en tensor (embedding + etiquetas)
                        embedding = torch.tensor(row['embedding'], dtype=torch.float)
                        y_inic = torch.tensor(row['punct_inic'], dtype=torch.long)
                        y_final = torch.tensor(row['punct_final'], dtype=torch.long)
                        y_caps = torch.tensor(row['capitalization'], dtype=torch.long)

                        yield embedding, (y_inic, y_final, y_caps)
#recorrerlo con algo tipo: 
# 
# # from torch.utils.data import DataLoader

# streaming_dataset = StreamingTextDataset("data/gutenberg", tokenizer="bert")
# dataloader = DataLoader(streaming_dataset, batch_size=64)  # batch_size = tokens

# for epoch in range(epochs):
#     for batch in dataloader:
#         x, (y_inic, y_final, y_caps) = batch
#         # x: (batch_size, emb_dim)
#         # convertilo a (B, L, D) si querés hacer secuencias (ver nota abajo)