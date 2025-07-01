from datasets import TextDataset, assign_classes
import numpy as np
import torch

class RNNEndTextDataset(TextDataset):
    def __init__(self, texts: list[str], tokenizer="bert"):
        super().__init__(texts, tokenizer)
        self.y_classes = np.array(assign_classes(self.inputs['punt_final']))
        self.X = np.concatenate([text.embeddings for text in self.texts])
        num_classes = self.y_classes.max() + 1    # or set manually if known
        self.y = np.eye(num_classes)[self.y_classes]

    def __getitem__(self, idx):
        return {'X': self.X[idx], 'y': self.y[idx]}
    
    def __len__(self):
        return len(self.X)
    
class RNNTextDataset(TextDataset):
    def __init__(self, texts: list[str], tokenizer="bert"):
        super().__init__(texts, tokenizer)
        self.embeddings = [text.embeddings for text in self.texts]

        self.labels_punt_inic = [torch.tensor(assign_classes(text.punt_inicial)) for text in self.texts]
        self.labels_punt_final = [torch.tensor(assign_classes(text.punt_final)) for text in self.texts]
        self.labels_cap = [torch.tensor(text.cap) for text in self.texts]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = torch.tensor(self.embeddings[idx])  # (seq_len, embedding_dim)
        y1 = self.labels_punt_inic[idx]         # (seq_len,)
        y2 = self.labels_punt_final[idx]        # (seq_len,)
        y3 = self.labels_cap[idx]               # (seq_len,)
        return x, y1, y2, y3