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
        return {'X': x,
                'p_inicial': y1,
                'p_final': y2,
                'cap': y3}
    
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    """
    batch: lista de dicts con keys: 'X', 'p_inicial', 'p_final', 'cap'
    """
    # Extraer los elementos por separado
    Xs = [item['X'] for item in batch]
    y_inic = [item['p_inicial'] for item in batch]
    y_final = [item['p_final'] for item in batch]
    y_cap = [item['cap'] for item in batch]

    # Calcular longitudes originales
    lengths = torch.tensor([x.shape[0] for x in Xs])  # (batch,)

    # Padding (batch_size, max_seq_len, embedding_dim)
    Xs_padded = pad_sequence(Xs, batch_first=True, padding_value=0.0)

    # Padding de labels (batch_size, max_seq_len)
    y_inic_padded = pad_sequence(y_inic, batch_first=True, padding_value=-100)  # -100 se ignora en CrossEntropy
    y_final_padded = pad_sequence(y_final, batch_first=True, padding_value=-100)
    y_cap_padded = pad_sequence(y_cap, batch_first=True, padding_value=-100)

    return {
        'X': Xs_padded,                # (B, T, D)
        'p_inicial': y_inic_padded,   # (B, T)
        'p_final': y_final_padded,
        'cap': y_cap_padded,
        'lengths': lengths             # (B,)
    }