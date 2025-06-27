from datasets import TextDataset
import numpy as np

class RNNEndTextDataset(TextDataset):
    def __init__(self, texts: list[str], tokenizer="bert"):
        super().__init__(texts, tokenizer)
        self.y = self.inputs['punt_final']
        self.X = np.concatenate([text.embeddings for text in self.texts])
        print(self.texts[0].embeddings.shape)
        print(len(self.X), self.X[0].shape)

    def __getitem__(self, idx):
        return {'X': self.X[idx], 'y': self.y[idx]}
    
    def __len__(self):
        return len(self.X)