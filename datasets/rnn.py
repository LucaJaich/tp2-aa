from datasets import TextDataset, assign_classes
import numpy as np

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