import numpy as np
from datasets import TextDataset

class CapTextDataset(TextDataset):
    def __init__(self, texts: list[str], tokenizer="bert"):
        super().__init__(texts, tokenizer)
        self.y = self.inputs['cap']
        self.X = np.concatenate([text.embeddings for text in self.texts])
        print(self.X.shape)

    def __getitem__(self, idx):
        return {
            'X': self.X[idx],
            'y': self.y[idx]
        }

class BeginningTextDataset(TextDataset):
    def __init__(self, texts: list[str], tokenizer="bert"):
        super().__init__(texts, tokenizer)
        self.y = assign_classes(self.inputs['punt_inicial'])
        self.X = np.concatenate([text.embeddings for text in self.texts])

    def __getitem__(self, idx):
        return {
            'X': self.X[idx],
            'y': self.y[idx]
        }

class WBeginningTextDataset(TextDataset):
    def __init__(self, texts: list[str], tokenizer="bert", window_size=1):
        super().__init__(texts, tokenizer)
        self.y = assign_classes(self.inputs['punt_inicial'])
        self.embeddings = np.concatenate([text.embeddings for text in self.texts])
        self.X = []
        for i in range(len(self.embeddings)):
            start = max(0, i - window_size)
            end = min(len(self.embeddings), i + window_size + 1)
            self.X.append(self.embeddings[start:end].flatten())
        self.X = np.array(self.X)

    def __getitem__(self, idx):
        return {
            'X': self.X[idx],
            'y': self.y[idx]
        }

class EndTextDataset(TextDataset):
    def __init__(self, texts: list[str], tokenizer="bert"):
        super().__init__(texts, tokenizer)
        self.y = assign_classes(self.inputs['punt_final'])
        self.X = np.concatenate([text.embeddings for text in self.texts])

    def __getitem__(self, idx):
        return {
            'X': self.X[idx],
            'y': self.y[idx]
        }
    
class WEndTextDataset(TextDataset):
    def __init__(self, texts: list[str], tokenizer="bert", window_size=0):
        super().__init__(texts, tokenizer)
        self.y = assign_classes(self.inputs['punt_final'])
        self.embeddings = np.concatenate([text.embeddings for text in self.texts])
        self.X = []
        for i in range(len(self.embeddings)):
            window = create_window(i, self.embeddings, window_size)
            self.X.append(np.concatenate(window))
        
        # print(len(self.X), self.X[0].shape, self.X[1].shape)
        self.X = np.array(self.X)

    def __getitem__(self, idx):
        return {
            'X': self.X[idx],
            'y': self.y[idx]
        }
    
def create_window(i: int, embeddings: np.ndarray, window_size: int) -> np.ndarray:
    start = max(0, i - window_size)
    end = min(len(embeddings), i + window_size + 1)
    emb_size = embeddings.shape[1]
    left_padding_size = window_size*emb_size - len(embeddings[start:i]) * emb_size
    right_padding_size = window_size*emb_size - len(embeddings[min(i+1, len(embeddings)):end]) * emb_size
    left_padding = np.zeros((left_padding_size,))
    right_padding = np.zeros((right_padding_size,))

    return [
        embeddings[i],
        np.concatenate([left_padding, embeddings[start:i].flatten()]),
        np.concatenate([embeddings[min(i+1, len(embeddings)):end].flatten(), right_padding])
    ]

def assign_classes(labels: list[str]) -> list[int]:
    """
    Assigns a unique integer to each unique string in the list,
    in the order of first appearance.
    """
    label_to_class = {}
    classes = []
    current_class = 0
    for label in labels:
        if label not in label_to_class:
            label_to_class[label] = current_class
            current_class += 1
        classes.append(label_to_class[label])
    return classes

if __name__ == "__main__":
    # Example usage
    texts = ["Hello world.", "This is a test.", "Another example text."]
    dataset = BeginningTextDataset(texts)