from datasets import TextDataset, assign_classes
import numpy as np
import torch

def one_hot(y):
    classes = assign_classes(y)
    num_classes = max(classes) + 1  # or set manually if known
    return torch.tensor(np.eye(num_classes)[classes])

class RNNTextDataset(TextDataset):
    def __init__(self, texts: list[str], tokenizer="bert"):
        super().__init__(texts, tokenizer)
        #self.X = torch.tensor(np.concatenate([text.embeddings for text in self.texts]))
        self.X = [torch.tensor(text.embeddings) for text in self.texts]  # lista de tensores (seq_len, embedding_dim)

        self.punt_inicial = torch.tensor(assign_classes(self.inputs['punt_inicial']))
        self.punt_final = torch.tensor(assign_classes(self.inputs['punt_final']))
        self.cap = torch.tensor(self.inputs['cap'])  # Capitalization classes

    def __getitem__(self, idx):
            print(f"Fetching idx={idx}")
            print(f"X length: {len(self.X)}")
            print(f"punt_final length: {len(self.punt_final)}")
            print(f"punt_inicial length: {len(self.punt_inicial)}")
            print(f"cap length: {len(self.cap)}")
            return {
                'X': self.X[idx],  # tensor (seq_len, embedding_dim)
                'p_final': torch.tensor(assign_classes(self.inputs['punt_final'][idx])),
                'p_inicial': torch.tensor(assign_classes(self.inputs['punt_inicial'][idx])),
                'cap': torch.tensor(self.inputs['cap'][idx])
            }

    # def __getitem__(self, idx):
    #     return {
    #         'X': self.X[idx],
    #         'p_final': self.punt_final[idx],
    #         'p_inicial': self.punt_inicial[idx],
    #         'cap': self.cap[idx]
    #     }
# class RNNEndTextDataset(TextDataset):
#     def __init__(self, texts: list[str], tokenizer="bert"):
#         super().__init__(texts, tokenizer)
#         self.y_classes = np.array(assign_classes(self.inputs['punt_final']))
#         self.X = np.concatenate([text.embeddings for text in self.texts])
#         num_classes = self.y_classes.max() + 1    # or set manually if known
#         self.y = np.eye(num_classes)[self.y_classes]

#     def __getitem__(self, idx):
#         return {'X': self.X[idx], 'y': self.y[idx]}
    
#     def __len__(self):
#         return len(self.X)