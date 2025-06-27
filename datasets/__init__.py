import pandas as pd
from torch.utils.data import Dataset
from utils import Text, format_input

class TextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer="bert"):
        # falta super init
        self.texts = [Text(text, tokenizer) for text in texts]
        self.inputs = pd.concat([format_input(text, id=i) for i, text in enumerate(self.texts)], ignore_index=True)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs.iloc[idx]