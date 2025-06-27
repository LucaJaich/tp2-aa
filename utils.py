import pandas as pd
from transformers import BertTokenizer, BertModel

MODEL = "bert-base-multilingual-cased"  # Default model for BERT tokenizer

tokenizers = {
    "bert": BertTokenizer.from_pretrained(MODEL),
    "gpt2": None,  # Placeholder for GPT-2 tokenizer
    "t5": None,   # Placeholder for T5 tokenizer
}

models = {
    "bert":  BertModel.from_pretrained(MODEL),
    "gpt2": None,  # Placeholder for GPT-2 model
    "t5": None,   # Placeholder for T5 model
}

PUNCTUATION = {
    "start": ["¿"],
    "end": [".", ",", "?"],
}

class Text:
    def __init__(self, text: str, tokenizer="bert"):
        # OJO CON CAMBIAR EL ORDEN DE LAS COSAS
        self.text = text
        self.tokens_raw = tokenizers[tokenizer].tokenize(text)
        self.tokens = [token for token in self.tokens_raw if token not in PUNCTUATION["start"] + PUNCTUATION["end"]]
        self.punt_inicial, self.punt_final = self._get_punts()
        self.cap = self._get_cap()  # OJO: esto cambia los tokens a minúsculas
        self.tokens_ids = tokenizers[tokenizer].convert_tokens_to_ids(self.tokens) # esto debería ir desp de hacerlo minúscula
        self.embeddings = get_embeddings(self.tokens_ids, tokenizer=tokenizer)

    def _get_punts(self):
        padding = 0
        punt_inicial = [""] * len(self.tokens)
        punt_final = [""] * len(self.tokens)
        
        for i, token in enumerate(self.tokens_raw):
            if token in PUNCTUATION["end"]:
                punt_final[i-padding-1] = token
                padding += 1

            elif token in PUNCTUATION["start"]:
                punt_inicial[i-padding] = token
                padding += 1

        return punt_inicial, punt_final
    
    def _get_cap(self):
        # OJO: cambian los ids de los tokens porque lo paso a minúsculas, cuidado con cambiar el orden de las cosas
        cap = [0] * len(self.tokens)
        last_cap = 0

        for i, token in enumerate(self.tokens):
            if token.startswith("##"):
                cap[i] = last_cap
            elif token[0].isupper():
                cap[i] = 1
                last_cap = 1
            elif any(char.isupper() for char in token):
                cap[i] = 2
                last_cap = 2
            elif token.isupper():
                cap[i] = 3
                last_cap = 3
            else:
                cap[i] = 0
                last_cap = 0
        
        self.tokens = [token.lower() for token in self.tokens]

        return cap

    def __str__(self):
        return f"Text: {self.text}\nTokens: {self.tokens}\nToken IDs: {self.tokens_ids}"
    
def format_input(text: Text, id=1):
    return pd.DataFrame({
        "instancia_id": [id for _ in range(len(text.tokens))],
        "token_id": text.tokens_ids,
        "token": text.tokens,
        "punt_inicial": text.punt_inicial,
        "punt_final": text.punt_final,
        "cap": text.cap,
    })


def get_embeddings(ids: list[int], tokenizer="bert"):
    """
    Get the embedding of a text using the specified tokenizer.
    This function assumes that the tokenizer has a method `encode` that returns the token IDs.
    """
    if tokenizer not in tokenizers:
        raise ValueError(f"Tokenizer '{tokenizer}' is not supported.")
    
    embeddings = models[tokenizer].embeddings.word_embeddings.weight[ids].detach().numpy()
    return embeddings

if __name__ == "__main__":
    # Example usage
    text = Text("Hola, ¿cómo estás?")
    print(text)
    formatted_input = format_input(text)
    print(formatted_input)
    embeddings = models["bert"].embeddings.word_embeddings.weight[text.tokens_ids].detach().numpy()
    print(embeddings)