from models.rnn import RNN
from utils import Text, format_input
import torch

INPUT_SIZE = 768  # por bert cases
HIDDEN_DIM = 64  # hidden dimension size
NUM_LAYERS = 1
BIDIRECTIONAL = True

PATH="./models/checkpoints/exported_program_brnn.pt2"

model = RNN(INPUT_SIZE, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, bidirectional=BIDIRECTIONAL)
model.load_state_dict(torch.load(PATH, weights_only=True))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(DEVICE)
model.eval()

TEXTO = "El perro ladra, el gato maulla y el pez nada."
text = Text(TEXTO)
print(format_input(text, id=0)) 

X = torch.tensor(text.embeddings).to(DEVICE)  # Añadir batch dimension y mover a DEVICE
print(model(X))
