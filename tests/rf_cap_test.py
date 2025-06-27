from sklearn.ensemble import RandomForestClassifier
# from sklearn.utils.validation import train_test_split
from torch.utils.data import random_split
from datasets import CapTextDataset
from utils import Text

dataset = CapTextDataset(["Hola mi nombre es luca. Estoy intenando hacer un nuevo modelo de KNN que no tenia", "Hola, mi nombre es Juan y estoy encantado de conocerte. ¿Cómo estás?"], tokenizer="bert")

train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
print(test_dataset)
# get all X from train_dataset
X_train = [item['X'] for item in train_dataset]
y_train = [item['y'] for item in train_dataset]
X_test = [item['X'] for item in test_dataset]
y_test = [item['y'] for item in test_dataset]

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
# Evaluate the model with aucroc
from sklearn.metrics import accuracy_score, classification_report
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

# prediction
text = "Estoy bastante cansado, no quiero saber nada."
textObj = Text(text, tokenizer="bert")
text_embeddings = textObj.embeddings
text_cap = textObj.cap

predicted_cap = clf.predict(text_embeddings)
print(f"Predicted cap for the text: {predicted_cap}")
print(f"Actual cap for the text: {text_cap}")
print(f"Text: {text}")
