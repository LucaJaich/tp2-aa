from sklearn.ensemble import RandomForestClassifier
# from sklearn.utils.validation import train_test_split
from torch.utils.data import random_split
from datasets import BeginningTextDataset
import json

# Load the dataset from a JSON file
with open('datasets/data/film_reviews.json', 'r') as file:
    data = json.load(file)

dataset = BeginningTextDataset(data, tokenizer="bert")

# HAY QUE HACER GROUP PARA SPLIT, NO PUEDEN QUEDAR SEPARADOS LOS TOKENS
train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
# get all X from train_dataset
X_train = [item['X'] for item in train_dataset]
y_train = [item['y'] for item in train_dataset]
print(y_train)
X_test = [item['X'] for item in test_dataset]
y_test = [item['y'] for item in test_dataset]

# Create and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
# Evaluate the model
accuracy = rf_model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

