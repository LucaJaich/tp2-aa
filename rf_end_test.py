from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
# from sklearn.utils.validation import train_test_split
from torch.utils.data import random_split
from datasets.rf import WEndTextDataset
import json

# Load the dataset from a JSON file
print("Loading dataset...")
with open('datasets/data/film_reviews.json', 'r') as file:
    data = json.load(file)
print("Dataset loaded.")

print("Creating dataset...")
dataset = WEndTextDataset(data[:100], tokenizer="bert", window_size=2)
print("Dataset created.")
# print(dataset.X[0])

train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
# get all X from train_dataset
X_train = [item['X'] for item in train_dataset]
y_train = [item['y'] for item in train_dataset]
X_test = [item['X'] for item in test_dataset]
y_test = [item['y'] for item in test_dataset]

# # Create and train the Random Forest model
print("Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print("Model trained.")
# Evaluate the model
y_pred = rf_model.predict(X_test)
print(classification_report(y_test, y_pred))
