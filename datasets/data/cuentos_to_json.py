from datasets import load_dataset
import json
import re
import os


def clean_text(text):
    # Remove everything except letters, commas, dots, and question marks (opened and close). NO NUMBERS
    text = re.sub(r"[^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ,.\?\¿]", " ", text)
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)
    # Remove leading and trailing spaces
    text = text.strip()
    # change enters to spaces
    text = text.replace("\n", " ")
    return text


ds = load_dataset("Fernandoefg/cuentos_es", split="train")
print(len(ds["content"]))
# keep only letters, commas, dots, and question marks with regex


# get file path
my_path = os.path.dirname(__file__)
# create cuentos directory if it doesn't exist
if not os.path.exists(os.path.join(my_path, "cuentos")):
    os.makedirs(os.path.join(my_path, "cuentos"))

my_path = os.path.join(my_path, "cuentos")

with open(os.path.join(my_path, "cuentos_cleaned_dummy.json"), "w") as f:
    out = clean_text(ds["content"][0]).split(".")
    out = [s.strip() + "." for s in out if s.strip()]  # Add dot back and remove empty strings
    json.dump(out, f, ensure_ascii=False, indent=4)

with open(os.path.join(my_path, "cuentos_cleaned_train.json"), "w") as f:
    train_out = []
    for i in range(60):
        sentences = clean_text(ds["content"][i]).split(".")
        sentences = [s.strip() + "." for s in sentences if s.strip()]  # Add dot back and remove empty strings
        train_out += sentences
    json.dump(train_out, f, ensure_ascii=False, indent=4)

with open(os.path.join(my_path, "cuentos_cleaned_test.json"), "w") as f:
    test_out = []
    for i in range(1):
        sentences = clean_text(ds["content"][i]).split(".")
        sentences = [s.strip() + "." for s in sentences if s.strip()]  # Add dot back and remove empty strings
        test_out += sentences
    json.dump(test_out, f, ensure_ascii=False, indent=4)
    

print("TRAIN_SIZE:", len(train_out))
print("TEST_SIZE:", len(test_out))