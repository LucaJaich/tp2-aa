from datasets import load_dataset
import os
import json

MAX_TEXTS = 3

def save_to_json(processed_text: str, output_dir: str, i: int):
    """
    Divide un texto en párrafos y lo guarda como una lista JSON en text_{i}.json.
    
    - processed_text: string largo (texto completo)
    - output_dir: directorio donde guardar los archivos
    - i: índice del archivo
    """
    os.makedirs(output_dir, exist_ok=True)

    paragraphs = [p.strip() for p in processed_text.split("\n") if p.strip()]

    filename = os.path.join(output_dir, f"text_{i}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(paragraphs, f, ensure_ascii=False, indent=2)

def main():    
    output_dir = "gutenberg"
    os.makedirs(output_dir, exist_ok=True)

    dataset = load_dataset("manu/project_gutenberg", split="es", streaming=True)

    print('Lodaded dataset')

    for i, example in enumerate(dataset):

        if i >= MAX_TEXTS:
            break
        
        full_text = example["text"][:3000]

        print(f"Processing text {i+1}/{MAX_TEXTS}: {full_text[:50]}...")

        # Encontrar las posiciones de las apariciones de '***'
        parts = full_text.split("***")

        if len(parts) >= 6:
            # sacar licencia y aclaraciones gutenberg
            processed_text = parts[2]
        else:
            pass

        save_to_json(processed_text, output_dir, i)

if __name__ == "__main__":
    main()
