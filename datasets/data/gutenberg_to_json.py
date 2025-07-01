from datasets import load_dataset
import os
import json
import re

MAX_TEXTS = 50

def remove_gutenberg_boilerplate(text):
    """
    Intenta remover el encabezado y pie de los textos de Gutenberg.
    """
    start_match = re.search(r"\*\*\* START OF.*?\*\*\*", text, re.IGNORECASE)
    end_match = re.search(r"\*\*\* END OF.*?\*\*\*", text, re.IGNORECASE)

    start = start_match.end() if start_match else 0
    end = end_match.start() if end_match else len(text)

    return text[start:end].strip()

def save_to_json(text: str, output_dir: str, i: int):
    """
    Divide el texto en párrafos (usando dobles saltos de línea) y guarda como JSON.
    """
    # limpiar el texto de Gutenberg
    text = remove_gutenberg_boilerplate(text)

    # divido por párrafo para no cargar en memoria todo el texto
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip().replace('\n', ' ') for p in paragraphs if p.strip()]

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"text_{i}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(paragraphs, f, ensure_ascii=False, indent=2)

def main():    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "gutenberg")  
    os.makedirs(output_dir, exist_ok=True)

    dataset = load_dataset("manu/project_gutenberg", split="es", streaming=True)

    print('Lodaded dataset')

    for i, example in enumerate(dataset):
        #tarda mucho el primero, despues arranca

        if i >= MAX_TEXTS:
            break
        
        print('Processing example', i)
        full_text = example["text"]

        save_to_json(full_text, output_dir, i)


if __name__ == "__main__":
    main()
