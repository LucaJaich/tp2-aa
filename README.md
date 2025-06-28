## Como hacer un dataset para pytorch en el proyecto?

Voy a la carpeta datasets y creo un archivo (o elijo uno ya creado).
```python
from datasets import TextDataset, assign_classes

class TestTextDataset(TextDataset):
    def __init__(self, texts: list[str], tokenizer="bert"):
        super().__init__(texts, tokenizer)
        self.y = assign_classes(self.inputs["punt_final"])
        self.X = np.concatenate([text.embeddings for text in self.texts])
```

### Que atributos tiene TextDataset?
- `self.texts`: Agarra la lista de strings que le pasamos al inicializar el dataset y la convierte a una lista de la clase `Text`.
- `self.inputs`: Es la misma lista de strings, pero en un dataframe con el formato que pide la consigna. Ej le puedo pedir `self.inputs["punt_final"]` para obtener la columna de punt final.

### Que hace assign_classes?
`assign_classes` es una función que asigna clases numericas a una lista de caracteres.
Esto es útil para convertir etiquetas de texto en números, lo que es necesario para el entrenamiento de modelos de aprendizaje automático.
Por ejemplo, si tienes etiquetas como "positivo", "negativo" y "neutral", `assign_classes` las convertirá en 0, 1 y 2 respectivamente.

### Que atributos tiene Text?
(hecho con chatgpt)
| Atributo       | Tipo         | Descripción                                                                                                         |
| -------------- | ------------ | ------------------------------------------------------------------------------------------------------------------- |
| `text`         | `str`        | Texto original ingresado.                                                                                           |
| `tokens_raw`   | `list[str]`  | Tokens del texto incluyendo signos de puntuación.                                                                   |
| `tokens`       | `list[str]`  | Tokens sin puntuación, todos en minúscula.                                                                          |
| `tokens_ids`   | `list[int]`  | IDs numéricos de cada token según el tokenizer.                                                                     |
| `punt_inicial` | `list[str]`  | Lista con los signos de puntuación inicial asociados a cada token (por ejemplo `"¿"`).                              |
| `punt_final`   | `list[str]`  | Lista con los signos de puntuación final asociados a cada token (por ejemplo `"."`, `","`).                         |
| `cap`          | `list[int]`  | Capitalización del token original: <br>0 = minúscula, 1 = primera letra mayúscula, 2 = mezcla, 3 = todo mayúsculas. |
| `embeddings`   | `np.ndarray` | Vectores de embeddings obtenidos del modelo para cada token.                                                        |

