#Se instalan las dependencias necesarias
"""

#!pip install openai PyPDF2 tiktoken

"""
## Aqui se importaran librerias y se definiran variables que se usaran a lo largo del codigo"""

import PyPDF2
import numpy as np
import tiktoken
import openai
from scipy.spatial.distance import cosine

openai.api_key = ""
document = 'file.pdf' #Liga del PDF
max_tokens = 300
tokenizer = tiktoken.get_encoding("cl100k_base")

"""### La siguiente funcion es para saber de cuantas tokens se compone un texto"""

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

"""### La siguiente funcion es para escribir en un arreglo el contenido del PDF por paginas donde [[pagina]...[pagina_n]]:"""

def obtener_contenido_paginas(documento):
    with open(documento, 'rb') as pdf_file:
        pdf = PyPDF2.PdfReader(pdf_file)
        contenido_paginas = []
        for page_num in range(len(pdf.pages)):
            contenido = pdf.pages[page_num].extract_text()
            if contenido:
                # Agregar el contenido de la página como una sola cadena
                contenido_paginas.append(contenido.replace('\n', ''))
            else:
                # Agregar una cadena vacía si no hay contenido en la página
                contenido_paginas.append('')
        return contenido_paginas

"""### La siguiente funcion es para seprar en trozos un fragmento de texto dependiendo si excede o no un numero maximo de tokens dado"""

def split_into_many(text, max_tokens = max_tokens):

    # Split the text into sentences
    sentences = text.split('.')

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    
    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater 
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of 
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks

"""#### Una vez definidas estas funciones, debemos convertir el documento a la lista de 2 dimensiones, y en otra lista (shortened) agregaremos todos los strings que no superen el numero maximo de tokens, y los que lleguen a superar el numero maximo de tokens se pasara a la funcion split_into_many para que se parta y luego se agrege a la lista shortened


"""

contenido_paginas = obtener_contenido_paginas(document) #Se crea arreglo para cada pagina
shortened= []

# Iterar para cada pagina y si rebasa el max tokens se dividira en pedasos mas pequenios
for i in range(len(contenido_paginas)):
  if num_tokens_from_string(contenido_paginas[i], "cl100k_base") > max_tokens:
    shortened += split_into_many(contenido_paginas[i])
  else:
    shortened.append(contenido_paginas[i])

"""### Ya que tenemos una lista con un numero de tokens abajo de nuestro maximo de tokens, debemos crear los embedding de cada elemento en shortened

### Estos embeddings vienen en formato JSON, por lo que para saber el resultado del vector resultante y este lo agregaremos a nuestro arreglo embeddings_pdf

## Este proceso se repetira para cada elemento que hay en la lista shortened
"""

# Crear una lista para almacenar los embeddings de cada fragmento
embeddings_pdf = []

# Iterar sobre cada fragmento de texto y crear un embedding para cada uno
for chunk in shortened:
    # Crear un embedding del fragmento de texto actual
    embedding = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=chunk, 
        temperature=0.5,
        max_tokens=1024
    )
    
    # Obtener el vector de embeddings del fragmento de texto actual
    vector = np.array(embedding['data'][0]['embedding']) 
    # Agregar el vector de embeddings a la lista de embeddings
    embeddings_pdf.append(vector)

"""### Una vez teniendo nuestros embeddings agregados a embeddings_pdf, los comvertiremos a una matriz para que sea mas facil su manipulacion mas adelante."""

# Convertir la lista de embeddings en una matriz NumPy
embeddings_pdf = np.array(embeddings_pdf)

"""### Una vez teniendo los embeddings del PDF, necesitaremos los embeddings de la entrada del usuario para poderlos comparar"""

#Entrada del usuario, pregunta respecto al PDF
input_text = input('Cual es tu pregunta respecto al PDF?: ')

embedding_input = openai.Embedding.create(
    model="text-embedding-ada-002",
    input=input_text,
    temperature=0.5,
    max_tokens=1024
)

# Obtener la matriz de embeddings del input

vector_input = np.array(embedding_input['data'][0]['embedding'])

"""### Se calcula la similitud entre los dos embeddings, para encotnrar la relacion y el fragmento de texto donde se pueda encontrar la solicitud del usuario."""

# Calcular la similitud del coseno entre el vector de entrada y los vectores de los fragmentos de texto
similarities = []
for vector in embeddings_pdf:
    similarity = 1 - cosine(vector, vector_input)
    similarities.append(similarity)

# Obtener el índice del fragmento de texto con la mayor similitud
max_similarity_idx = np.argmax(similarities)

# Obtener el fragmento de texto con la mayor similitud
most_similar_chunk = shortened[max_similarity_idx] #Si la respuesta esta entre dos chunks que pasa?

"""###Se hace la peticion para una respuesta con lenguaje natural respecto al PDF y la solicitud del usuario. Se imprime la respuesta."""

# Generar una respuesta natural 
response = openai.Completion.create(
    model="text-curie-001", #text-ada-001 ada
    prompt=f"El fragmento de texto más relevante en el PDF es: '{most_similar_chunk}'. ¿Puedes responder a mi pregunta sobre el PDF?: '{input_text}'",
    temperature=0.5, #aleatoriedad del texto generado de 0 - 1 donde 1: mas creativo
    max_tokens=1024, #max tokens de respuesta 
    n=1, #cuantas respuestas diferentes se generan
    stop=None #palabras o frases que detienen la generacion de texto
)

# Obtener la respuesta generada del modelo
generated_response = response['choices'][0]['text'].strip()

#Respuesta final
print(generated_response)

#Problema, no fucniona con PDF con copyrigth