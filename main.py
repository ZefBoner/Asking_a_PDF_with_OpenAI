##Se instalan las dependencias necesarias
"""

#!pip install pypdf2 tiktoken openai

"""##Se importan las librerias necesarias."""

import pandas as pd
import csv, os, PyPDF2, tiktoken, openai, io, requests
import numpy as np
from openai.embeddings_utils import distances_from_embeddings

"""##Definimos la clave privada para la conexion con openai, ademas de definir los tokenizer"""

openai.api_key = ""
# Load the cl100k_base tokenizer which is designed to work with the ada-002 model
tokenizer = tiktoken.get_encoding("cl100k_base")

# Verificar si el archivo CSV existe
if os.path.exists('pdf_data.csv'):
    # Leer el archivo CSV y cargar los datos en un DataFrame de Pandas
    pdf_df = pd.read_csv('pdf_data.csv')
else:
    # Si el archivo no existe, crear un DataFrame vacío de Pandas
    pdf_df = pd.DataFrame(columns=['Nombre de archivo', 'Contenido de texto'])

"""##Funcion recursiva para preguntar al usuario si subira un archivo o seleccionara una url"""

def url_o_cargar():
  res = input('Escribe \'url\' si el archivo PDF a interactuar esta en una url de google drive, o \'cargar\' si se cargara el documento:  ')
  if res.lower() == 'url' or res.lower() == 'cargar':
    return res.lower()
  else:
    print('Tu seleccion no es valida, selecciona entre \'url\' o \'cargar\'')
    url_o_cargar()

"""##Funcion para transformar una url de google drive con un pdf a texto"""

def pdf_to_text_google_drive(url):
  file_id = url.split("/")[-2]
  response = requests.get(f"https://drive.google.com/uc?id={file_id}&export=download")
  pdf_file = io.BytesIO(response.content)# Se escribe en binario el contenido de response

  pdf_reader = PyPDF2.PdfReader(pdf_file) 
  text = ''
  for page in range(len(pdf_reader.pages)):
    text += pdf_reader.pages[page].extract_text()
  return text

"""##Funcion para transformar un pdf a texto"""

def text_to_pdf(pdf_path):
  with open(pdf_path, 'rb') as f:
    pdf_reader = PyPDF2.PdfReader(f)
    text = ''
    for page in range(len(pdf_reader.pages)):
      text += pdf_reader.pages[page].extract_text()
  return text

"""##Aqui se definen los valores de texto y de url de ser necesario"""

url_o_local = url_o_cargar()
if url_o_local == 'url':
  url = input('Introduce la url del pdf, SOLO URL GOOGLE DRIVE: ')
  nombre_archivo = input('Introduce el nombre del archivo PDF:  ')
  text = pdf_to_text_google_drive(url)
else:
  nombre_archivo = input('Introduce el nombre del archivo pdf ya cargado(con extension):  ')
  text = text_to_pdf(nombre_archivo)

"""Se agregan los valores anteriores al csv creado al principio"""

# Agregar una nueva fila al DataFrame
#pdf_df = pdf_df.append({'Nombre de archivo': nombre_archivo, 'Contenido de texto': text}, ignore_index=True)
pdf_df = pd.concat([pdf_df, pd.DataFrame({'Nombre de archivo': [nombre_archivo], 'Contenido de texto': [text]})], ignore_index=True)

pdf_df.head()

# Guardar los datos actualizados en el archivo CSV
pdf_df.to_csv('pdf_data.csv', index=False)

# Leer el archivo CSV con los datos cargados
df = pd.read_csv('pdf_data.csv')

# Tokenizar el texto y guardar el número de tokens en una nueva columna
df['n_tokens'] = df['Contenido de texto'].apply(lambda x: len(tokenizer.encode(x)))

# Visualizar la distribución del número de tokens por fila usando un histograma
df['n_tokens'].hist()

"""##Funcion para hacer limpieza de un string"""

def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie

"""##Dividimos el texto en la columna de texto generado en diferentes filas con su respectivo numero de tokens"""

max_tokens = 1000

# Function to split the text into chunks of a maximum number of tokens
def split_into_many(text, max_tokens = max_tokens):

    # Split the text into sentences
    sentences = text.split('. ')

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
    

shortened = []

# Loop through the dataframe
for row in df.iterrows():

    # If the text is None, go to the next row
    if row[1]['Contenido de texto'] is None:
        continue

    # If the number of tokens is greater than the max number of tokens, split the text into chunks
    if row[1]['n_tokens'] > max_tokens:
        shortened += split_into_many(row[1]['Contenido de texto'])
    
    # Otherwise, add the text to the list of shortened texts
    else:
        shortened.append(row[1]['Contenido de texto'] )

df = pd.DataFrame(shortened, columns = ['Contenido de texto'])
# Tokenizar el texto y guardar el número de tokens en una nueva columna
df['n_tokens'] = df['Contenido de texto'].apply(lambda x: len(tokenizer.encode(x)))

# Visualizar la distribución del número de tokens por fila usando un histograma
df['n_tokens'].hist()

df.tail() #Se comprueban las ultimas 5 filas para comprobar informacion

"""##Se crean embeddings para cada fraccion de texto"""

df['embeddings'] = df['Contenido de texto'].apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])

df.to_csv('embeddings.csv')
df.head()

df=pd.read_csv('embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array) #Los embeddings los hacemos vectores para numpy

df.head()

def create_context(
    question, df, max_len=1000, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        
        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4
        
        # If the context is too long, break
        if cur_len > max_len:
            break
        
        # Else add it to the text that is being returned
        returns.append(row["Contenido de texto"])

    # Return the context
    return "\n\n###\n\n".join(returns)

def answer_question(
    df,
    model="text-davinci-003",
    question='',
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=300,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the question and context
        response = openai.Completion.create(
            prompt=f"Answer the question based on the context below,and if the question can't be answered based on the context, say \"I don't know\ \n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0.4,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""

pregunta_pdf = input('Escribe tu pregunta al PDF: ')
answer_question(df,"text-davinci-003",pregunta_pdf)