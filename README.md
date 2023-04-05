# Asking_a_PDF_with_OpenAI
A python code wich open a PDF and do a connection to OpenAI to answer questions about the PDF, the PDF split in chunks and then openAI embedd all of the chunks,
then we ask a question to a user and we embedd the question, then we compare the vectors of both embeddings, then we need to input to openAI the most similar
part of the PDF where the answer of the question is, GPT read the part of the text and the question and generates an answer.
