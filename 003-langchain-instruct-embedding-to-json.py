"""
Retrieved from https://python.langchain.com/en/latest/modules/models/text_embedding/examples/instruct_embeddings.html
"""

from langchain.embeddings import HuggingFaceInstructEmbeddings
import os
import json
import pathlib
import tomllib



# Initialize the model with the specified name and device
TORCH_CACHE = "F:/C/cache/torch/sentence_transformers"
SOURCE_DOCUMENT_PATH = "sources-documents/emails"

model_name = f"{{TORCH_CACHE}}/hkunlp_instructor-xl"
model_kwargs = {"device": "cuda"}
# Define the embedding instruction and query instruction to use
embed_instruction = "Please embed this document using the hkunlp_instructor-xl model."
query_instruction = "Please embed this query using the hkunlp_instructor-xl model."
# Create an instance of the HuggingFaceInstructEmbeddings class with the given instructions
hf = HuggingFaceInstructEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    embed_instruction=embed_instruction,
    query_instruction=query_instruction,
)

# Define the directory where the text files are located
directory = SOURCE_DOCUMENT_PATH

# Create an empty list to store the documents and their embeddings
documents_and_embeddings = []

# Iterate through each file in the directory
for filename in os.listdir(directory):
    # Check if the file is a text file
    if filename.endswith(".txt"):
        # Open the file and read its content
        with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
            content = f.read()
        print(f"Embedding {filename}\n")
        # Embed the content using the model and the embedding instruction
        embedding = hf.embed_documents([content])[0]
        # Create a dictionary with the embedding, content and filename as keys and values
        document_and_embedding = {
            "embedding": embedding,
            "text": content,
            "filename": filename,
        }
        # Add the dictionary to the list of documents and embeddings
        documents_and_embeddings.append(document_and_embedding)

# Embed a single query using the model and the query instruction
query = "What is the meaning of life?"
embedding = hf.embed_query(query)

print(embedding)


# Try to save the list of documents to a JSON file
try:
    with open("documents.json", "w") as json_file:
        # Dump the list of documents to the JSON file with 4 spaces for indentation
        json.dump(documents_and_embeddings, json_file, indent=4)
# Handle any errors that could occur while opening or writing to the JSON file
except OSError as e:
    # Log or print an error message with the JSON file name and the error details
    print(f"Error while saving documents.json: {e}")
