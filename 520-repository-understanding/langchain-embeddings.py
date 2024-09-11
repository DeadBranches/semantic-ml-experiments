import os
import pprint
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import DeepLake

model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

text = "This is a test document."
query_result = embeddings.embed_query(text)
doc_result = embeddings.embed_documents([text])

# pprint(query_result)