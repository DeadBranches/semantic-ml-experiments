import os
from langchain.document_loaders import PyMuPDFLoader

#os.environ['OPENAI_API_KEY'] = 'ENTER YOUR API KEY'

def pdf_loader(filepath: str):
    loader = PyMuPDFLoader(filepath)
    documents = loader.load()
    return documents