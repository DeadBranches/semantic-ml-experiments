"""
Retrieved from https://python.langchain.com/en/latest/modules/models/text_embedding/examples/instruct_embeddings.html
"""
import os
import tomllib

from langchain.embeddings import HuggingFaceInstructEmbeddings

EMBEDDING_MODEL_NAME = "hkunlp_instructor-xl"

#
# Configuration file settings
#
# GLOBAL_CONFIG_FILENAME
#   Some settings, such as cache paths, are common within the project.
#
# PROJECT_CONFIG_FILENAME
#   Settings specific to this project file. May include sensitive information.
# 
CONFIG_DIRECTORY_PATH = ".config"
GLOBAL_CONFIG_FILENAME = "000-global-settings.toml"

#
# Variable initialization from configuration values.
#
# Load project settings from file and assign the needed values to variables.
#
with open(
    os.path.join(CONFIG_DIRECTORY_PATH, GLOBAL_CONFIG_FILENAME),
    "rb"
    ) as f:
    global_setting = tomllib.load(f)
    
cache_paths = global_setting["cache_path"]


#
# Model initialization
#
# model_name
#   Is a local path to the embedding model.
model_name = os.path.join(
    cache_paths.get("sentence_transformers"),
    EMBEDDING_MODEL_NAME)
model_kwargs = {"device": "cuda"}

hf = HuggingFaceInstructEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs)


#
# Create embeddings
#
documents = [
    "This is the first document.",
    "This is the second document."]
embeddings = hf.embed_documents(documents)

query = "What is the meaning of life?"
embedding = hf.embed_query(query)

print(embedding)