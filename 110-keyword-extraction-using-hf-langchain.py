import json
import os
import pathlib
import tomllib

import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings

#
# Global constant definitions
#
SOURCE_DOCUMENT_DIRECTORY = "emails"
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
OUTPUT_FILENAME_SUFFIX = "results"

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
PROJECT_CONFIG_FILENAME = "100-housing.toml"


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
with open(
    os.path.join(CONFIG_DIRECTORY_PATH, PROJECT_CONFIG_FILENAME),
    "rb"
    ) as f:
    project_setting = tomllib.load(f)
    
model_paths = global_setting["model_path"]
project_paths = global_setting["project_path"]


#
# Misc variable initialization
#
# output_filename
#   Use the current python file name as a prefix for output files.
#
output_filename_prefix = pathlib.PurePath(__file__).stem


#
# Model initialization
# 

model_name = os.path.join(
    model_paths.get("sentence_transformers"),
    EMBEDDING_MODEL_NAME)
model_kwargs = {'device': 'cuda'}

#
# Embedding creation
#
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs)

document_directory = os.path.join(
    project_paths.get("source_document_root"),
    SOURCE_DOCUMENT_DIRECTORY)

documents_and_keywords = []

for filename in os.listdir(document_directory):
    if filename.endswith(".txt"):
        with open(
            os.path.join(document_directory, filename),
            "r",
            encoding="utf-8") as f:
            
            content = f.read()
            print(f"Extracting keywords from {filename}\n")
            
            words = content.split()
        
            # Embed each word using the model and the embed_documents method
            # Create a dictionary with words as keys and embeddings as values using zip
            # Filter out words that have negative or zero embeddings using numpy
            embeddings = hf.embed_documents(words)
            word_embedding = dict(zip(words, embeddings))
            keywords = [
                word for word,
                value in word_embedding.items() if np.any(np.array(value) > 0)
                ]
            
            document_and_keyword = {
                "filename": filename,
                "keywords": keywords
            }
            
            documents_and_keywords.append(document_and_keyword)

try:
    with open(
        os.path.join(
            output_filename_prefix, f"{OUTPUT_FILENAME_SUFFIX}.json"),
            "w",
            encoding="utf8") as json_file:
        json.dump(documents_and_keywords, json_file, indent=4)
except OSError as e:
    print(f"Error while saving documents.json: {e}")
