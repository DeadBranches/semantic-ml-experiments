"""
Retrieved from https://python.langchain.com/en/latest/modules/models/text_embedding/examples/instruct_embeddings.html
"""

import json
import os
import pathlib
import tomllib

from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer
from langchain.embeddings import HuggingFaceInstructEmbeddings

#
# Global constant definitions
#
# SOURCE_DOCUMENT_DIRECTORY
#   The subdirectory relative to source_documents containing documents to use in
#   this project file.
#
# OUTPUT_FILENAME_SUFFIX
#   This project produces document embeddings from input files. When saving the
#   generated embeddings, append this string to the filename. 
# 
SOURCE_DOCUMENT_DIRECTORY = "emails"
EMBEDDING_MODEL_NAME = "hkunlp_instructor-xl"
OUTPUT_FILENAME_SUFFIX = "document-embeddings"


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
    
cache_paths = global_setting["cache_path"]
project_paths = global_setting["project_path"]
embed = project_setting["embed"]
embed_instruction = embed["instruction"]
query = embed["query_strings"]


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
    cache_paths.get("sentence_transformers"),
    EMBEDDING_MODEL_NAME)
model_kwargs = {"device": "cuda"}

#
# Embedding creation
#
hf = HuggingFaceInstructEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    embed_instruction=embed_instruction.get("document"),
    query_instruction=embed_instruction.get("query"),
)
kw_model = KeyBERT()

document_directory = os.path.join(
    project_paths.get("source_document_root"),
    SOURCE_DOCUMENT_DIRECTORY)

documents_and_embeddings = []

for filename in os.listdir(document_directory):
    if filename.endswith(".txt"):
        with open(
            os.path.join(document_directory, filename),
            "r",
            encoding="utf-8") as f:
            content = f.read()
        print(f"Embedding {filename}\n")
        
        embedding = hf.embed_documents([content])[0]

        # Keybert
        keyword_object = kw_model.extract_keywords(
            docs=content,
            vectorizer=KeyphraseCountVectorizer())
        keywords = [item[0] for item in keyword_object]
        keywords = ", ".join(keywords)

        document_and_embedding = {
            "filename": filename,
            "text": content,
            "keywords": keywords,
            "embedding": embedding,
        }
        documents_and_embeddings.append(document_and_embedding)

# Embed a single query using the model and the query instruction
embedding = hf.embed_query(query.get("accommodations_requested"))

print(embedding)


try:
    with open(
        os.path.join(
            output_filename_prefix, f"{OUTPUT_FILENAME_SUFFIX}.json"),
            "w",
            encoding="utf8") as json_file:
        json.dump(documents_and_embeddings, json_file, indent=4)
except OSError as e:
    print(f"Error while saving documents.json: {e}")
