import json
import os
import pathlib
import tomllib

import numpy as np
from langchain.embeddings import HuggingFaceInstructEmbeddings
from scipy.spatial.distance import cosine

#
# Global constant definitions
#
# INPUT_FILENAME
#   JSON file containing document embeddings produced by instructor-xl in a previous
#   step.
#
# OUTPUT_FILENAME_SUFFIX
#   This project produces a list of documents most similar to the query string. Append
#   this string to a json file containing the list of documents. 
#
SOURCE_DOCUMENT_DIRECTORY = ""
INPUT_FILENAME = "110-embed-with-metadata-keywords-document-embeddings.json"
EMBEDDING_MODEL_NAME = "hkunlp_instructor-xl"
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
    
cache_paths = global_setting["cache_path"]
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


try:
    with open(INPUT_FILENAME, "r", encoding="utf8") as f:
        documents_and_embeddings = json.load(f)
except OSError as e:
    print(f"Error while opening or reading {INPUT_FILENAME}: {e}")


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
    model_kwargs=model_kwargs,
    embed_instruction=embed_instruction.get("document"),
    query_instruction=embed_instruction.get("query"),
)

#
# Create embeddings
#
query_embedding = hf.embed_query(query.get("cognitive_impairment"))

similarity_scores_and_filenames = []
for document_and_embedding in documents_and_embeddings:
    filename = document_and_embedding["filename"]
    document_embedding = document_and_embedding["embedding"]
    
    similarity_score = 1 - cosine(query_embedding, document_embedding)
    similarity_score_and_filename = (similarity_score, filename)
    similarity_scores_and_filenames.append(similarity_score_and_filename)

# Sort the list of similarity scores and filenames in descending order by
# similarity score
similarity_scores_and_filenames.sort(key=lambda x: x[0], reverse=True)


try:
    with open(
        os.path.join(
            output_filename_prefix, f"{OUTPUT_FILENAME_SUFFIX}.json"),
            "w",
            encoding="utf8") as json_file:
        json.dump(similarity_scores_and_filenames, json_file, indent=4)
except OSError as e:
    print(f"Error while saving documents.json: {e}")



# Print the sorted list of similarity scores and filenames
print(similarity_scores_and_filenames)
