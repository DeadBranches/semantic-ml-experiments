import os
import tomllib

from langchain.embeddings import HuggingFaceInstructEmbeddings

#
# Global constant definitions
#
# SOURCE_DOCUMENT_DIRECTORY
#   The subdirectory relative to source_documents containing documents to use in
#   this project file.
SOURCE_DOCUMENT_DIRECTORY = "emails"
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


document_directory = os.path.join(
    project_paths.get("source_document_root"),
    SOURCE_DOCUMENT_DIRECTORY)

documents = []
for dirpath, dirnames, filenames in os.walk(document_directory):
    for file in filenames:
        if file.endswith(".txt"):
            with open(
                os.path.join(document_directory, file),
                "r",
                encoding="utf-8") as f:
                document = {"filename": file, "embedding": hf.embed_document(f.read())}
                documents.append(document)
