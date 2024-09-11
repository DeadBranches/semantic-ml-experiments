import json
import os
import pathlib
import tomllib

from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer

# https://github.com/TimSchopf/KeyphraseVectorizers
# https://scribe.rip/enhancing-keybert-keyword-extraction-results-with-keyphrasevectorizers-3796fa93f4db


#
# Global constant definitions
#
# OUTPUT_FILENAME_SUFFIX
#   This project produces a list of document filenames and associated keywords. Append
#   this string to the end of the filname when saving.
#
SOURCE_DOCUMENT_DIRECTORY = "emails"
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
project_paths = global_setting["project_path"]


#
# Misc variable initialization
#
# output_filename
#   Use the current python file name as a prefix for output files.
#
output_filename_prefix = pathlib.PurePath(__file__).stem

document_directory = os.path.join(
    project_paths.get("source_document_root"),
    SOURCE_DOCUMENT_DIRECTORY)


kw_model = KeyBERT()

documents_and_keywords = []
for filename in os.listdir(document_directory):
    if filename.endswith(".txt"):
        with open(
            os.path.join(document_directory, filename),
            "r",
            encoding="utf-8") as f:
            
            content = f.read()
            print(f"Extracting keywords from {filename}\n")
            keywords = kw_model.extract_keywords(
                docs=content,
                vectorizer=KeyphraseCountVectorizer())
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
