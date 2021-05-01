import urllib.request
import pathlib
import os
import logging
import zipfile
import pandas as pd
import json

def download_annotated_data(raw_data_folder):
    raw_data_folder = pathlib.Path(raw_data_folder)
    if not os.path.exists(raw_data_folder):
        os.makedirs(raw_data_folder)
        
    additional_data_zip_url = r"https://zenodo.org/record/4707104/files/Additional%20data.zip?download=1"
    decs2020_obo_url = r"https://zenodo.org/record/4707104/files/DeCS2020.obo?download=1"
    decs2020_tsv_url = r"https://zenodo.org/record/4707104/files/DeCS2020.tsv?download=1"
    subtrack1_scientific_literature_zip_url = r"https://zenodo.org/record/4707104/files/Subtrack1-Scientific_Literature.zip?download=1"
    subtrack2_clinical_trials_zip_url = r"https://zenodo.org/record/4707104/files/Subtrack2-Clinical_Trials.zip?download=1"
    subtrack_3_patents_zip_url = r"https://zenodo.org/record/4707104/files/Subtrack3-Patents.zip?download=1"
    
    additional_data_zip_name = "Additional data.zip"
    if not additional_data_zip_name in os.listdir(raw_data_folder):
        logging.info("downloading additional_data_zip")
        urllib.request.urlretrieve(additional_data_zip_url, raw_data_folder / additional_data_zip_name)
        
    decs2020_obo_name = "DeCS2020.obo"
    if not decs2020_obo_name in os.listdir(raw_data_folder): 
        logging.info("downloading decs2020_obo")
        urllib.request.urlretrieve(decs2020_obo_url, raw_data_folder / decs2020_obo_name)
        
    decs2020_tsv_name = "DeCS2020.tsv"
    if not decs2020_tsv_name in os.listdir(raw_data_folder): 
        logging.info("downloading decs2020_tsv")
        urllib.request.urlretrieve(decs2020_tsv_url, raw_data_folder / decs2020_tsv_name)
        
    subtrack1_scientific_literature_zip_name = "Subtrack1-Scientific_Literature.zip"
    if not subtrack1_scientific_literature_zip_name in os.listdir(raw_data_folder):
        logging.info("downloading subtrack1_scientific_literature_zip")
        urllib.request.urlretrieve(subtrack1_scientific_literature_zip_url, raw_data_folder / subtrack1_scientific_literature_zip_name)
        
    subtrack2_clinical_trials_zip_name = "Subtrack2-Clinical_Trials.zip"
    if not subtrack2_clinical_trials_zip_name in os.listdir(raw_data_folder):
        logging.info("downloading subtrack2_clinical_trials_zip")
        urllib.request.urlretrieve(subtrack2_clinical_trials_zip_url, raw_data_folder / subtrack2_clinical_trials_zip_name)
        
    subtrack_3_patents_zip_name = "Subtrack3-Patents.zip"
    if not subtrack_3_patents_zip_name in os.listdir(raw_data_folder):
        logging.info("downloading subtrack_3_patents_zip")
        urllib.request.urlretrieve(subtrack_3_patents_zip_url, raw_data_folder / subtrack_3_patents_zip_name)

def unzip_annotated_data(raw_data_folder):
    raw_data_folder = pathlib.Path(raw_data_folder)
    for filename in os.listdir(raw_data_folder):
        if filename.endswith(".zip"):
            with zipfile.ZipFile(raw_data_folder / filename, 'r') as zip_ref:
                zip_ref.extractall(raw_data_folder)

def load_dataset(raw_data_folder, subtrack = 1):
    raw_data_folder = pathlib.Path(raw_data_folder)
    if subtrack == 1:
        with open(raw_data_folder / "Subtrack1-Scientific_Literature/Train/training_set_subtrack1_all.json", encoding="utf-8") as j:
            train = pd.DataFrame.from_dict(json.load(j)["articles"])
        with open(raw_data_folder / "Subtrack1-Scientific_Literature/Development/development_set_subtrack1.json", encoding="utf-8") as j:
            development = pd.DataFrame.from_dict(json.load(j)["articles"])
        with open(raw_data_folder / "Subtrack1-Scientific_Literature/Test/test_set_subtrack1.json", encoding="utf-8") as j:
            test = pd.DataFrame.from_dict(json.load(j)["articles"])
    
    if subtrack == 2:
        with open(raw_data_folder / "Subtrack2-Clinical_Trials/Train/training_set_subtrack2.json", encoding="utf-8") as j:
            train = pd.DataFrame.from_dict(json.load(j)["articles"])
        with open(raw_data_folder / "Subtrack2-Clinical_Trials/Development/development_set_subtrack2.json", encoding="utf-8") as j:
            development = pd.DataFrame.from_dict(json.load(j)["articles"])
        with open(raw_data_folder / "Subtrack2-Clinical_Trials/Test/test_set_subtrack2.json", encoding="utf-8") as j:
            test = pd.DataFrame.from_dict(json.load(j)["articles"])
    
    if subtrack == 3:
        train = pd.DataFrame()
        with open(raw_data_folder / "Subtrack3-Patents/Development/development_set_subtrack3.json", encoding="utf-8") as j:
            development = pd.DataFrame.from_dict(json.load(j)["articles"])
        with open(raw_data_folder / "Subtrack3-Patents/Test/test_set_subtrack3.json", encoding="utf-8") as j:
            test = pd.DataFrame.from_dict(json.load(j)["articles"])
        
    return train, development, test