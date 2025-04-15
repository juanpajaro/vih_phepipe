import os
import json
import datetime
import csv
import re

#Function to create a directory to save models if it does not exist
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("the directory was created {}".format(path))
    else:
        print("the directory already exists {}".format(path))
    print("save the data in {}".format(path))
    return path

#function to save a json file based on a list of dictionary (e.g., list of patients)
def save_json(data, path_data_save):
    data_json = json.dumps(data, indent=2)
    with open(path_data_save, "w") as file:
        file.write(data_json)
    print("json file saved")
    return None

#function to get the date and time
def get_time():    
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#function to extract the name of h5 model in a directoy
def extract_name_model(path, extension):
    list_models = os.listdir(path)
    list_models = [model for model in list_models if model.endswith(extension)]
    return list_models

#function with regular expression to extract the number of the version return a string
def extract_number_version(string):
    number = re.findall(r'\d+', string)
    if number == []:
        number = [0]
    return number[0]

#function to extract the last version of h5 model in a directory
def extract_last_version_model(list_models):
    list_versions = []
    for model in list_models:
        version_numbers = int(extract_number_version(model))
        list_versions.append(version_numbers)
    #print(list_versions)
    #print(type(list_versions))
    if len(list_versions) == 0:
        last_version = 0
    else:
        last_version = max(list_versions)
    return last_version

#function to versionate the model
def counter_version(num_version):
    num_version += 1
    return num_version

#function to load a json file
def load_json(path_data_save, name_file):
    with open(path_data_save + name_file, "r") as file:
        data_json = file.read()
        data = json.loads(data_json)
    #print("json file loaded")
    return data

def load_mapping_icd(file_path, name_file):
    with open(file_path + name_file, 'r') as input_file:
        reader = csv.DictReader(input_file)
        #print(type(reader))
        lista_example_umls_icd = list(reader)    
        input_file.close()
    print("mapping icd10-umls loaded")
    return lista_example_umls_icd

def get_current_path():
    current_path = os.getcwd()
    print("current path is {}".format(current_path))
    return current_path

def directories_in_path(path):
    if not os.path.exists(path):
        create_directory(path)
        directories = os.listdir(path)
        directories = [directory for directory in directories if os.path.isdir(path + directory)]
    else:
        directories = os.listdir(path)
        directories = [directory for directory in directories if os.path.isdir(path + directory)]
        
    return directories

def load_hyperparams_as_tuples(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    return [
        (            
            entry['embedding_dim'],
            entry['block_layers'],
            entry['hidden_units'],
            entry['learning_rate'],
            entry['epochs'],
            entry['batch_size']
        )
        for entry in data
    ]
