#!/usr/bin/env python3
import os
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import utils_general_porpose
#import utils_train_models
import numpy as np

def load_data(path, filename):
    """
    Load the data from a JSON file.
    """
    data = utils_general_porpose.load_json(path, filename)
    print("Data loaded")
    return data

def get_labels(data):
    """
    Get the labels from the dataset.
    """
    labels = [patient.get("label") for patient in data]
    return labels

def get_data_to_tensor_string(data):
    """
    Convert the entire dataset to tensors.
    """
    tensor_data = []
    for patient in data:
        seq_tensor = patient['seq']
        tensor_data.append(seq_tensor)
        #tensor_data = np.vstack(np.ravel(seq_tensor))
    return tensor_data

def get_vectorized_layer(X_train, max_tokens, max_len):
    """
    Create and adapt a TextVectorization layer.
    """
    vectorize_layer = TextVectorization(
        max_tokens=max_tokens,
        standardize=None,
        output_mode='int',
        output_sequence_length=max_len)
    vectorize_layer.adapt(X_train)
    return vectorize_layer

current_path = os.getcwd()
#print(current_path)
filename = ["train", "test"]
#print(filename[0])
filename_train = "/" + filename[0] + "/" + filename[0] +"_20250408_144607.json"
train = load_data(current_path, filename_train)
print("Train data loaded")
print(len(train))
print(train[0])
print(train[0]["seq"])

train_string = get_data_to_tensor_string(train)
print("data_to_tensor_string:", train_string[:2])

encoder = get_vectorized_layer(train_string, max_tokens=5000, max_len=4)
vocab = np.array(encoder.get_vocabulary())
print("Vocabulary:", vocab[:100])
print("Vocabulary size:", len(vocab))
print(type(encoder))
X_train = encoder(train_string)
print("X_token:", X_train[:2])
print("X_token shape:", X_train.shape)
print(type(X_train))
X_train_np = np.array(X_train)
print(X_train_np.shape)

filename_test = "/" + filename[1] + "/" + filename[1] +"_20250408_144607.json"
test = load_data(current_path, filename_test)
print("Test data loaded")
print(len(test))

test_s = get_data_to_tensor_string(test)
print("data_to_tensor_string:", test_s[:2])
X_test = encoder(test_s)
print("X_test_token:", X_test[:2])


hyper_paramts_lstm = utils_general_porpose.load_json(current_path, "/models_parameters/hyper_params_lstm.json")
print("Hyperparameters loaded")
print(hyper_paramts_lstm)
print(type(hyper_paramts_lstm))

y_train = get_labels(train)
y_test = get_labels(test)
print("Labels created")
print("y_train:", y_train[0])
print("y_train:", y_train[1])
print("y_test:", y_test[0])
print("y_test:", y_test[1])

path_model_save = os.path.join(current_path, "models", "lstm_model")

y_train = np.array(y_train)
y_test = np.array(y_test)
print(X_train[:1])

import utils_train_models
now, name, acc, loss, model, num_classes = utils_train_models.train_lstm_model(X_train, y_train, X_test, y_test, hyper_paramts_lstm, path_model_save)