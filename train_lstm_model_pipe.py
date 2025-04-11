#!/usr/bin/env python3
import os
import utils_general_porpose
import utils_train_models
import numpy as np

def load_data(path, filename):
    """
    Load the data from a JSON file.
    """
    data = utils_general_porpose.load_json(path, filename)
    print("Data loaded")
    return data

def get_vocab(train):
    """
    Get the vocabulary from the training data.
    """
    Vocab = {'__PAD__': 0, '__UNK__': 1}    
    for patient in train:
        for concept in patient['seq'].split():
            if concept not in Vocab:
                Vocab[concept] = len(Vocab)
    print("Vocabulary created")
            
    return Vocab

def seq_to_tensor(seq, vocab, unk_token='__UNK__', verbose=False):
    """
    Convert a sequence of concepts to a tensor.
    """
    seq_tensor = []
    for token in seq.split():
        if token in vocab:
            seq_tensor.append(vocab[token])            
        else:
            seq_tensor.append(vocab[unk_token])            
    if verbose:
        print("seq_to_tensor:", seq_tensor)
    return np.array(seq_tensor)
    
def data_to_tensor(data, vocab, unk_token='__UNK__', verbose=False):
    """
    Convert the entire dataset to tensors.
    """
    tensor_data = []
    for patient in data:
        seq_tensor = seq_to_tensor(patient['seq'], vocab, unk_token, verbose)
        tensor_data.append(seq_tensor)
        #tensor_data = np.vstack(np.ravel(seq_tensor))
    return tensor_data

def get_labels(data):
    """
    Get the labels from the dataset.
    """
    labels = [patient.get("label") for patient in data]
    return labels

current_path = os.getcwd()
#print(current_path)
filename = ["train", "test"]
#print(filename[0])
filename_train = "/" + filename[0] + "/" + filename[0] +"_20250408_144607.json"
train = load_data(current_path, filename_train)
print("Train data loaded")
print(len(train))
filename_test = "/" + filename[1] + "/" + filename[1] +"_20250408_144607.json"
test = load_data(current_path, filename_test)
print("Test data loaded")
print(len(test))

Vocab = get_vocab(train)
print("Total words in vocab are",len(Vocab))
print(type(Vocab))
print("Vocabulary:", Vocab)

print(train[0]["seq"])

tensor_result = seq_to_tensor(train[0]["seq"], Vocab, verbose=False)
print("seq_to_tensor:", tensor_result)
print(type(tensor_result))

X_train = data_to_tensor(train, Vocab, verbose=False)
print("data_to_tensor:", X_train[0])
print("data_to_tensor:", X_train[1])
print("data_to_tensor:", X_train[2])
print(type(X_train))
print(X_train.shape)
X_train = np.array(X_train)
print(X_train.shape)
print(X_train[0].shape)

X_test = data_to_tensor(test, Vocab, verbose=False)
print("data_to_tensor:", X_test[0])
print("data_to_tensor:", X_test[1])
print("data_to_tensor:", X_test[2])
print(type(X_test))


print(train[0]["label"])
print(train[1]["label"])

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

now, name, acc, loss, model, num_classes = utils_train_models.train_lstm_model(X_train, y_train, X_test, y_test, hyper_paramts_lstm, path_model_save)