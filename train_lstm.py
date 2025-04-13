#!/usr/bin/env python3
import os
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import utils_general_porpose
import utils_build_models
import utils_explore_data
import numpy as np
import sys
#import utils_train_models

X_train_g = None
y_train_g = None
X_test_g = None
y_test_g = None
max_len_seq = None

def load_data(path, filename):
    """
    Load the data from a JSON file.
    """
    data = utils_general_porpose.load_json(path, filename)
    print("Data loaded")
    return data

def get_labels(train, test):
    """
    Get the labels from the dataset.
    """
    y_train = [patient.get("label") for patient in train]
    y_train = np.array(y_train)
    y_train = y_train.astype(int)
    global y_train_g
    y_train_g = y_train

    y_test = [patient.get("label") for patient in test]
    y_test = np.array(y_test)
    y_test = y_test.astype(int)    
    global y_test_g
    y_test_g = y_test
    

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
    global max_len_seq
    max_len_seq = max_len
    vectorize_layer = TextVectorization(
        max_tokens=max_tokens,
        standardize=None,
        output_mode='int',
        output_sequence_length=max_len)
    vectorize_layer.adapt(X_train)
    return vectorize_layer

def get_X_train(vectorize_layer, x_train_s):
    X_train = vectorize_layer(x_train_s)
    global X_train_g
    X_train_g = X_train

def get_X_test(vectorize_layer, x_test_s):
    X_test = vectorize_layer(x_test_s)
    global X_test_g
    X_test_g = X_test

#Function to train a lstm model    
def train_lstm_model(hyper_paramts_lstm):
    """Trains LSTM model on the given dataset.

    # Arguments
        data: tuples of training and test texts and labels.
        num_features: int, number of input features.
        embedding_dim: int, dimension of the embedding vectors.
        block_layers: int, number of `Dense` layers in the model.
        units: int, output dimension of Dense layers in the model.
        learning_rate: float, learning rate for training model.
        epochs: int, number of epochs.
        batch_size: int, number of samples per batch.

    # Raises
        ValueError: If validation data has label values which were not seen
            in the training data.
    """

    #Get the hyperparameters from dictionary
    num_features = hyper_paramts_lstm['num_features']
    embedding_dim = hyper_paramts_lstm['embedding_dim']
    block_layers = hyper_paramts_lstm['block_layers']
    units = hyper_paramts_lstm['hidden_units']
    learning_rate = hyper_paramts_lstm['learning_rate']
    epochs = hyper_paramts_lstm['epochs']
    batch_size = hyper_paramts_lstm['batch_size']

    
    # Verify that validation labels are in the same range as training labels.
    
    num_classes = utils_explore_data.get_num_classes(y_train_g)
    
    unexpected_labels = [v for v in y_test_g if v not in range(num_classes)]
    if len(unexpected_labels):
        raise ValueError('Unexpected label values found in the validation set:'
                         ' {unexpected_labels}. Please make sure that the '
                         'labels in the validation set are in the same range '
                         'as training labels.'.format(
                             unexpected_labels=unexpected_labels))
    
    #num_features = min(len(tokenizer) + 1, TOP_K)
    
    # Create model instance.
    model = utils_build_models.lstm_model(num_classes=num_classes,
                                   num_features=num_features,
                                   embedding_dim=embedding_dim,
                                   input_shape=X_train_g.shape[1:],
                                   block_layers=block_layers,
                                   units=units,
                                   learning_rate=learning_rate)
    
    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5)]
    
    print()
    model.summary()
    print("lstm model created")
    print()
    
    # Train and validate model.
    history = model.fit(
        X_train_g,
        y_train_g,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=(X_test_g, y_test_g),
        verbose=2,  # Logs once per epoch.
        batch_size=batch_size)
    
    # Print results.
    history = history.history
    #print("the performances of the lstm model are:")
    #print('Validation accuracy: {acc}, loss: {loss}'.format(acc=history['val_accuracy'][-1], loss=history['val_loss'][-1]))
    
    #Get the date and time
    #now = utils_general_porpose.get_time()

    #Get the list of models in the directory
    #list_models = utils_general_porpose.extract_name_model(path_model_save, ".h5")

    #Get the last version of the model
    #last_version = utils_general_porpose.extract_last_version_model(list_models)

    #Get the number of the new version
    #version = str(utils_general_porpose.counter_version(last_version))

    # Save model.
    #name = 'lstm_model_v' + version +'.h5'
    #model.save(path_model_save + name)
    return history['val_accuracy'][-1], history['val_loss'][-1], model, num_classes

def save_model(model, current_path, timestamp):
#save the dataset
    path_save = current_path + "/models/"
    path_save = utils_general_porpose.create_directory(path_save)

    if os.path.exists(path_save):
        #path_version = path_save + "model_" + timestamp + ".pkl"
        path_version = path_save + "model_" + timestamp
        #data.to_csv(path_version, index = False)
        model.save(path_version)

if __name__ == "main":
    if len(sys.argv) !=3:
        print("faltan hyperparametros")
        sys.exit(1)
        
    #timestamp = sys.argv[2]
    timestamp = "2023-10-01_12-00-00"
    #current_path = sys.argv[3]
    current_path = "/home/pajaro/compu_Pipe_V3/"
    d_filename = ["train", "test"]
    filename_train = "/" + d_filename[0] + "/" + d_filename[0] + "_" + "20250408_144607" + ".json"
    train = load_data(current_path, filename_train)
    train_string = get_data_to_tensor_string(train)
    print("data_to_tensor_string:", train_string[:2])
    encoder = get_vectorized_layer(train_string, max_tokens=5000, max_len=4)
    vocab = np.array(encoder.get_vocabulary())
    print("Vocabulary:", vocab[:100])
    print("Vocabulary size:", len(vocab))
    print(type(encoder))
    get_X_train(encoder, train_string)
    print(X_train_g[:2])
    filename_test = "/" + d_filename[1] + "/" + d_filename[1] +"_" + "20250408_144607" + ".json"
    test = load_data(current_path, filename_test)
    print(len(test))
    test_s = get_data_to_tensor_string(test)
    print("data_to_tensor_string:", test_s[:2])
    get_X_test(encoder, test_s)
    print(X_test_g[:2])
    hyper_paramts_lstm = utils_general_porpose.load_json(current_path, "/models_parameters/hyper_params_lstm.json")
    print("Hyperparameters loaded")
    print(hyper_paramts_lstm)
    print(type(hyper_paramts_lstm))
    get_labels(train, test)
    print("y_train:", y_train_g[0])
    print("y_train:", y_train_g[1])
    print("y_test:", y_test_g[0])
    print("y_test:", y_test_g[1])
    acc, loss, model, num_classes = train_lstm_model(hyper_paramts_lstm)
    save_model(model, current_path, timestamp)