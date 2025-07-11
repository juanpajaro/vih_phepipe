#!/usr/bin/env python3
import os
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import utils_general_porpose
import utils_build_models
import utils_explore_data
import numpy as np
import sys
import multiprocessing
#import utils_train_models

X_train_g = None
y_train_g = None
X_test_g = None
y_test_g = None
max_len_seq = None
vocab_size_g = None

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

def get_vocab_size(vectorize_layer):
    """
    Get the size of the vocabulary.
    """
    vocab = np.array(encoder.get_vocabulary())
    vocab_size = len(vocab)
    global vocab_size_g
    vocab_size_g = vocab_size

def load_hyperparameters(path_h):
    """
    Load the hyperparameters from a JSON file.
    """    
    hyper_paramts_lstm = utils_general_porpose.load_hyperparams_as_tuples(path_h)
    print("Hyperparameters loaded")
    print(type(hyper_paramts_lstm))
    return hyper_paramts_lstm
    
#Function to train a lstm model    
def train_lstm_model(embedding_dim, 
     block_layers,
     units,
     learning_rate,
     epochs,
     batch_size):

    #Get the hyperparameters from dictionary
    num_features = vocab_size_g
    """
    (embedding_dim, 
     block_layers,
     units,
     learning_rate,
     epochs,
     batch_size
    ) = hyper_paramts_lstm
    """
    
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
    
    return history['val_accuracy'][-1], history['val_loss'][-1], model, num_classes

def save_model(model, current_path, timestamp, index):
#save the dataset
    path_save = current_path + "/models/"
    path_save = utils_general_porpose.create_directory(path_save)

    if os.path.exists(path_save):
        #path_version = path_save + "model_" + timestamp + ".pkl"
        path_version = path_save + "model_" + timestamp + "_" + str(index)
        #data.to_csv(path_version, index = False)
        model.save(path_version)

if __name__ == "__main__":
    if len(sys.argv) !=5:
        print("faltan hyperparametros")
        sys.exit(1)
        
    timestamp = sys.argv[1]
    current_path = sys.argv[2]
    max_tokens = int(sys.argv[3])
    max_len = int(sys.argv[4])

    #timestamp = "20250408_144607"
    #current_path = "/home/pajaro/compu_Pipe_V3/"
    #max_tokens = 5000
    #max_len = 4

    d_filename = ["train", "test"]
    filename_train = d_filename[0] + "/" + d_filename[0] + "_" + timestamp + ".json"
    train = load_data(current_path, filename_train)
    train_string = get_data_to_tensor_string(train)    
    encoder = get_vectorized_layer(train_string, max_tokens, max_len)
    #vocab = np.array(encoder.get_vocabulary())    
    #print("Vocabulary size:", len(vocab))
    get_vocab_size(encoder)
    #print("Vocabulary size:", vocab_size_g)
    get_X_train(encoder, train_string)    
    filename_test = d_filename[1] + "/" + d_filename[1] +"_" + timestamp + ".json"
    test = load_data(current_path, filename_test)
    #print(len(test))
    test_s = get_data_to_tensor_string(test)    
    get_X_test(encoder, test_s)

    get_labels(train, test)

    #n_workers = multiprocessing.cpu_count()
    n_workers = 8
    print(f"Using {n_workers} workers...")

    #load the ngram list of parameters
    #current_path = "/home/pajaro/compu_Pipe_V3/"
    filename_h = "models_parameters/list_hyper_params_lstm.json"
    list_seq_params = load_hyperparameters(current_path + filename_h)
    print(len(list_seq_params))
    print(type(list_seq_params))
    print(list_seq_params[0])
    print("list seq params loaded")
    
    """
    # Split the data into chunks for parallel processing
    chunk_size = len(list_seq_params) // n_workers
    print(f"Chunk size: {chunk_size}")
    chunks = [list_seq_params[i:i + chunk_size] for i in range(0, len(list_seq_params), chunk_size)]

    # If there are any remaining patients, add them to the last chunk
    if len(list_seq_params) % n_workers != 0:
        chunks[-1].extend(list_seq_params[len(chunks) * chunk_size:])
    """

    # Run parallel extraction
    print("Running parallel extraction...")
    with multiprocessing.Pool(processes=n_workers) as pool:
        results = pool.starmap(train_lstm_model, list_seq_params)

    for i, result in enumerate(results, 1):
        acc, loss, model, num_classes = result
        print("index{}".format(i))
        print("acc{}".format(acc))
        print("loss{}".format(loss))
        # Save the model
        save_model(model, current_path, timestamp, i)
    
    """
    hyper_paramts_lstm = load_hyperparameters(current_path + "/models_parameters/hyper_params_lstm.json")
    print("Hyperparameters loaded")
    print(hyper_paramts_lstm)
    print(type(hyper_paramts_lstm))
    
    acc, loss, model, num_classes = train_lstm_model(hyper_paramts_lstm[0])
    print(acc, loss)
    #print("PARAM1={}".format(acc))
    #print("PARAM2={}".format(loss))
    save_model(model, current_path, timestamp, 1)
    """