#!/usr/bin/env python3
import os
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import utils_general_porpose
import utils_build_models
import utils_explore_data
import numpy as np
import sys
import utils_performance_analysis
import datetime
import pickle

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
        #output_mode='int',
        output_sequence_length=max_len,
        dtype="int32")
    
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

def get_vocab_size(encoder):
    """
    Get the size of the vocabulary.
    """
    vocab = np.array(encoder.get_vocabulary())    
    vocab_size = len(vocab)
    global vocab_size_g
    vocab_size_g = vocab_size
    return vocab

def save_tokens(current_path, timestamp, vocab):
    """
    Save the tokens in a file.
    """
    path_save = os.path.join(current_path, "tokens")
    path_tokens = os.path.join(path_save, timestamp)
    path_token_save = utils_general_porpose.create_directory(path_tokens)

    np.save(path_token_save + "/" + "X_train", X_train_g)
    np.save(path_token_save + "/" + "X_test", X_test_g)
    np.save(path_token_save + "/" + "y_train", y_train_g)
    np.save(path_token_save + "/" + "y_test", y_test_g)
    np.save(path_token_save + "/" + "vocab", vocab)
    
    return path_token_save

#Function to save a pickle object in a file
def save_pickle_file(obj, path):
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path

def load_hyperparameters(path_h):
    """
    Load the hyperparameters from a JSON file.
    """    
    hyper_paramts_lstm = utils_general_porpose.load_hyperparams_as_tuples(path_h)
    print("Hyperparameters loaded")
    print(type(hyper_paramts_lstm))
    return hyper_paramts_lstm
    
#Function to train a lstm model    
def train_lstm_model(h_dictionary):

    #Get the hyperparameters from dictionary
    num_features = vocab_size_g    
    embedding_dim = h_dictionary["embedding_dim"]
    block_layers = h_dictionary["block_layers"]
    units = h_dictionary["hidden_units"]
    learning_rate = h_dictionary["learning_rate"]
    epochs = h_dictionary["epochs"]
    batch_size = h_dictionary["batch_size"]    
    
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

def save_model(model, current_path):
    # Crear el directorio de modelos si no existe
    path_save = os.path.join(current_path, "models")
    path_model_save = utils_general_porpose.create_directory(path_save)

    #Get the list of models in the directory
    list_models = utils_general_porpose.extract_name_model(path_model_save, ".h5")

    #Get the last version of the model
    last_version = utils_general_porpose.extract_last_version_model(list_models)

    #Get the number of the new version
    version = str(utils_general_porpose.counter_version(last_version))

    # Archivo de versiones
    model_name = "lstm_v" + version + ".h5"
    model.save(path_save + "/" + model_name)
    
    return model_name
    #path_save = current_path + "/" + "performance_report.csv"
    #utils_performance_analysis.save_model_info(timestamp, dataset_name, num_classes, vectorize_technique, vectorization_hyperparameters, path_vectorization, model_name, model_hyperparameters, acc, loss, precision_train, recall_train, f1_train, precision_test, recall_test, f1_test, path_save)

    
if __name__ == "__main__":
    if len(sys.argv) !=6:
        print("faltan hyperparametros")
        sys.exit(1)
        
    timestamp = sys.argv[1]
    current_path = sys.argv[2]
    max_tokens = int(sys.argv[3])
    max_len = int(sys.argv[4])
    semantic_cat = sys.argv[5]

    #timestamp = "20250408_144607"
    #current_path = "/home/pajaro/compu_Pipe_V3/"
    #max_tokens = 5000
    #max_len = 4

    print("Starting LSTM pipeline...")

    d_filename = ["train", "test"]
    filename_train = d_filename[0] + "/" + d_filename[0] + "_" + timestamp + ".json"
    print("path_train {}".format(filename_train))
    train = load_data(current_path, filename_train)
    train_string = get_data_to_tensor_string(train)    
    encoder = get_vectorized_layer(train_string, max_tokens, max_len)
    #encoder.adapt(train_string)
        
    #vocab = np.array(encoder.get_vocabulary())    
    #print("Vocabulary size:", len(vocab))
    vocab = get_vocab_size(encoder)
    print("Vocabulary size:", vocab_size_g)
    get_X_train(encoder, train_string)    
    filename_test = d_filename[1] + "/" + d_filename[1] +"_" + timestamp + ".json"
    test = load_data(current_path, filename_test)
    #print(len(test))
    test_s = get_data_to_tensor_string(test)    
    get_X_test(encoder, test_s)
    get_labels(train, test)
    path_token_save = save_tokens(current_path, timestamp, vocab)
    save_pickle_file(encoder, path_token_save + "/" + "tokenizer_obj.pkl")

    #load lstm hyperparameters
    filename_h = "models_parameters/list_hyper_params_lstm.json"
    list_seq_params = utils_general_porpose.load_json(current_path, filename_h)
    print(list_seq_params[0])
    print("list seq params loaded")
    print(list_seq_params[0]["embedding_dim"])
    dict_p = list_seq_params[0]
    print(dict_p["embedding_dim"])
    
    # Run parallel extraction
    print("Running lstm loop...")
    for i in range(len(list_seq_params)):
        print(list_seq_params[i])        
        acc, loss, model, num_classes = train_lstm_model(list_seq_params[i])        
        print("acc {}".format(acc))
        print("loss {}".format(loss))
        #Make predictions in training
        y_pred_train = utils_performance_analysis.predict_values(model, X_train_g, num_classes)

        #make predictions in test
        y_pred_test = utils_performance_analysis.predict_values(model, X_test_g, num_classes)

        #Get the metrics train
        precision_train, recall_train, f1_train = utils_performance_analysis.get_model_metrics(y_train_g, y_pred_train, num_classes)
        print("precision_train: {}".format(precision_train))
        print("recall_train: {}".format(recall_train))
        print("f1_train: {}".format(f1_train))

        #Get the metrics test
        precision_test, recall_test, f1_test = utils_performance_analysis.get_model_metrics(y_test_g, y_pred_test, num_classes)
        print("precision_test: {}".format(precision_test))
        print("recall_test: {}".format(recall_test))
        print("f1_test: {}".format(f1_test))
        # Save the model
        model_name = save_model(model, current_path)
        path_pr_save = current_path + "/" + "performance_report.csv"
        #chage the format of the timestamp
        timestamp_s = datetime.datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
        utils_performance_analysis.save_model_info(timestamp_s, semantic_cat, num_classes, "TextVectorize layer", {"max_tokens":max_tokens, "max_len":max_len, "vectorize_technique":"other-sequence"}, path_token_save, model_name, list_seq_params[i], acc, loss, precision_train, recall_train, f1_train, precision_test, recall_test, f1_test, path_pr_save)
    
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
    print("LSTM loop finished")