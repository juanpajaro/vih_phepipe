#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Attention, Concatenate, Dropout, Bidirectional, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import utils_general_porpose
import utils_explore_data
import utils_performance_analysis
import datetime
import sys

def load_data(data_dir):
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    vocab = np.load(os.path.join(data_dir, "vocab.npy"))
    return X_train, X_test, y_train, y_test, vocab

def build_lstm_attention_model(input_shape, units=64, dropout=0.2, bidirectional=True, vocab_size=None, embedding_dim=128):
    """
    Construye un modelo LSTM con atención y una capa Embedding.

    Parámetros:
    - input_shape: tupla (max_len,) para secuencias de enteros.
    - units: número de unidades LSTM.
    - dropout: tasa de dropout.
    - bidirectional: si True, usa LSTM bidireccional.
    - vocab_size: tamaño del vocabulario para la capa Embedding.
    - embedding_dim: dimensión de salida de la capa Embedding.
    """
    input_layer = Input(shape=input_shape)
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_shape[0])(input_layer)
    if bidirectional:
        lstm_out = Bidirectional(LSTM(units, return_sequences=True, dropout=dropout))(x)
    else:
        lstm_out = LSTM(units, return_sequences=True, dropout=dropout)(x)
    attention = Attention()([lstm_out, lstm_out])
    concat = Concatenate()([lstm_out, attention])
    flat = tf.keras.layers.GlobalAveragePooling1D()(concat)
    drop = Dropout(dropout)(flat)
    output = Dense(1, activation='sigmoid')(drop)
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=30, batch_size=32, patience=5):

    num_classes = utils_explore_data.get_num_classes(y_train)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)        
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

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
    model_name = "attention_v" + version + ".h5"
    model.save(path_save + "/" + model_name)
    
    return model_name

if __name__ == "__main__":
    if len(sys.argv) !=6:
        print("faltan hyperparametros")
        sys.exit(1)
        
    timestamp = sys.argv[1]
    current_path = sys.argv[2]
    max_tokens = int(sys.argv[3])    
    semantic_cat = sys.argv[4].split(",")
    dic_local = sys.argv[5]

    semantic_cat.append(dic_local)
    print("semantic categories {}".format(semantic_cat))

    #current_path = os.getcwd()
    #timestamp = "20250520_053753"  # Example timestamp, replace with actual value

    print("Cargando datos...")
    X_train, X_test, y_train, y_test, vocab = load_data(current_path + "/tokens/" + timestamp) 
    print("Datos cargados. Tamaño de X_train: {}, X_test: {}, y_train: {}, y_test: {}".format(
        X_train.shape, X_test.shape, y_train.shape, y_test.shape))
    
    #max_tokens = 1000
    max_len = X_train.shape[1]  # Assuming X_train is a 2D array with shape (num_samples, max_len)
    path_token_save = current_path + "/tokens/" + timestamp
    #semantic_cat = ["icd_proof", "disease_proof"]
    
    #vocab_size
    vocab_size_ = len(vocab)

    #load lstm hyperparameters
    filename_h = "/models_parameters/list_hyper_params_attention.json"
    list_seq_params = utils_general_porpose.load_json(current_path, filename_h)
    print(len(list_seq_params))

    
    print("Running attention loop...")
    for i in range(len(list_seq_params)):
        embedding_dim_ = list_seq_params[i]["embedding_dim"]
        units = list_seq_params[i]["units"]
        dropout = list_seq_params[i]["dropout"]        
        epochs = list_seq_params[i]["epochs"]
        batch_size = list_seq_params[i]["batch_size"]        

        print(f"Configuración {i+1}:")
        print(f"  embedding_dim: {embedding_dim_}")
        print(f"  units: {units}")
        print(f"  dropout: {dropout}")        
        print(f"  epochs: {epochs}")
        print(f"  batch_size: {batch_size}")

        

        # Build and train the model
        model = build_lstm_attention_model(
            input_shape=(X_train.shape[1],),
            units=units,
            dropout=dropout,
            vocab_size=vocab_size_,
            embedding_dim=embedding_dim_            
        )
        model.summary()
        print("Entrenando...")
        acc, loss, model, num_classes = train_model(
            model, X_train, y_train, X_test, y_test,        
            epochs=epochs,
            batch_size=batch_size,
            
        )
        print("acc {}".format(acc))
        print("loss {}".format(loss))
        print("Entrenamiento completado.")
        #Make predictions in training
        y_pred_train = utils_performance_analysis.predict_values(model, X_train, num_classes)

        #make predictions in test
        y_pred_test = utils_performance_analysis.predict_values(model, X_test, num_classes)

        #Get the metrics train
        precision_train, recall_train, f1_train = utils_performance_analysis.get_model_metrics(y_train, y_pred_train, num_classes)
        print("precision_train: {}".format(precision_train))
        print("recall_train: {}".format(recall_train))
        print("f1_train: {}".format(f1_train))

        #Get the metrics test
        precision_test, recall_test, f1_test = utils_performance_analysis.get_model_metrics(y_test, y_pred_test, num_classes)
        print("precision_test: {}".format(precision_test))
        print("recall_test: {}".format(recall_test))
        print("f1_test: {}".format(f1_test))
        # Save the model
        model_name = save_model(model, current_path)
        path_pr_save = current_path + "/" + "performance_report.csv"
        #chage the format of the timestamp
        timestamp_s = datetime.datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
        utils_performance_analysis.save_model_info(timestamp_s, str(semantic_cat), num_classes, "keras-vectorizer", {"max_tokens":max_tokens, "max_len":max_len, "vectorize_technique":"other-sequence"}, path_token_save, model_name, list_seq_params[i], acc, loss, precision_train, recall_train, f1_train, precision_test, recall_test, f1_test, path_pr_save)
        
        


    