#!/usr/bin/env python3
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
import os
import utils_general_porpose
import joblib
import utils_performance_analysis
import utils_explore_data
import datetime
import sys


def load_data(path, filename):
    """
    Load the data from a JSON file.
    """
    data = utils_general_porpose.load_json(path, filename)
    print("Data loaded")
    return data

def get_labels(train, test):
    """
    Extract labels from the data.
    """
    y_train = [patient.get("label") for patient in train]
    y_train = np.array(y_train)
    y_train = y_train.astype(int)

    y_test = [patient.get("label") for patient in test]
    y_test = np.array(y_test)
    y_test = y_test.astype(int)

    return y_train, y_test

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

def save_tokens(current_path, timestamp, X_train_enc, X_test_enc, y_train, y_test, encoder):
    """
    Save the tokens in a file.
    """
    path_save = os.path.join(current_path, "tokens")
    path_tokens = os.path.join(path_save, timestamp)
    path_token_save = utils_general_porpose.create_directory(path_tokens)

    np.save(path_token_save + "/" + "X_train_one", X_train_enc)
    np.save(path_token_save + "/" + "X_test_one", X_test_enc)
    np.save(path_token_save + "/" + "y_train_one", y_train)
    np.save(path_token_save + "/" + "y_test_one", y_test)

    joblib.dump(encoder, path_token_save + "/" + "encoder_one.pkl")
        
    return path_token_save

def one_hot_encode_sequences(X_train, X_test):
    # Concatenar para ajustar el encoder a todos los posibles tokens
    #all_sequences = np.concatenate((X_train, X_test), axis=0)
    #encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    # Suponemos que cada muestra es una secuencia de enteros (tokens)
    # Convertimos cada secuencia en una cadena para que OneHotEncoder trate cada token como una categoría
    #X_all = np.array([' '.join(map(str, seq)) for seq in all_sequences]).reshape(-1, 1)
    #X_train_enc = encoder.fit_transform(np.array([' '.join(map(str, seq)) for seq in X_train]).reshape(-1, 1))
    #X_train_enc = encoder.fit_transform(X_train).reshape(-1, 1)
    #X_train_enc = encoder.transform(np.array([' '.join(map(str, seq)) for seq in X_train]).reshape(-1, 1))
    #X_test_enc = encoder.transform(np.array([' '.join(map(str, seq)) for seq in X_test]).reshape(-1, 1))
    #X_test_enc = encoder.transform(X_test).reshape(-1, 1)

    #join tokens in a string
    #X_train_str = [' '.join(seq) for seq in X_train]
    #X_test_str = [' '.join(seq) for seq in X_test]

    vectorizer = CountVectorizer(binary=True)

    #Fit the vectorizer on the training data
    X_train_enc = vectorizer.fit_transform(X_train).toarray()

    #Transform the test data
    X_test_enc = vectorizer.transform(X_test).toarray()


    return X_train_enc, X_test_enc, vectorizer

def train_logistic_regression(X_train_enc, y_train, penaltize_, max_iter_):
    clf = LogisticRegression(penalty = penaltize_, max_iter = max_iter_)
    clf.fit(X_train_enc, y_train)
    return clf

def save_model(model, current_path):
    # Crear el directorio de modelos si no existe
    path_save = os.path.join(current_path, "models")
    path_model_save = utils_general_porpose.create_directory(path_save)
    print("Model directory created at: {}".format(path_model_save))

    #Get the list of models in the directory
    list_models = utils_general_porpose.extract_name_model(path_model_save, ".pkl")
    print("List of models: {}".format(list_models))

    #Get the last version of the model
    last_version = utils_general_porpose.extract_last_version_model(list_models)
    print("Last version of the model: {}".format(last_version))

    #Get the number of the new version
    version = str(utils_general_porpose.counter_version(last_version))
    print("New version of the model: {}".format(version))

    # Archivo de versiones
    model_name = "logistic_v" + version + ".pkl"
    joblib.dump(model, path_model_save + "/" + model_name)
    print("Model saved as: {}".format(model_name))
    
    return model_name

if __name__ == "__main__":
    if len(sys.argv) !=8:
        print("faltan hyperparametros")
        sys.exit(1)
        
    timestamp = sys.argv[1]
    current_path = sys.argv[2]
    max_len = int(sys.argv[3])  # Maximum length of sequences, not used in this script but can be useful for future modifications    
    semantic_cat = sys.argv[4].split(",")
    dic_local = sys.argv[5]
    days_pw = int(sys.argv[6])  # Days predictive window
    days_ow = int(sys.argv[7])  # Days observational window

    semantic_cat.append(dic_local)
    print("semantic categories {}".format(semantic_cat))

    #current_path = os.getcwd()
    #print("current_path {}".format(current_path))
    #timestamp = "20250520_053753"  # Example timestamp, replace with actual value
    #semantic_cat = ["icd_proof", "disease_proof"]

    d_filename = ["/train", "/test"]
    filename_train = d_filename[0] + "/" + d_filename[0] + "_" + timestamp + ".json"
    #print("path_train {}".format(filename_train))
    train = load_data(current_path, filename_train)
    #print("size train {}".format(len(train)))
    #print("train sample: {}".format(train[:5]))
    filename_test = d_filename[1] + "/" + d_filename[1] +"_" + timestamp + ".json"
    test = load_data(current_path, filename_test)
    #print("size test {}".format(len(test)))

    y_train, y_test = get_labels(train, test)
    #print("y_train shape: {}, y_test shape: {}".format(y_train.shape, y_test.shape))

    train_string = get_data_to_tensor_string(train)
    test_string = get_data_to_tensor_string(test)
    print("train_string sample: {}".format(train_string[:2]))
    print("train_string type: {}".format(type(train_string)))

    # convertir los datos a one-hot encoding
    X_train_enc, X_test_enc, encoder = one_hot_encode_sequences(train_string, test_string)
    print("X_train_enc shape: {}, X_test_enc shape: {}".format(X_train_enc.shape, X_test_enc.shape))
    #print("Number of features after encoding: {}".format(X_train_enc.shape[1]))
    print("X_train_enc sample: {}".format(X_train_enc[:1]))
    print("X_train_type: {}".format(type(X_train_enc)))
    #print("encoder categories: {}".format(encoder.categories_))
    #print("encoder feature names: {}".format(encoder.get_feature_names_out()))
    print("encoder type: {}".format(type(encoder)))

    # Guardar los tokens
    path_token_save = save_tokens(current_path, timestamp, X_train_enc, X_test_enc, y_train, y_test, encoder)
    #print("Tokens saved in: {}".format(path_token_save))

    # Entrenar el modelo de regresión logística
    max_iter_ = 1000  # Puedes ajustar este valor según tus necesidades
    penaltize_ = 'l2'  # Regularization type, can be 'l1', 'l2', or 'elasticnet'
    print("Training Logistic Regression model...")
    model = train_logistic_regression(X_train_enc, y_train, penaltize_, max_iter_)
    print("Logistic Model trained successfully.")

    num_classes = utils_explore_data.get_num_classes(y_train)

    #Make predictions in training
    y_pred_train = utils_performance_analysis.predict_values_logistic_regression(model, X_train_enc)

    #make predictions in test
    y_pred_test = utils_performance_analysis.predict_values_logistic_regression(model, X_test_enc)

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

    #calculate accuracy
    acc = utils_performance_analysis.accuracy(y_test, model.predict(X_test_enc))
    loss = "N/A"  # Logistic regression does not have a loss function like neural networks

    # Save the model
    model_name = save_model(model, current_path)
    path_pr_save = current_path + "/" + "performance_report.csv"
    #chage the format of the timestamp
    timestamp_s = datetime.datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    utils_performance_analysis.save_model_info(timestamp_s, str(semantic_cat), num_classes, "one-hot", {"max_len":max_len,"n_feature":X_train_enc.shape[1]}, path_token_save, model_name, {"penalties":penaltize_, "max_iter":max_iter_}, acc, loss, precision_train, recall_train, f1_train, precision_test, recall_test, f1_test, path_pr_save, days_pw, days_ow)

