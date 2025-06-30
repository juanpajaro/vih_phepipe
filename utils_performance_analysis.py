#!/usr/bin/env python3
import csv
import os
import pandas as pd
import tensorflow as tf
import keras
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from scipy import sparse

#function to save the information of the model in list of dictionaries
def save_model_info(now, dataset_name, num_classes, vectorize_technique, vectorization_hyperparameters, path_vectorization, model_name, model_hyperparameters, acc, loss, precision_train, recall_train, f1_train, precision_test, recall_test, f1_test, path_save, days_pw, days_ow):

    #check if the file exists
    if os.path.isfile(path_save):
        #if the file exists, then append the new information
        with open(path_save, mode='a') as file:
            writer = csv.writer(file)
            writer.writerow([now, dataset_name, num_classes, vectorize_technique, vectorization_hyperparameters, path_vectorization, model_name, model_hyperparameters, acc, loss, precision_train, recall_train, f1_train, precision_test, recall_test, f1_test, days_pw, days_ow])
    else:
        #if the file does not exist, then create the file and add the header
        with open(path_save, mode='w') as file:
            writer = csv.writer(file)
            writer.writerow(["date", "semantic_categories", "num_classes", "vectorize_technique", "vectorization_hyperparameters", "path_vectorization", "model_name", "model_hyperparameters", "accuracy", "loss", "precision_train", "recall_train", "f1_train", "precision_test", "recall_test", "f1_test", "days_pw", "days_ow"])
            writer.writerow([now, dataset_name, num_classes, vectorize_technique, vectorization_hyperparameters, path_vectorization, model_name, model_hyperparameters, acc, loss, precision_train, recall_train, f1_train, precision_test, recall_test, f1_test, days_pw, days_ow])

def read_analysis(path):
    #read the file
    document = pd.read_csv(path)
    return document

#function to get a list of the models
def get_models_info(path_performance_document):
    #read the file
    with open(path_performance_document, mode='r') as file:
        reader = csv.DictReader(file)
        list_models_info = list(reader)
        file.close()
    return list_models_info

#Function to get the best accuracy model in a list
def get_best_model(list_models_info, metric_compare):
    best_acc = 0
    for model_info in list_models_info:
        if float(model_info[metric_compare]) > best_acc:
            best_acc = float(model_info[metric_compare])
            best_model_info = model_info
    return best_model_info

#function to predict values of neural networks models
def predict_values(model, x_val, num_classes):
    y_pred = model.predict(x_val)
    if num_classes == 2:
        y_pred = np.where(y_pred > 0.5, 1, 0)
    else:
        y_pred = np.argmax(y_pred, axis=1)    
    return y_pred

#function to predict values of logistic regression models
def predict_values_logistic_regression(model, x_val):
    y_pred = model.predict(x_val)
    return y_pred

#function to get the precision, recall, f1 and accuracy of the model
def get_model_metrics(y_true, y_pred, num_classes):
    """
    Function to get the precision, recall, f1 and accuracy of the model
    # Arguments
        y_true: int, true values
        y_pred: int, predicted values
    
    # Returns
        precision, recall, f1, accuracy
    """
    if num_classes == 2:
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')
    else:
        precision = precision_score(y_true, y_pred, average=None)
        recall = recall_score(y_true, y_pred, average=None)
        f1 = f1_score(y_true, y_pred, average=None)

        #precision = np.array2string(precision, precision=3, separator=',')
        #recall = np.array2string(recall, precision=3, separator=',')
        #f1 = np.array2string(f1, precision=3, separator=',')

        precision = np.mean(precision)
        recall = np.mean(recall)
        f1 = np.mean(f1)
    #accuracy = accuracy_score(y_true, y_pred)
        
    precision = round(precision, 3)
    recall = round(recall, 3)
    f1 = round(f1, 3)
        
    return precision, recall, f1

#Get the test set
def get_test_set(path):
    #Get the test set
    X_test = sparse.load_npz(path + '/vectorized_data/X_test_vectors.npz')
    y_test = np.load(path + '/vectorized_data/y_test.npy')    
    return X_test, y_test

#Load model
def load_model(path, best_model_info):
    model = tf.keras.models.load_model(path + "/models/" + best_model_info["model_name"])
    return model

def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)