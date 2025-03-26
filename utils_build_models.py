#!/usr/bin/env python3
import tensorflow as tf
import keras
from keras import layers
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB

def _get_last_layer_units_and_activation(num_classes):
    """Gets the # units and activation function for the last network layer.

    # Arguments
        num_classes: int, number of classes.

    # Returns
        units, activation values.
    """
    if num_classes == 2:
        activation = 'sigmoid'
        units = 1
    else:
        activation = 'softmax'
        units = num_classes
    return units, activation


def mlp_model(block_layers, units, dropout_rate, input_shape, num_classes, learning_rate):
    """Creates an instance of a multi-layer perceptron model.

    # Arguments
        block_layers: int, number of `Dense` layers in the model.
        units: int, output dimension of the layers.
        dropout_rate: float, percentage of input to drop at Dropout layers.
        input_shape: tuple, shape of input to the model.
        num_classes: int, number of output classes.

    # Returns
        An MLP model instance.
    """
    op_units, op_activation = _get_last_layer_units_and_activation(num_classes)
    model = tf.keras.Sequential()
    model.add(keras.Input(shape=input_shape))
    model.add(layers.Dense(units=units, activation='relu'))
    model.add(layers.Dropout(rate=dropout_rate))

    # Add one or more dense layers with a dropout.
    for _ in range(block_layers-1):
        model.add(layers.Dense(units=units, activation='relu'))
        model.add(layers.Dropout(rate=dropout_rate))

    model.add(layers.Dense(units=op_units, activation=op_activation))

    # Compile model with learning parameters.
    if num_classes == 2:
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                     loss=keras.losses.BinaryCrossentropy(),
                     metrics=['accuracy'])
    else:
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                     loss=keras.losses.SparseCategoricalCrossentropy(),
                     metrics=['accuracy'])
            
    return model

def lstm_model(num_classes, num_features, embedding_dim, input_shape, block_layers, units, learning_rate):
    """Creates an instance of a LSTM network model.

    # Arguments
        num_classes: int, number of output classes.
        num_features: int, number of input features.
        embedding_dim: int, number of dimensions for word embedding.
        input_shape: tuple, shape of input to the model.
        block_layers: int, number of LSTM blocks.
        units: int, output dimension of the layers.

    # Returns
        A LSTM network model instance.
        
    """
    op_units, op_activation = _get_last_layer_units_and_activation(num_classes)
    model = tf.keras.Sequential()
    model.add(layers.Embedding(input_dim=num_features, output_dim=embedding_dim, input_length=input_shape[0]))
    model.add(layers.Bidirectional(layers.LSTM(units, activation='relu')))
    for _ in range(block_layers-1):
        model.add(layers.Bidirectional(layers.LSTM(units, activation='relu')))    

    model.add(layers.Dense(units, activation = 'relu'))
    model.add(layers.Dense(units=op_units, activation=op_activation))

    # Compile model with learning parameters.
    if num_classes == 2:
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), 
                 loss=keras.losses.BinaryCrossentropy(), 
                 metrics=["accuracy"])
    else:
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), 
                 loss=keras.losses.SparseCategoricalCrossentropy(), 
                 metrics=["accuracy"])
    
    print("LSTM model created")
    return model

def logistic_regression(c_parameter, num_max_iteration):
    """Creates an instance of a logistic regression model.

    # Arguments
        c_parameter: int, inverse of regularization strength.
        num_max_iteration: int, number of iterations to converge.

    # Returns
        A logistic regression model instance.
    """
    model = LogisticRegression(C=c_parameter, max_iter=num_max_iteration)
    return model

def naive_bayes(alpha):
    """Creates an instance of a Naive Bayes model.

    # Arguments
        alpha: float, smoothing parameter.

    # Returns
        A Naive Bayes model instance.
    """
    model = ComplementNB(alpha=alpha)
    return model