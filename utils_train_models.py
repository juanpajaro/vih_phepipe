#!/usr/bin/env python3
import utils_build_models as build_models
import utils_explore_data
import tensorflow as tf
import keras
import pickle
import utils_performance_analysis
import utils_general_porpose


def train_mlp_model(X_train_vectors,
                    y_train,
                    X_test_vectors,
                    y_test, 
                    hyper_params_mlp, 
                    path_model_save):
    """Trains n-gram model on the given dataset.

    # Arguments
        data: tuples of training and test texts and labels.
        learning_rate: float, learning rate for training model.
        epochs: int, number of epochs.
        batch_size: int, number of samples per batch.
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of Dense layers in the model.
        dropout_rate: float: percentage of input to drop at Dropout layers.

    # Raises
        ValueError: If validation data has label values which were not seen
            in the training data.
    """

    #Get the hyperparameters from dictionary
    block_layers = hyper_params_mlp['block_layers']
    learning_rate = hyper_params_mlp['learning_rate']
    epochs = hyper_params_mlp['epochs']
    batch_size = hyper_params_mlp['batch_size']
    units = hyper_params_mlp['hidden_units']
    dropout_rate = hyper_params_mlp['dropout']
    
    # Verify that validation labels are in the same range as training labels.
    y_train = y_train.astype(int)
    num_classes = utils_explore_data.get_num_classes(y_train)
    y_test = y_test.astype(int)
    unexpected_labels = [v for v in y_test if v not in range(num_classes)]
    if len(unexpected_labels):
        raise ValueError('Unexpected label values found in the validation set:'
                         ' {unexpected_labels}. Please make sure that the '
                         'labels in the validation set are in the same range '
                         'as training labels.'.format(
                             unexpected_labels=unexpected_labels))
    
    # Create model instance.
    model = build_models.mlp_model(block_layers=block_layers,
                                  units=units,
                                  dropout_rate=dropout_rate,
                                  input_shape=X_train_vectors.shape[1:],
                                  num_classes=num_classes, 
                                  learning_rate=learning_rate)    
    
    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5)]
    
    print()
    model.summary()
    print("MLP model created")
    print()

    # Train and validate model.
    history = model.fit(
        X_train_vectors,
        y_train,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=(X_test_vectors, y_test),
        verbose=2,  # Logs once per epoch.
        batch_size=batch_size)
    
    # Print results.
    history = history.history
    #print("the performances of the n-gram model are:")
    #print('Validation accuracy: {acc}, loss: {loss}'.format(acc=history['val_accuracy'][-1], loss=history['val_loss'][-1]))

    #Get the date and time
    now = utils_general_porpose.get_time()

    #Get the list of models in the directory
    list_models = utils_general_porpose.extract_name_model(path_model_save, ".h5")

    #Get the last version of the model
    last_version = utils_general_porpose.extract_last_version_model(list_models)

    #Get the number of the new version
    version = str(utils_general_porpose.counter_version(last_version))
    
    # Save model.
    name = 'mlp_model_v' + version +'.h5'
    model.save(path_model_save + name)
    return now, name, history['val_accuracy'][-1], history['val_loss'][-1], model, num_classes


#Function to train a lstm model    
def train_lstm_model(X_train,
                    y_train,
                    X_test,
                    y_test,
                    hyper_paramts_lstm, 
                    path_model_save):
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
    y_train = y_train.astype(int)
    num_classes = utils_explore_data.get_num_classes(y_train)
    y_test = y_test.astype(int)
    unexpected_labels = [v for v in y_test if v not in range(num_classes)]
    if len(unexpected_labels):
        raise ValueError('Unexpected label values found in the validation set:'
                         ' {unexpected_labels}. Please make sure that the '
                         'labels in the validation set are in the same range '
                         'as training labels.'.format(
                             unexpected_labels=unexpected_labels))
    
    #num_features = min(len(tokenizer) + 1, TOP_K)
    
    # Create model instance.
    model = build_models.lstm_model(num_classes=num_classes,
                                   num_features=num_features,
                                   embedding_dim=embedding_dim,
                                   input_shape=X_train.shape[1:],
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
        X_train,
        y_train,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=(X_test, y_test),
        verbose=2,  # Logs once per epoch.
        batch_size=batch_size)
    
    # Print results.
    history = history.history
    #print("the performances of the lstm model are:")
    #print('Validation accuracy: {acc}, loss: {loss}'.format(acc=history['val_accuracy'][-1], loss=history['val_loss'][-1]))
    
    #Get the date and time
    now = utils_general_porpose.get_time()

    #Get the list of models in the directory
    list_models = utils_general_porpose.extract_name_model(path_model_save, ".h5")

    #Get the last version of the model
    last_version = utils_general_porpose.extract_last_version_model(list_models)

    #Get the number of the new version
    version = str(utils_general_porpose.counter_version(last_version))

    # Save model.
    name = 'lstm_model_v' + version +'.h5'
    model.save(path_model_save + name)
    return now, name, history['val_accuracy'][-1], history['val_loss'][-1], model, num_classes

#function to train the logistic regression model
def train_logistic_regression(X_train_vectors,
                              y_train,
                    X_test_vectors,
                    y_test, 
                    hyper_params_logistic_regression, 
                    path_model_save):
    """Trains logistic regression model on the given dataset.

    # Arguments
        data: tuples of training and test texts and labels.
        c_parameter: float, inverse of regularization strength.
        num_max_iteration: int, number of maximum iterations taken for the solvers to converge.

    # Raises
        ValueError: If validation data has label values which were not seen
            in the training data.
    """

    #Get the hyperparameters from dictionary
    c_parameter = hyper_params_logistic_regression['c_parameter']
    num_max_iteration = hyper_params_logistic_regression['num_max_iteration']
    
    # Verify that validation labels are in the same range as training labels.
    y_train = y_train.astype(int)
    num_classes = utils_explore_data.get_num_classes(y_train)
    y_test = y_test.astype(int)
    unexpected_labels = [v for v in y_test if v not in range(num_classes)]
    if len(unexpected_labels):
        raise ValueError('Unexpected label values found in the validation set:'
                         ' {unexpected_labels}. Please make sure that the '
                         'labels in the validation set are in the same range '
                         'as training labels.'.format(
                             unexpected_labels=unexpected_labels))
    
    # Create model instance.
    model = build_models.logistic_regression(c_parameter=c_parameter,
                                  num_max_iteration=num_max_iteration)    
    
    # Train and validate model.
    model.fit(X_train_vectors, y_train)
    
    # Print results.
    #print("the performances of the logistic regression model are:")
    #print('Validation accuracy: {acc}, loss: {loss}'.format(acc=history['val_accuracy'][-1], loss=history['val_loss'][-1]))

    #Get the date and time
    now = utils_general_porpose.get_time()

    #Get the list of models in the directory
    list_models = utils_general_porpose.extract_name_model(path_model_save, ".pkl")

    #Get the last version of the model
    last_version = utils_general_porpose.extract_last_version_model(list_models)

    #Get the number of the new version
    version = str(utils_general_porpose.counter_version(last_version))

    #calculate accuracy
    acc = utils_performance_analysis.accuracy(y_test, model.predict(X_test_vectors))

    #calculate loss
    loss = "Logistic regression does not have loss"
    
    # Save model.
    name = 'logistic_regression_model_v' + version +'.pkl'
    pickle.dump(model, open(path_model_save + name, "wb"))
    return now, name, acc, loss, model, num_classes

#function to train navie bayes model
def train_naive_bayes(X_train_vectors,
                      y_train,
                      X_test_vectors,
                      y_test, 
                      hyper_params_naive_bayes, 
                      path_model_save):
    """Trains naive bayes model on the given dataset.

    # Arguments
        data: tuples of training and test texts and labels.
        alpha: float, smoothing parameter.

    # Raises
        ValueError: If validation data has label values which were not seen
            in the training data.
    """

    #Get the hyperparameters from dictionary
    alpha = hyper_params_naive_bayes['alpha']
    
    # Verify that validation labels are in the same range as training labels.
    y_train = y_train.astype(int)
    num_classes = utils_explore_data.get_num_classes(y_train)
    y_test = y_test.astype(int)
    unexpected_labels = [v for v in y_test if v not in range(num_classes)]
    if len(unexpected_labels):
        raise ValueError('Unexpected label values found in the validation set:'
                         ' {unexpected_labels}. Please make sure that the '
                         'labels in the validation set are in the same range '
                         'as training labels.'.format(
                             unexpected_labels=unexpected_labels))
    
    # Create model instance.
    model = build_models.naive_bayes(alpha=alpha)    
    
    # Train and validate model.
    model.fit(X_train_vectors, y_train)
    
    # Print results.
    #print("the performances of the naive bayes model are:")
    #print('Validation accuracy: {acc}, loss: {loss}'.format(acc=history['val_accuracy'][-1], loss=history['val_loss'][-1]))

    #Get the date and time
    now = utils_general_porpose.get_time()

    #Get the list of models in the directory
    list_models = utils_general_porpose.extract_name_model(path_model_save, ".pkl")

    #Get the last version of the model
    last_version = utils_general_porpose.extract_last_version_model(list_models)

    #Get the number of the new version
    version = str(utils_general_porpose.counter_version(last_version))

    #calculate accuracy
    acc = utils_performance_analysis.accuracy(y_test, model.predict(X_test_vectors))

    #calculate loss
    loss = "Naive Bayes does not have loss"
    
    # Save model.
    name = 'naive_bayes_model_v' + version +'.pkl'
    pickle.dump(model, open(path_model_save + name, "wb"))
    return now, name, acc, loss, model, num_classes