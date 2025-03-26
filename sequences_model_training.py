#print("1 primera squences")
import os
#print("2 sequences")
import utils_train_models
#print("3 sequeces")
import utils_performance_analysis
#print("4 sequences")
import utils_general_porpose

"""#Function to create a directory to save models if it does not exist
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path"""

#Function to train the model
def run_mlp_model(X_train_vectors,
                      y_train,
                      X_test_vectors,
                      y_test,                      
                      vec_ngram_params,
                      path_vectorization, 
                      hyper_params_mlp,                       
                      path_model_save,
                      dataset_name):
    
    #Train the MLP model
    now, model_name, acc, loss, model, num_classes = utils_train_models.train_mlp_model(X_train_vectors,
                                                            y_train,
                                                            X_test_vectors,
                                                            y_test,                                                            
                                                            hyper_params_mlp,
                                                            path_model_save)
    
    #Make predictions in training
    y_pred_train = utils_performance_analysis.predict_values(model, X_train_vectors, num_classes)

    #make predictions in test
    y_pred_test = utils_performance_analysis.predict_values(model, X_test_vectors, num_classes)

    #Get the metrics train
    precision_train, recall_train, f1_train = utils_performance_analysis.get_model_metrics(y_train, y_pred_train, num_classes)

    #Get the metrics test
    precision_test, recall_test, f1_test = utils_performance_analysis.get_model_metrics(y_test, y_pred_test, num_classes)
    
    #save the info and results of the model
    vectorize_technique = vec_ngram_params["vectorize_technique"]
    vectorization_hyperparameters = vec_ngram_params    
    model_hyperparameters = hyper_params_mlp
    acc = round(acc, 3)
    loss = round(loss, 3)

    current_path = utils_general_porpose.get_current_path()
    path_save = current_path + "/" + "performance_report.csv"
    utils_performance_analysis.save_model_info(now, dataset_name, num_classes, vectorize_technique, vectorization_hyperparameters, path_vectorization, model_name, model_hyperparameters, acc, loss, precision_train, recall_train, f1_train, precision_test, recall_test, f1_test, path_save)

    #Print the results for quick check
    return now, dataset_name, num_classes, vectorize_technique, vectorization_hyperparameters, path_vectorization, model_name, model_hyperparameters, acc, loss, precision_train, recall_train, f1_train, precision_test, recall_test, f1_test
    
#Function to train the lstm model
def run_lstm_model(X_train,
                      y_train,
                      X_test,
                      y_test,     
                      vec_seq_params,
                      path_vectorization, 
                      hyper_params_lstm, 
                      path_model_save,
                      dataset_name):
    
    #Train the LSTM model
    now, model_name, acc, loss, model, num_classes = utils_train_models.train_lstm_model(X_train,
                                                            y_train,
                                                            X_test,
                                                            y_test,                                                            
                                                            hyper_params_lstm,
                                                            path_model_save)
    
    #Make predictions in training
    y_pred_train = utils_performance_analysis.predict_values(model, X_train, num_classes)

    #make predictions in test
    y_pred_test = utils_performance_analysis.predict_values(model, X_test, num_classes)

    #Get the metrics train
    precision_train, recall_train, f1_train = utils_performance_analysis.get_model_metrics(y_train, y_pred_train, num_classes)

    #Get the metrics test
    precision_test, recall_test, f1_test = utils_performance_analysis.get_model_metrics(y_test, y_pred_test, num_classes)
    
    #save the info and results of the model
    vectorize_technique = vec_seq_params["vectorize_technique"]
    vectorization_hyperparameters = vec_seq_params    
    model_hyperparameters = hyper_params_lstm
    acc = round(acc, 3)
    loss = round(loss, 3)
    
    current_path = utils_general_porpose.get_current_path()
    path_save = current_path + "/" + "performance_report.csv"
    #print(path_save)
    utils_performance_analysis.save_model_info(now, dataset_name, num_classes, vectorize_technique, vectorization_hyperparameters, path_vectorization, model_name, model_hyperparameters, acc, loss, precision_train, recall_train, f1_train, precision_test, recall_test, f1_test, path_save)

    #Print the results for quick check
    return now, dataset_name, num_classes, vectorize_technique, vectorization_hyperparameters, path_vectorization, model_name, model_hyperparameters, acc, loss, precision_train, recall_train, f1_train, precision_test, recall_test, f1_test

#function to train the logistic regression model
def run_logistic_regression(X_train_vectors,
                      y_train,
                      X_test_vectors,
                      y_test,                      
                      vec_ngram_params,
                      path_vectorization, 
                      hyper_params_logistic,                       
                      path_model_save,
                      dataset_name):
    
    #Train the logistic regression model
    now, model_name, acc, loss, model, num_classes = utils_train_models.train_logistic_regression(X_train_vectors,
                                                            y_train,
                                                            X_test_vectors,
                                                            y_test,                                                            
                                                            hyper_params_logistic,
                                                            path_model_save)
    
    #Make predictions in training
    y_pred_train = utils_performance_analysis.predict_values_logistic_regression(model, X_train_vectors)

    #make predictions in test
    y_pred_test = utils_performance_analysis.predict_values_logistic_regression(model, X_test_vectors)

    #Get the metrics train
    precision_train, recall_train, f1_train = utils_performance_analysis.get_model_metrics(y_train, y_pred_train, num_classes)

    #Get the metrics test
    precision_test, recall_test, f1_test = utils_performance_analysis.get_model_metrics(y_test, y_pred_test, num_classes)
    
    #save the info and results of the model
    vectorize_technique = vec_ngram_params["vectorize_technique"]
    vectorization_hyperparameters = vec_ngram_params    
    model_hyperparameters = hyper_params_logistic
    acc = round(acc, 3)
    #loss = round(loss, 3)

    current_path = utils_general_porpose.get_current_path()
    path_save = current_path + "/" + "performance_report.csv"
    utils_performance_analysis.save_model_info(now, dataset_name, num_classes, vectorize_technique, vectorization_hyperparameters, path_vectorization, model_name, model_hyperparameters, acc, loss, precision_train, recall_train, f1_train, precision_test, recall_test, f1_test, path_save)

    #Print the results for quick check
    return now, dataset_name, num_classes, vectorize_technique, vectorization_hyperparameters, path_vectorization, model_name, model_hyperparameters, acc, loss, precision_train, recall_train, f1_train, precision_test, recall_test, f1_test

#function to train the naive bayes model
def run_naive_bayes(X_train_vectors,
                      y_train,
                      X_test_vectors,
                      y_test,                      
                      vec_ngram_params, 
                      hyper_params_naive_bayes,                       
                      path_model_save,
                      dataset_name):
    
    #Train the naive bayes model
    now, model_name, acc, loss, model, num_classes = utils_train_models.train_naive_bayes(X_train_vectors,
                                                            y_train,
                                                            X_test_vectors,
                                                            y_test,                                                            
                                                            hyper_params_naive_bayes,
                                                            path_model_save)
    
    #Make predictions in training
    y_pred_train = utils_performance_analysis.predict_values_logistic_regression(model, X_train_vectors)

    #make predictions in test
    y_pred_test = utils_performance_analysis.predict_values_logistic_regression(model, X_test_vectors)

    #Get the metrics train
    precision_train, recall_train, f1_train = utils_performance_analysis.get_model_metrics(y_train, y_pred_train, num_classes)

    #Get the metrics test
    precision_test, recall_test, f1_test = utils_performance_analysis.get_model_metrics(y_test, y_pred_test, num_classes)
    
    #save the info and results of the model
    vectorize_technique = vec_ngram_params["vectorize_technique"]
    vectorization_hyperparameters = vec_ngram_params    
    model_hyperparameters = hyper_params_naive_bayes
    acc = round(acc, 3)
    #loss = round(loss, 3)

    current_path = utils_general_porpose.get_current_path()
    path_save = current_path + "/" + "performance_report.csv"
    utils_performance_analysis.save_model_info(now, dataset_name, num_classes, vectorize_technique, vectorization_hyperparameters, model_name, model_hyperparameters, acc, loss, precision_train, recall_train, f1_train, precision_test, recall_test, f1_test, path_save)

    #Print the results for quick check
    return now, dataset_name, num_classes, vectorize_technique, vectorization_hyperparameters, model_name, model_hyperparameters, acc, loss, precision_train, recall_train, f1_train, precision_test, recall_test, f1_test
