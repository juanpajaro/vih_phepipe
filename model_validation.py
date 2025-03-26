#!/usr/bin/env python3
import os
import csv
import utils_performance_analysis
import utils_explore_data
#path = "/home/pajaro/pipeline_project_v2/computational_pipe_v2"

#TODO: no se usa, eliminar
def run_model_validation_pipe(path, metric_compare):
    #Get the models info
    list_models_info = utils_performance_analysis.get_models_info(path + "/performance_report.csv")
    print("info models loaded")
    #Get the best model
    best_model_info = utils_performance_analysis.get_best_model(list_models_info, metric_compare)
    print("best model info loaded")
    #load the best model
    model = utils_performance_analysis.load_model(path, best_model_info)
    print("best model loaded")
    #Get the test set
    X_test, y_test = utils_performance_analysis.get_test_set(path)
    print("test set loaded")
    #Make predictions using the best model and test set
    num_classes = utils_explore_data.get_num_classes(y_test)
    y_pred = utils_performance_analysis.predict_values(model, X_test, num_classes)
    print("predictions made")
    #Get the metrics
    precision, recall, f1 = utils_performance_analysis.get_model_metrics(y_test, y_pred, num_classes)    
    print("metrics calculated")
    
    #save the info and results of the model
    #save verification results
    path_save = path + "/verification_report.csv"
    if os.path.exists(path_save):
        with open(path_save, mode='a') as file:
            writer = csv.writer(file)
            writer.writerow([best_model_info["model_name"], precision, recall, f1])
    else:
        with open(path_save, mode='w') as file:
            writer = csv.writer(file)
            writer.writerow(["model_name", "precision", "recall", "f1"])
            writer.writerow([best_model_info["model_name"], precision, recall, f1])
    print("results saved in verification_report.csv")

    return best_model_info["model_name"], precision, recall, f1    

