#!/usr/bin/env python3
#print("1. primera estacion")
import numpy as np
#print("2. segunda estacion")
import utils_general_porpose
#print("3. tercera estatcion")
import utils_vector_transformations
#print("4. estacion")
import sequences_model_training
#print("fin del viacrucis")

class Pipe:
    def __new__(cls, *args, **kwargs):
        print("Pipe object created")
        return super().__new__(cls)
    
    def __init__(self, 
                 path_data_train, 
                 path_data_test, 
                 file_vector_paramameters, 
                 name_file_ngram_params, 
                 name_file_seq_params,
                 data_name, 
                 file_data, 
                 path_var,
                 file_models_parameters, 
                 name_file_hyper_params_mlp,
                 name_file_hyper_params_lr, 
                 name_file_hyper_params_lstm):
        
        self.path_data_train = path_data_train
        self.path_data_test = path_data_test
        self.file_vector_params = file_vector_paramameters
        self.name_file_ngram_params = name_file_ngram_params
        self.name_file_seq_params = name_file_seq_params
        self.data_name = data_name
        self.file_data = file_data        
        self.path_var = path_var
        self.file_models_parameters = file_models_parameters
        self.name_file_hyper_params_mlp = name_file_hyper_params_mlp
        self.name_file_hyper_params_lr = name_file_hyper_params_lr
        self.name_file_hyper_params_lstm = name_file_hyper_params_lstm

        self.current_path = None

        self.data_train = None
        self.data_test = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.X_train_vectors = None
        self.X_test_vectors = None
        self.path_vectors = None
        self.ngram_vectorizer = None
        self.selector = None

        self.X_train_sequences = None
        self.X_test_sequences = None
        self.tokenizer = None
        self.tokenizer_obj = None

    
    def __repr__(self):
        return "filter_name: {}, p_data_train: {}, p_data_test: {}, data_name: {}".format(type(self).__name__, self.path_data_train, self.path_data_test, self.data_name)
    
    def create_version_directory(self):        

        _name_directories = utils_general_porpose.directories_in_path(self.current_path + self.file_data + self.data_name + "/")
        print(self.current_path + self.file_data + self.data_name)
        print(_name_directories)

        #if _name_directories is empty
        if not _name_directories:
            self.path_vectors = utils_general_porpose.create_directory(self.current_path + self.file_data + self.data_name + "/" + "01/")
            print(self.path_vectors)
        
        else:
            last_version = utils_general_porpose.extract_last_version_model(_name_directories)
            print(last_version)
            self.path_vectors = utils_general_porpose.create_directory(self.current_path + self.file_data + self.data_name + "/" + "{:02d}".format(last_version + 1) + "/")
            print(self.path_vectors)

        
    def load_data(self):
        
        self.data_train = utils_general_porpose.load_json(self.current_path, self.path_data_train)
        print("data_train loaded")
        self.X_train = utils_vector_transformations.get_seq(self.data_train, self.path_var)
        y_train = utils_vector_transformations.get_labels(self.data_train)
        self.y_train = np.array(y_train)        
         
        self.data_test = utils_general_porpose.load_json(self.current_path, self.path_data_test)
        print("data_test loaded")
        self.X_test = utils_vector_transformations.get_seq(self.data_test, self.path_var)
        y_test = utils_vector_transformations.get_labels(self.data_test)
        self.y_test = np.array(y_test)
        
    def vectorize_ngram_data(self, vec_ngram_params):
        vec_ngram_params["NGRAM_RANGE"] = utils_vector_transformations.convert_to_tuple(vec_ngram_params["NGRAM_RANGE"])
        self.X_train_vectors, self.X_test_vectors, self.ngram_vectorizer, self.selector, list_p_values, list_selected_features = utils_vector_transformations.ngram_vectorize(self.X_train, self.y_train, self.X_test, vec_ngram_params)

    #save vectorized data
    def save_vectorized_data(self):
        #save y_train and y_test vectors
        utils_vector_transformations.save_pickle_file(self.y_train, self.path_vectors + "y_train.pkl")
        utils_vector_transformations.save_pickle_file(self.y_test, self.path_vectors + "y_test.pkl")

        #save X_train and X_test vectors
        utils_vector_transformations.save_pickle_file(self.X_train_vectors, self.path_vectors + "X_train_vectors.pkl")
        utils_vector_transformations.save_pickle_file(self.X_test_vectors, self.path_vectors + "X_test_vectors.pkl")

        #save the ngram_vectorizer and selector
        utils_vector_transformations.save_pickle_file(self.ngram_vectorizer, self.path_vectors + "ngram_vectorizer.pkl")
        utils_vector_transformations.save_pickle_file(self.selector, self.path_vectors + "selector.pkl")


    def __train_mlp_model(self, vec_ngram_params, hyper_params_mlp):
        path_models = utils_general_porpose.create_directory(self.current_path + "/models/")
        print(path_models)
        now, dataset_name, num_classes, vectorize_technique, vectorization_hyperparameters, path_vectorization, model_name, model_hyperparameters, acc, loss, precision_train, recall_train, f1_train, precision_test, recall_test, f1_test = sequences_model_training.run_mlp_model(self.X_train_vectors,
                                                                                                                                                                                        self.y_train, 
                                                                                                                                                                                        self.X_test_vectors,
                                                                                                                                                                                        self.y_test,
                                                                                                                                                                                        vec_ngram_params,
                                                                                                                                                                                        self.path_vectors, 
                                                                                                                                                                                        hyper_params_mlp,                                                                                                                                                                                         
                                                                                                                                                                                        path_models,
                                                                                                                                                                                        self.data_name)
        print(now, dataset_name, num_classes, vectorize_technique, vectorization_hyperparameters, path_vectorization, model_name, model_hyperparameters, acc, loss, precision_train, recall_train, f1_train, precision_test, recall_test, f1_test)

    #Train logistic regression
    def __train_lr_model(self, vec_ngram_params, hyper_params_lr):        
        path_models = utils_general_porpose.create_directory(self.current_path + "/models/")
        print(path_models)        

        now, dataset_name, num_classes, vectorize_technique, vectorization_hyperparameters, path_vectorization, model_name, model_hyperparameters, acc, loss, precision_train, recall_train, f1_train, precision_test, recall_test, f1_test = sequences_model_training.run_logistic_regression(self.X_train_vectors,
                            self.y_train,
                            self.X_test_vectors,
                            self.y_test,                      
                            vec_ngram_params,
                            self.path_vectors, 
                            hyper_params_lr,                       
                            path_models,
                            self.data_name)

        print(now, dataset_name, num_classes, vectorize_technique, vectorization_hyperparameters, path_vectorization, model_name, model_hyperparameters, acc, loss, precision_train, recall_train, f1_train, precision_test, recall_test, f1_test)


    #Function to run NO sequencial models
    def run_no_seq_models(self):

        #get the current path
        self.current_path = utils_general_porpose.get_current_path()

        #load the data_train and data_test
        self.load_data()
        
        #load the ngram list of parameters
        list_ngram_params = utils_general_porpose.load_json(self.current_path + self.file_vector_params, self.name_file_ngram_params)
        #print(len(list_ngram_params))

        #load the list of hyper parameters for the mlp model
        list_hyper_params_mlp = utils_general_porpose.load_json(self.current_path + self.file_models_parameters, self.name_file_hyper_params_mlp)        

        #load the list of hyper parameters for the logistic regression model
        list_hyper_params_lr = utils_general_porpose.load_json(self.current_path + self.file_models_parameters, self.name_file_hyper_params_lr)

        for i in range(len(list_ngram_params)):
            
            #create the version directory for vectorized data
            self.create_version_directory()

            vec_ngram_params = list_ngram_params[i]
            print("ngram index: {}".format(i))
            print("ngram parameters: {}".format(vec_ngram_params))
            
            self.vectorize_ngram_data(vec_ngram_params)
            print(self.X_train_vectors.shape, self.X_test_vectors.shape)

            self.save_vectorized_data()
            
            for i in range(len(list_hyper_params_mlp)):
                hyper_params_mlp = list_hyper_params_mlp[i]
                print("hyper parameters mlp: {}".format(hyper_params_mlp))
                self.__train_mlp_model(vec_ngram_params, hyper_params_mlp)
                print("---------------------------------")
            
            for i in range(len(list_hyper_params_lr)):
                hyper_params_lr = list_hyper_params_lr[i]
                print("hyper parameters lr: {}".format(hyper_params_lr))
                self.__train_lr_model(vec_ngram_params, hyper_params_lr)
                print("---------------------------------")

    def tokenize_data(self, vec_seq_params): 
        self.X_train_sequences, self.X_test_sequences, self.tokenizer, self.tokenizer_obj = utils_vector_transformations.sequence_vectorize(self.X_train, self.X_test, vec_seq_params)
        #print(self.X_train_sequences.shape, self.X_test_sequences.shape)

    def save_sequences(self):
        #save the sequences
        utils_vector_transformations.save_pickle_file(self.X_train_sequences, self.path_vectors + "X_train_sequences.pkl")
        utils_vector_transformations.save_pickle_file(self.X_test_sequences, self.path_vectors + "X_test_sequences.pkl")
        utils_vector_transformations.save_pickle_file(self.tokenizer, self.path_vectors + "tokenizer.pkl")
        utils_vector_transformations.save_pickle_file(self.tokenizer_obj, self.path_vectors + "tokenizer_obj.pkl")

    def train_lstm_model(self, vec_seq_params, hyper_params_lstm):
        path_models = utils_general_porpose.create_directory(self.current_path + "/models/")
        print(path_models)
        now, dataset_name, num_classes, vectorize_technique, vectorization_hyperparameters, path_vectorization, model_name, model_hyperparameters, acc, loss, precision_train, recall_train, f1_train, precision_test, recall_test, f1_test = sequences_model_training.run_lstm_model(self.X_train_sequences,
                                                                                                                                                                                        self.y_train, 
                                                                                                                                                                                        self.X_test_sequences,
                                                                                                                                                                                        self.y_test,
                                                                                                                                                                                        vec_seq_params,
                                                                                                                                                                                        self.path_vectors, 
                                                                                                                                                                                        hyper_params_lstm,                                                                                                                                                                                         
                                                                                                                                                                                        path_models,
                                                                                                                                                                                        self.data_name)
        print(now, dataset_name, num_classes, vectorize_technique, vectorization_hyperparameters, path_vectorization, model_name, model_hyperparameters, acc, loss, precision_train, recall_train, f1_train, precision_test, recall_test, f1_test)

#Function to run sequencial models
    def run_seq_models(self):
        #get the current path
        self.current_path = utils_general_porpose.get_current_path()

        #load the data_train and data_test
        self.load_data()
        
        #load the ngram list of parameters
        list_seq_params = utils_general_porpose.load_json(self.current_path + self.file_vector_params, self.name_file_seq_params)
        print(len(list_seq_params))
        print("list seq params loaded")

        #load the list of hyper parameters for the lstm model
        list_hyper_params_lstm = utils_general_porpose.load_json(self.current_path + self.file_models_parameters, self.name_file_hyper_params_lstm)
        print(len(list_hyper_params_lstm))

        for i in range(len(list_seq_params)):
            #create the version directory for vectorized data
            self.create_version_directory()

            vec_seq_params = list_seq_params[i]
            print("seq index: {}".format(i))
            print("seq parameters: {}".format(vec_seq_params))

            self.tokenize_data(vec_seq_params)
            print(self.X_train_sequences.shape, self.X_test_sequences.shape)

            self.save_sequences()

            for i in range(len(list_hyper_params_lstm)):
                hyper_params_lstm = list_hyper_params_lstm[i]
                hyper_params_lstm["num_features"] = min(len(self.tokenizer) + 1, vec_seq_params["TOP_K"])
                print("hyper parameters lstm: {}".format(hyper_params_lstm))
                self.train_lstm_model(vec_seq_params, hyper_params_lstm)
                print("---------------------------------")

"""
path_data_train = "/concepts/train/train.json"
path_data_test = "/concepts/test/test.json"
file_vectors_parameters = "/parameters/"
name_file_ngram_params = "list_ngram_params.json"
name_file_seq_params = "list_seq_params.json"
data_name = "Entities_EHR_OSA"
file_data = "/vectors/"
path_var = "entities"
file_models_parameters = "/models_parameters/"
name_file_hyper_params_mlp = "list_hyper_params_mlp.json"
name_file_hyper_params_lr = "list_hyper_params_logistic_regression.json"
name_file_hyper_params_lstm = "list_hyper_params_lstm.json"

pipe = Pipe(path_data_train, 
            path_data_test, 
            file_vectors_parameters, 
            name_file_ngram_params, 
            name_file_seq_params, 
            data_name, 
            file_data, 
            path_var, 
            file_models_parameters, 
            name_file_hyper_params_mlp, 
            name_file_hyper_params_lr, name_file_hyper_params_lstm)

pipe
pipe.run_no_seq_models()
pipe.run_seq_models()
"""