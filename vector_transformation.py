#print("1. vector trans")
import numpy as np
#print("2. vector trans")
import utils_general_porpose
#print("3. vector trans")
import utils_vector_transformations
#print("4. vector trans")
import os

class VectorTransformation:
    def __new__(cls, *args, **kwargs):
        print("VectorTransformation object created")
        return super().__new__(cls)
    
    def __init__(self, path_data_train, path_data_test, vec_ngram_params, vec_seq_params, data_name, file_data, path_var):
        self.path_data_train = path_data_train
        self.path_data_test = path_data_test
        self.vec_ngram_params = vec_ngram_params
        self.vec_seq_params = vec_seq_params
        self.data_name = data_name
        self.file_data = file_data
        self.current_path = None
        self.path_var = path_var
        print("VectorTransformation object instantiated")

    def __repr__(self):
        return "filter_name: {}, p_data_train: {}, p_data_test: {}, vec_ngram_params: {}, vec_seq_params: {}, data_name: {}, file_data {}, path_var {}".format(type(self).__name__, self.path_data_train, self.path_data_test, self.ngram_params, self.seq_params, self.data_name, self.file_data, self.path_var)


    #Function to transform the clinical concepts into vectors
    def run_vector_transformation_pipe(self):
        
        #get the current path
        self.current_path = utils_general_porpose.get_current_path()

        #load the data_train
        data_train = utils_general_porpose.load_json(self.current_path, self.path_data_train)
        X_train = utils_vector_transformations.get_seq(data_train, self.path_var)
        y_train = utils_vector_transformations.get_labels(data_train)
        y_train = np.array(y_train)

        #load the data_test
        data_test = utils_general_porpose.load_json(self.current_path, self.path_data_test)
        X_test = utils_vector_transformations.get_seq(data_test, self.path_var)
        y_test = utils_vector_transformations.get_labels(data_test)
        y_test = np.array(y_test)

        print("type text data: {}, type label: {}".format(type(X_train), type(y_train)))
        print("class balance train: {}".format(np.unique(y_train, return_counts=True)))
        print("class balance test: {}".format(np.unique(y_test, return_counts=True)))

        #create the path to save the vectors
        path_vectors = utils_general_porpose.create_directory(self.current_path + self.file_data + self.data_name + "/")
        print(path_vectors)
        utils_vector_transformations.save_pickle_file(y_train, path_vectors + "y_train.pkl")
        utils_vector_transformations.save_pickle_file(y_test, path_vectors + "y_test.pkl")

        #load the ngram dictionary parameters
        #vec_ngram_params = utils_general_porpose.load_json(self.current_path, self.ngram_params)
        self.vec_ngram_params["NGRAM_RANGE"] = utils_vector_transformations.convert_to_tuple(self.vec_ngram_params["NGRAM_RANGE"])

        #Vectorize the text using n-grams
        X_train_vectors, X_test_vectors, ngram_vectorizer, selector, list_p_values, list_selected_features = utils_vector_transformations.ngram_vectorize(X_train, y_train, X_test, self.vec_ngram_params)
        print("data vectorized using n-grams")
        #save vectors
        utils_vector_transformations.save_pickle_file(X_train_vectors, path_vectors + "X_train_vectors.pkl")
        utils_vector_transformations.save_pickle_file(X_test_vectors, path_vectors + "X_test_vectors.pkl")

        #ngram dictorionary parameters
        #utils_general_porpose.save_json(vec_ngram_params, path_vectors_save + "vec_ngram_params.json")

        #save the ngram_vectorizer and selector
        utils_vector_transformations.save_pickle_file(ngram_vectorizer, path_vectors + "ngram_vectorizer.pkl")
        utils_vector_transformations.save_pickle_file(selector, path_vectors + "selector.pkl")

        #load the sequence dictionary parameters
        #vec_seq_params = utils_general_porpose.load_json(self.current_path, self.seq_params)

        #Convert the text into sequences
        X_train_sequences, X_test_sequences, tokenizer = utils_vector_transformations.sequence_vectorize(X_train, X_test, self.vec_seq_params)
        print("data vectorized using sequences")

        #info: this are only indeces of the selected features. TODO: the next step is to make a Word2Vec function to convert word to vectors embeddings
        utils_vector_transformations.save_pickle_file(X_train_sequences, path_vectors + "X_train_sequences.pkl")
        utils_vector_transformations.save_pickle_file(X_test_sequences, path_vectors + "X_test_sequences.pkl")
        utils_vector_transformations.save_pickle_file(tokenizer, path_vectors + "tokenizer.pkl")    

        #save the sequence dictionary parameters
        #utils_general_porpose.save_json(vec_seq_params, self.path_vectors_save + "vec_seq_params.json")

        print("type ngram vector data: {}, type sequence vector data: {}".format(type(X_train_vectors), type(X_train_sequences)))
        print("type y_train data: {}, type y_test data: {}".format(type(y_train), type(y_test)))
        print("type ngram vectorizer: {}, type tokenizer: {}".format(type(ngram_vectorizer), type(tokenizer)))
        print("X_train_vectors shape: {}, X_test_vectors shape: {}".format(X_train_vectors.shape, X_test_vectors.shape))
        print("X_train_sequences shape: {}, X_test_sequences shape: {}".format(X_train_sequences.shape, X_test_sequences.shape))
        print("vec_ngram_params: {}".format(self.vec_ngram_params))
        print("type vec_ngram_params: {}".format(type(self.vec_ngram_params)))
        print("vec_seq_params: {}".format(vec_seq_params))
        print("type vec_seq_params: {}".format(type(vec_seq_params)))
        print("data_name: {}".format(self.data_name))
        #return print("vector transformation process finished")
        return X_train_vectors, X_test_vectors, ngram_vectorizer, X_train_sequences, X_test_sequences, tokenizer, selector, list_p_values, list_selected_features, y_train, y_test

#python3
#from vector_transformation import VectorTransformation
#path_data_train = "/datasets/train/train.json"
#path_data_test = "/datasets/test/test.json"
#path_ngram_params = "/parameters/ngram_params.json"
#path_seq_params = "/parameters/seq_params.json"
#data_name = "Raw_EHR_OSA"
#file_data = "/vectors/"
#path_var = "seq"
#transformation = VectorTransformation(path_data_train, path_data_test, path_ngram_params, path_seq_params, data_name, file_data, path_var)
#transformation
#transformation.run_vector_transformation_pipe()

#python3
#from vector_transformation import VectorTransformation
#path_data_train = "/concepts/train/train.json"
#path_data_test = "/concepts/test/test.json"
#path_ngram_params = "/parameters/ngram_params.json"
#path_seq_params = "/parameters/seq_params.json"
#data_name = "Entities_EHR_OSA"
#file_data = "/vectors/"
#path_var = "entities"
#transformation = VectorTransformation(path_data_train, path_data_test, path_ngram_params, path_seq_params, data_name, file_data, path_var)
#transformation
#transformation.run_vector_transformation_pipe()
