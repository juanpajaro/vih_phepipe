#!/usr/bin/env python3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import numpy as np
import pickle

#https://stackoverflow.com/questions/54025795/how-to-fix-modulenotfounderror-no-module-named-tensorflow-python-keras-impo
from keras.preprocessing import sequence
from keras.preprocessing import text


#TODO: write the readme file using the cavenats of installation modules, libraries, packages and how to use the functions
#https://www.tensorflow.org/install/pip#windows-wsl2
#https://scikit-learn.org/stable/install.html
#quick_umls how to install according to your experience

#Function to vectorize the text using n-grams
def ngram_vectorize(train_texts, train_labels, val_texts, vec_ngram_params):
    """Vectorizes texts as ngram vectors.

    1 text = 1 tf-idf vector the length of vocabulary of uni-grams + bi-grams.

    # Arguments
        train_texts: list, training text strings.
        train_labels: np.ndarray, training labels.
        val_texts: list, validation text strings.

    # Returns
        x_train, x_val: vectorized training and validation texts
    """
    # Get the hyperparameters n-gram from the dictionary
    NGRAM_RANGE = vec_ngram_params['NGRAM_RANGE']
    TOP_K = vec_ngram_params['TOP_K']
    TOKEN_MODE = vec_ngram_params['TOKEN_MODE']
    MIN_DOCUMENT_FREQUENCY = vec_ngram_params['MIN_DOCUMENT_FREQUENCY']


    # Create keyword arguments to pass to the 'tf-idf' vectorizer.
    kwargs = {
            'ngram_range': NGRAM_RANGE,  # Use 1-grams + 2-grams.
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': TOKEN_MODE,  # Split text into word tokens.
            'min_df': MIN_DOCUMENT_FREQUENCY,
            'lowercase': False
    }
    ngram_vectorizer = TfidfVectorizer(**kwargs)

    # Learn vocabulary from training texts and vectorize training texts.
    x_train = ngram_vectorizer.fit_transform(train_texts)

    # Vectorize validation texts.
    x_val = ngram_vectorizer.transform(val_texts)

    # Select top 'k' of the vectorized features.
    selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
    selector.fit(x_train, train_labels)
    x_train = selector.transform(x_train)
    x_val = selector.transform(x_val)

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')

    #print("Data vectorized using n-grams")
    #print("X_train shape".format(x_train.shape))
    return x_train, x_val, ngram_vectorizer, selector, selector.pvalues_, selector.get_support(indices=True)

#Function to convert the text into index vectors
def sequence_vectorize(train_texts, val_texts, vec_seq_params):
    """Vectorizes texts as sequence vectors.

    1 text = 1 sequence vector with fixed length.

    # Arguments
        train_texts: list, training text strings.
        val_texts: list, validation text strings.

    # Returns
        x_train, x_val, word_index: vectorized training and validation
            texts and word index dictionary.
    """
    #Get the hyperparameters from the dictionary
    MAX_SEQUENCE_LENGTH = vec_seq_params['MAX_SEQUENCE_LENGTH']
    TOP_K = vec_seq_params['TOP_K']

    # Create vocabulary with training texts.
    #https://faroit.com/keras-docs/1.2.2/preprocessing/text/
    tokenizer = text.Tokenizer(num_words=TOP_K, lower= False)
    tokenizer.fit_on_texts(train_texts)

    # Vectorize training and validation texts.
    x_train = tokenizer.texts_to_sequences(train_texts)
    x_val = tokenizer.texts_to_sequences(val_texts)

    # Get max sequence length.
    max_length = len(max(x_train, key=len))
    if max_length > MAX_SEQUENCE_LENGTH:
        max_length = MAX_SEQUENCE_LENGTH

    # Fix sequence length to max value. Sequences shorter than the length are
    # padded in the beginning and sequences longer are truncated
    # at the beginning.
    x_train = sequence.pad_sequences(x_train, maxlen=max_length)
    x_val = sequence.pad_sequences(x_val, maxlen=max_length)
    #print("Data vectorized using sequences")
    return x_train, x_val, tokenizer.word_index, tokenizer

#TODO: a file for explore data in all the pipeline

#Function to save a numpy array in a file
def save_numpy_array(array, path):
    np.save(path, array)
    return path

#Function to save a pickle object in a file
def save_pickle_file(obj, path):
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path

#Function to load a pickle object from a file
def load_pickle_file(path):
    with open(path, 'rb') as handle:
        obj = pickle.load(handle)
    return obj

#Function to get the sequence of the data
def get_seq(data, path_var):
    seq = [patient.get(path_var) for patient in data]    
    return seq

#Function to get the labels of the data
def get_labels(data):
    labels = [patient.get("label") for patient in data]
    return labels

#function to convert a list into a tuple
def convert_to_tuple(list):
    return tuple(list)