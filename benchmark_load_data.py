#!/usr/bin/env python3
import os
import random
import pandas as pd
import numpy as np

def load_imdb_sentiment_analysis_dataset(data_path, seed=123):
    """Loads the Imdb movie reviews sentiment analysis dataset.

    # Arguments
        data_path: string, path to the data directory.
        seed: int, seed for randomizer.

    # Returns
        A tuple of training and validation data.
        Number of training samples: 25000
        Number of test samples: 25000
        Number of categories: 2 (0 - negative, 1 - positive)

    # References
        Mass et al., http://www.aclweb.org/anthology/P11-1015

        Download and uncompress archive from:
        http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
    """
    imdb_data_path = os.path.join(data_path, 'aclImdb')

    # Load the training data
    train_texts = []
    train_labels = []
    for category in ['pos', 'neg']:
        train_path = os.path.join(imdb_data_path, 'train', category)
        for fname in sorted(os.listdir(train_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(train_path, fname)) as f:
                    train_texts.append(f.read())
                train_labels.append(0 if category == 'neg' else 1)

    # Load the validation data.
    test_texts = []
    test_labels = []
    for category in ['pos', 'neg']:
        test_path = os.path.join(imdb_data_path, 'test', category)
        for fname in sorted(os.listdir(test_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(test_path, fname)) as f:
                    test_texts.append(f.read())
                test_labels.append(0 if category == 'neg' else 1)

    # Shuffle the training data and labels.
    random.seed(seed)
    random.shuffle(train_texts)
    random.seed(seed)
    random.shuffle(train_labels)

    return ((train_texts, np.array(train_labels)),
            (test_texts, np.array(test_labels)))

def load_rotten_tomatoes_sentiment_analysis_dataset(data_path,
                                                    validation_split=0.2,
                                                    seed=123):
    """Loads the rotten tomatoes sentiment analysis dataset.

    # Arguments
        data_path: string, path to the data directory.
        validation_split: float, percentage of data to use for validation.
        seed: int, seed for randomizer.

    # Returns
        A tuple of training and validation data.
        Number of training samples: 124848
        Number of test samples: 31212
        Number of categories: 5 (0 - negative, 1 - somewhat negative,
                2 - neutral, 3 - somewhat positive, 4 - positive)

    # References
        https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data

        Download and uncompress archive from:
        https://www.kaggle.com/c/3810/download/train.tsv.zip
    """
    columns = (2, 3)  # 2 - Phrases, 3 - Sentiment.
    data = _load_and_shuffle_data(data_path, 'train.tsv', columns, seed, '\t')

    # Get the review phrase and sentiment values.
    texts = list(data['Phrase'])
    labels = np.array(data['Sentiment'])
    return _split_training_and_validation_sets(texts, labels, validation_split)

def _load_and_shuffle_data(data_path,
                           file_name,
                           cols,
                           seed,
                           separator=',',
                           header=0):
    """Loads and shuffles the dataset using pandas.

    # Arguments
        data_path: string, path to the data directory.
        file_name: string, name of the data file.
        cols: list, columns to load from the data file.
        seed: int, seed for randomizer.
        separator: string, separator to use for splitting data.
        header: int, row to use as data header.
    """
    np.random.seed(seed)
    data_path = os.path.join(data_path, file_name)
    data = pd.read_csv(data_path, usecols=cols, sep=separator, header=header)
    return data.reindex(np.random.permutation(data.index))


def _split_training_and_validation_sets(texts, labels, validation_split):
    """Splits the texts and labels into training and validation sets.

    # Arguments
        texts: list, text data.
        labels: list, label data.
        validation_split: float, percentage of data to use for validation.

    # Returns
        A tuple of training and validation data.
    """
    num_training_samples = int((1 - validation_split) * len(texts))
    return ((texts[:num_training_samples], labels[:num_training_samples]),
            (texts[num_training_samples:], labels[num_training_samples:]))

def load_medical_error_dataset(data_path, validation_split=0.2, seed=123):
    """Loads the medical error dataset.

    # Arguments
        data_path: string, path to the data directory.
        validation_split: float, percentage of data to use for validation.
        seed: int, seed for randomizer.

    # Returns
        A tuple of training and validation data.
        Number of training samples: 640
        Number of test samples: 160
        Number of categories: 2 (0 - no, 1 - yes)

    # References
        https://archive.ics.uci.edu/ml/datasets/Medical+Quality
    """
    columns = (2, 4)  # 0 - Patient ID, 1 - Medical error.
    data = _load_and_shuffle_data(data_path, 'MEDIQA-CORR-2024-MS-TrainingData.csv', columns, seed)

    # Get the medical error values.
    texts = list(data['Text'])
    labels = np.array(data['Error Flag'])
    return _split_training_and_validation_sets(texts, labels, validation_split)