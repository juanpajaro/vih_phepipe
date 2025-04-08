#!/usr/bin/env python3
import os
import utils_general_porpose
import utils_split_dataset
import sys

#Load data
def load_data(path, filename):
    data = utils_general_porpose.load_json(path, filename)
    print("Data loaded")
    return data

def balanced_subsample(data):
    balanced_data = utils_split_dataset.balanced_subsample(data)
    print("Balanced data created")
    return balanced_data

def split_data(data, train_size):
    train_data, test_data = utils_split_dataset.split_data(data, train_size)
    print("Data split into train and test sets")
    return train_data, test_data

def save_data(data, directory, filename_prefix, timestamp):
    json_path = os.path.join(directory, f"{filename_prefix}_{timestamp}.json")
    utils_general_porpose.save_json(data, json_path)
    print("Data saved")

def main(path, filename, train_size, timestamp):
    data = load_data(path, filename)
    balanced_data = balanced_subsample(data)
    train_data, test_data = split_data(balanced_data, train_size)
    save_data(train_data, "./train", "train", timestamp)
    save_data(test_data, "./test", "test", timestamp)
    print("Split Data pipeline completed")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python split_data_pipeline.py <path> <filename>")
        sys.exit(1)
    
    path = sys.argv[1]    
    filename = sys.argv[2]
    train_size = sys.argv[3]
    timestamp = sys.argv[4]

    #path = "/home/pajaro/compu_Pipe_V3/"
    #filename = "concepts/concepts_${CURREN_DATE}.json/"
    #train_size = 0.8
    #timestamp = "2023-10-01_12-00-00"


    main(path, filename, train_size, timestamp)
