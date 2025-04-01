#!/usr/bin/env python3
import utils_general_porpose
import os

path = os.getcwd()
print(path)

path_data = "/early_data/early_prediction_data3.json"
patiensts_data = utils_general_porpose.load_json(path, path_data)