#!/usr/bin/env python3
#import utils_clinical_concept_extraction
import utils_general_porpose
import utils_split_dataset
import numpy as np

class DatasetSplit:
    def __new__(cls, *args, **kwargs):
        print("DatasetSplit object created")
        return super().__new__(cls)
    
    def __init__(self, path_data):
        self.path_data = path_data        
        #self.current_path = None
        #self.data = None
        #self.patients = None
        #self.patients_maxLength = None
        #self.balanced_patients = None
        #self.X_train = None
        #self.X_test = None
        #self.y_train = None
        #self.y_test = None
        self.patient_train = None
        self.patient_test = None
        print("DatasetSplit object instantiated")

    def __repr__(self):
        return "filter_name: {}, path_data: {}".format(type(self).__name__, self.path_data)

#patient_train = []
#patient_test = []
    
    def run_pipe_ds(self):

        current_path = utils_general_porpose.get_current_path()
        data = utils_general_porpose.load_json(current_path, self.path_data)
        patients = utils_split_dataset.send_patient_info(data)
        patients_maxLength = utils_split_dataset.make_patient_list_with_maxlength(patients, 1000000)
        """print(len(patients_maxLength))
        print(patients_maxLength[0].keys())"""
        balanced_patients = utils_split_dataset.balanced_subsample(patients_maxLength)
        """print(len(balanced_patients))
        print(balanced_patients[0].keys())"""

        """for patient in balanced_patients[:3]:
            print(patient.get("y"))
            print(len(patient.get("X")))
            print("----")"""

        self.patient_train, self.patient_test = utils_split_dataset.split_data(balanced_patients, 0.8)

        """print(len(X_train))
        print(len(X_test))
        print(len(y_train))
        print(len(y_test))"""

        #count_y_train = np.unique(y_train, return_counts=True)
        #print(count_y_train)
        #count_y_test = np.unique(y_test, return_counts=True)
        #print(count_y_test)

        #self.patient_train, self.patient_test = utils_split_dataset.append_train_test_data(X_train, X_test, y_train, y_test)
        """print(len(patient_train))
        print(len(patient_test))"""

        utils_split_dataset.save_split_proccess(self.patient_train, self.patient_test, current_path)
        return None

#python3
#from utils_class_split import DatasetSplit

"""
split = DatasetSplit("/early_data/early_prediction_data1.json")
split
split.run_pipe_ds()
"""