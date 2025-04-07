#!/usr/bin/env python3
#from sklearn.model_selection import train_test_split
import utils_general_porpose
import numpy as np
import os
import random

def balance_dataset(datos, col_label, col_clinical_concepts):
    df_pos = datos[datos[col_label]==1]
    train_pos = df_pos[col_clinical_concepts].to_list()

    df_neg = datos[datos[col_label]==0]
    train_neg = df_neg[col_clinical_concepts].to_list()
    
    g = datos.groupby(col_label)
    df_balanced = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))

    X = df_balanced[col_clinical_concepts].to_list()
    y = df_balanced[col_label].to_numpy()

    return X, y

#Function to splir the data into train and test
def split_data(balanced_patients, train_size):

    positive = []
    negative = []
    
    X_train = []
    y_train = []
    id_train = []

    X_test = []
    y_test = []
    id_test = []

    _train_pos = []
    _train_neg = []
    
    train_pos = []
    train_neg = []

    _test_pos = []
    _test_neg = []

    test_pos = []
    test_neg = []

    train = []
    test = []

    len_data = len(balanced_patients)
    print("len data {}".format(len_data))

    _train_size = round(len_data * train_size)
    print("number train size {}".format(_train_size))

    _test_size = round(len_data - _train_size)
    print("number test size {}".format(_test_size))

    for patient in balanced_patients:
        if patient.get("label") == 1:
            _train_pos = {"id_cliente": patient.get("id_cliente"), "label": patient.get("label"), "seq": patient.get("seq")}
            positive.append(_train_pos)
        else:
            _train_neg = {"id_cliente": patient.get("id_cliente"), "label": patient.get("label"), "seq": patient.get("seq")}
            negative.append(_train_neg)
    
    print("len positive {}".format(len(positive)))
    print("len negative {}".format(len(negative)))

    train_pos = positive[:int(_train_size*0.5)]
    print("len train_pos {}".format(len(train_pos)))
    test_pos = positive[:int(_test_size*0.5)]
    print("len test_pos {}".format(len(test_pos)))

    train_neg = negative[:int(_train_size*0.5)]
    test_neg = negative[:int(_test_size*0.5)]

    train = train_pos + train_neg
    random.shuffle(train)
    test = test_pos + test_neg
    random.shuffle(test)
    print("train len: {}".format(len(train)))
    print("test len: {}".format(len(test)))


    """train = balance_dataset[:int(_train_size)]
    test = balance_dataset[int(_train_size):]
    print("train len: {}".format(len(train)))
    print("test len: {}".format(len(test)))"""

    """for data in train:
        X_train.append(data.get("seq"))
        y_train.append(data.get("label"))
        id_train.append(data.get("id_cliente"))
        dict_train = {"X": X_train, "y": y_train, "id_cliente": id_train}
    
    for data in test:
        X_test.append(test.get("seq"))
        y_test.append(test.get("y"))
        id_test.append(test.get("id_cliente"))
        dict_test = {"X": X_train, "y": y_train, "id_cliente": id_train}"""
    
    #y = np.array(y)

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    print("Data splitted into train and test")
    #return X_train, X_test, y_train, y_test
    return train, test

def balanced_subsample(patient_in):
    
    #x = []
    y = []
    id_cliente = []

    #x_pos = []
    #y_pos = []
    #x_neg = []
    #y_neg = []
    id_pos = []
    id_neg = []

    pos_patient_selected = []

    #xs_pos = []
    #xs_neg = []    
    #xs_pos_selected = []
    
    #two_classes = []
    balanced_patients = []
    

    for patient in patient_in:
        #x.append(patient.get("seq"))
        y.append(patient.get("label"))
        id_cliente.append(patient.get("id_cliente"))
    
    #y = np.array(y)

    for patient in patient_in:
        if patient.get("label") == 1:
            #x_pos.append(patient.get("seq"))
            #y_pos.append(patient.get("label"))
            id_pos.append(patient)
        else:
            #x_neg.append(patient.get("seq"))
            #y_neg.append(patient.get("label"))
            id_neg.append(patient.get("id_cliente"))
        
        #x.append(patient.get("seq"))
        #y.append(patient.get("label"))
    
    #xs_pos.append((y_pos, x_pos))
    #xs_neg.append((y_neg, x_neg))
    
    """print("type xs_pos {}".format(type(xs_pos)))
    print("len xs_pos {}".format(len(xs_pos)))
    print("type xs_neg {}".format(type(xs_neg)))
    print("len xs_neg {}".format(len(xs_neg)))"""
    #print(class_xs[0])
    
    count_y = np.unique(y, return_counts=True)
    print("count_label: {}".format(count_y))
    min_ci = min(count_y[1])
    print("min_class: {}".format(min_ci))

    random_index = random.sample(range(0, len(id_pos)), min_ci)
    #print("len radom_index {}".format(len(random_index)))

    #class_xs_selected = None
    """for y_pos, x_pos in xs_pos:
        y_selected = [y_pos[i] for i in random_index]
        x_selected = [x_pos[i] for i in random_index]
        id_selected = [id_pos[i] for i in random_index]"""
    
    for i in random_index:
        pos_patient_selected.append(id_pos[i])

    print("ten random index to select positive patients: {}".format(random_index[:10]))
    #pos_patient_selected = [patient for patient in patient_in if patient.get("id_cliente") == id_pos_selected]
    
    print("len positive patients: {}".format(len(pos_patient_selected)))
    
    #xs_pos_selected.append((y_selected, x_selected))

    neg_patient_selected = [patient for patient in patient_in if patient.get("label") == 0]
    print("len negative patients: {}".format(len(neg_patient_selected)))

    #y, x = y_selected + y_neg, x_selected + x_neg
    all_patientes = pos_patient_selected + neg_patient_selected
    #two_classes.append((y, x))

    """list_index = list(range(len(y)))
    #print("list_index {}".format(list_index))
    _index_shuffle = random.shuffle(list_index)
    #print("list_index_shuffle {}".format(list_index))"""

    list_index = list(range(len(all_patientes)))
    _index_shuffle = random.shuffle(list_index)

    """for y, x in two_classes:
        len(y)        
        #print("len list_index {}".format(len(list_index)))
        for i in list_index:
            y_selected, x_selected = y[i], x[i]        
            dict_result = {"y": y_selected, "X": x_selected}    
            balanced_patients.append(dict_result)"""
    
    for i in list_index:
        patient_selected = all_patientes[i]
        #dict_result = {"id_cliente": patient_selected.get("id_cliente"), "label": patient_selected.get("label"), "seq": patient_selected.get("seq")}
        dict_result = {"id_cliente": patient_selected.get("id_cliente"), "label": patient_selected.get("label"), "seq": patient_selected.get("entities")}
        balanced_patients.append(dict_result)

    print("ten random indexes in which the patient list is shuffled {}".format(list_index[:10]))
    #return min_ci, xs_pos, random_index, xs_pos_selected, two_classes, balanced_patients
    print("data balanced")
    return balanced_patients

#funcion to concatenate the sequences of diagnostic and clinical notes
def paste_sequences_patients(dict_consultas):
    one_patient = []
    for num_consulta, values in dict_consultas.items():
        #print("diagnosticos {}, nota clinica {}".format(values["diagnosticos"], values["texto"]))
        codes_diagnostic = (str(values["diagnosticos"])).replace(","," ")
        text_diagnostic = str(values["texto"])
        one_patient.append(codes_diagnostic + " " + text_diagnostic.lower())
    #sequence_npatient.append(codes_diagnostic + " " + text_diagnostic.lower())
    return one_patient

def send_patient_info(data_list):
    patients = []    
    for patient in data_list:
        id_cliente = patient.get("id_cliente")
        label = patient.get("label")
        consults_patient = patient.get("consultas")
        sequence_patient = paste_sequences_patients(consults_patient)
        #print(sequence_patient[0])
        #print(sequence_patient[1])
        #print(sequence_patient[2])
        #print(sequence_patient[3])
        #print(sequence_patient[4])
        #print(len(sequence_patient))
        str_sequence = " ".join(sequence_patient)        
        patients.append({"id_cliente": id_cliente, "label": label, "seq":str_sequence})
    print("patients info convert into sequences")
    return patients

#function to create a list of patients with a maximum length caracter in the sequence
def make_patient_list_with_maxlength(patients_list, max_length):
    patients_list_max = []
    for patient in patients_list:
        if len(patient.get("seq")) < max_length:
            patients_list_max.append(patient)
    print("patients list with max length created")
    return patients_list_max

#function append train and test data
def append_train_test_data(X_train, X_test, y_train, y_test):
    patient_train = []
    patient_test = []
    for i in range(len(y_train)):
        train = {"X": X_train[i], "y": y_train[i]}
        patient_train.append(train)
    
    for i in range(len(y_test)):
        test = {"X": X_test[i], "y": y_test[i]}        
        patient_test.append(test)

    print("train and test data appended")
    return patient_train, patient_test

def save_split_proccess(patient_train, patient_test, current_path):
    _list_path_save = ["/datasets/train/", "/datasets/test/"]
    _list_data = [patient_train, patient_test]
    _name_data = ["train", "test"]
    for i in range(len(_list_path_save)):
        path_save = current_path + _list_path_save[i-1]
        path_save = utils_general_porpose.create_directory(path_save)
        if os.path.exists(path_save):
            #if some version of dataset exists, then get the name list of the dataset
            list_dataframes = utils_general_porpose.extract_name_model(path_save, ".json")
            #extract the last version of the dataset
            last_version = utils_general_porpose.extract_last_version_model(list_dataframes)
            #get the number of the new version
            version = str(utils_general_porpose.counter_version(last_version))
            #save the dataset
            path_version = path_save + _name_data[i-1] + ".json"
            #data.to_csv(path_version, index = False)
            utils_general_porpose.save_json(_list_data[i-1], path_version)

    return print("split process finished")
