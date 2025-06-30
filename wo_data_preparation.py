#!/usr/bin/env python3
import os
import pandas as pd
import utils_early_disease
import utils_general_porpose
import sys

#Funcion para cargar los datos
def load_data(data_path, data_name):
    # Load the data
    data_path = os.path.join(data_path, data_name)    
    return pd.read_csv(data_path, sep = "|", encoding = "latin1")

#funcion para agregar la variable objetivo
def agregar_variable_objetivo(datos, columna):
    #Condición para agregar el label    
    for index, value in datos[columna].items():
        if value == "NO":
            datos.loc[index, "label"] = 0
        elif value == "SI":
            datos.loc[index, "label"] = 1
        else:
            datos.loc[index, "label"] = 3

    return datos

def observar_seq_recortada(data):
    print(type(data["secuencia_recortada"][0]))
    print("nconsultas_recortada {} paciente {}".format(len(data["secuencia_recortada"][0]), data["NumeroContrato"][0]))
    print("nconsultas_total {} paciente {}".format(len(data["seq_diag"][0]), data["NumeroContrato"][0]))
    print("nconsultas_recortada {} paciente {}".format(len(data["secuencia_recortada"][1]), data["NumeroContrato"][1]))
    print("nconsultas_total {} paciente {}".format(len(data["seq_diag"][1]), data["NumeroContrato"][1]))
    print("nconsultas_recortada {} paciente {}".format(len(data["secuencia_recortada"][2]), data["NumeroContrato"][2]))
    print("nconsultas_total {} paciente {}".format(len(data["seq_diag"][2]), data["NumeroContrato"][2]))
    return None

def observar_seq_label(datos, name_paciente, name_fecha, name_label, name_seq):
    for key, value in datos[:10].iterrows():
        print("----------------------------------")        
        print("pacient: {}, fecha_dx: {}, label: {}".format(value[name_paciente], value[name_fecha], value[name_label]))
        for i in value[name_seq]:
            print("diag: {}, concept: {}, fecha: {}".format(i.get("Dx"), i.get("Análisis_y_Plan_de_Manejo"), i.get("FechaConsulta")))
        #print("paciente {}".format(datos[name_paciente]))    
    
    return None


def observar_seq_label_after_cut(datos, name_paciente, name_fecha, name_label, name_seq):
    for key, value in datos[:10].iterrows():
        print("----------------------------------")        
        print("pacient: {}, fecha_dx: {}, label: {}".format(value[name_paciente], value[name_fecha], value[name_label]))
        for i in value[name_seq]:
            print("diag: {}, concept: {}, fecha: {}".format(i.get("Diagnosticos_Consulta"), i.get("DesPlanYConcepto"), i.get("FechaConsulta")))
        #print("paciente {}".format(datos[name_paciente]))    
    
    return None

def save_dataset(data, directory, filename_prefix, timestamp):
    """Guarda el dataset en formato CSV y JSON con versionado por fecha y hora."""
    #timestamp = get_timestamp()
    directory = utils_general_porpose.create_directory(directory)
    
    csv_path = os.path.join(directory, f"{filename_prefix}_{timestamp}.csv")
    # Drop unnecessary columns for CSV saving
    data_csv = data.drop(columns=["secuencia_recortada", "seq_diag"])
    data_save = data_csv
    data_save.to_csv(csv_path, index=False)
    print(f"Dataset saved in CSV format: {csv_path}")

    #convert dataset as list of dictionary to save in json file
    
    data  = utils_early_disease.make_listDictionary_patients(data)
    print("data transformed into list of dictionaries")
    
    json_path = os.path.join(directory, f"{filename_prefix}_{timestamp}.json")
    utils_general_porpose.save_json(data, json_path)
    print(f"Dataset saved in JSON format: {json_path}")


def main(days_pw, days_ow, timestamp, data_path, data_name, label_name):
    
    
    # Load the data
    datos = load_data(data_path, data_name)
    # Display the first few rows of the dataframe
    #print(data.info())

    col_datos = list(datos.columns)
    for col in col_datos:
        print(col)

    #Se recortan los datos de HCE
    diagnosticos = datos[["NumeroContrato","FechaConsulta","Dx", "Análisis_y_Plan_de_Manejo"]]
    diagnosticos["FechaConsulta"] = pd.to_datetime(datos["FechaConsulta"])
    #print(diagnosticos.info())

    #Se carga el label de los pacientes
    label_data = load_data(data_path, label_name)
    label_data = label_data.rename(columns={"ï»¿ID_Contrato": "NumeroContrato"})
    label_data["Fecha_Dx_Prueba"] = pd.to_datetime(label_data["Fecha_Dx_Prueba"])

    #Group EHR data per patient as a sequence    
    lista_columnas = ['FechaConsulta', 'Dx', "Análisis_y_Plan_de_Manejo"]
    columna_id = "NumeroContrato"
    nombre_columna_final = "seq_diag"    
    datos_sec_consulta_pacientes = utils_early_disease.make_sequences_ehr(diagnosticos, columna_id, lista_columnas, nombre_columna_final)
    print("EHR data per patient joined")
    print("Only the follow variables are selected for this version of the clinical phenotyping pipeline in VIH: {}".format(lista_columnas))
    #print(datos_sec_consulta_pacientes.head())
    #print(datos_sec_consulta_pacientes.columns)
    #print(datos_sec_consulta_pacientes.shape)
    #print(datos_sec_consulta_pacientes["seq_diag"][0])

    lista_dataframes = [label_data, datos_sec_consulta_pacientes]
    datos_ehr_pacientes = utils_early_disease.mergue_datasets(lista_dataframes, "NumeroContrato")
    #print(datos_ehr_pacientes.info())
    #Existe una diferencia de 21 pacientes entre las bases de datos
    datos_ehr_pacientes.dropna(subset = ["seq_diag"], inplace=True)
    print(datos_ehr_pacientes.shape)

    datos_ehr_pacientes = agregar_variable_objetivo(datos_ehr_pacientes, "Pte_VIH")
    print(datos_ehr_pacientes.info())

    data = datos_ehr_pacientes    

    #observar_seq_label(data, "NumeroContrato", "Fecha_Dx_Prueba", "label", "seq_diag")    

    data["secuencia_recortada"] = data.apply(lambda x: utils_early_disease.recortar_historia(x["seq_diag"], x["Fecha_Dx_Prueba"], days_pw, days_ow), axis=1)
    print(f"Patients records cut: {days_pw} days, which is when prediction window starts")
    print(f"Patients records cut: {days_ow} days, which is when observation window ends")
    #print("The following columns are in the dataset: {}".format(data.columns))
    #print(data.info())
    #print(data["secuencia_recortada"][:10])

    #observar_seq_label_after_cut(data, "NumeroContrato", "Fecha_Dx_Prueba", "label", "secuencia_recortada")
    

    data["vacios"] = data["secuencia_recortada"].apply(lambda x: "not empty" if len(x) > 0 else "empty")
    data = data[data["vacios"] == "not empty"]
    #print("Patients without EHR data before the prediction window removed")
    #print(data.info())
    
    data = data[data["label"] < 3]
    print("Patients with label 3 removed")
    #print(data.info())

    data = data.rename(columns={"NumeroContrato": "IdCliente"})

    new_data = utils_early_disease.calculate_info_dates(data, days_pw)
    data = utils_early_disease.mergue_datasets([data, new_data], "IdCliente")
    #data = data[data["num_app_included"] > 1]
    print("prediction_window_start, number of appointments to included, total appointments were added to the dataset")
    print(data.info())

    save_dataset(data, "./data_transformation", "data_t", timestamp)
    print("Dataset transformation step finished successfully")
       
if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: python data_transformation_pipeline.py <days_pw> <days_ow> <timestamp> <path_data> <data_name> <label_data>")
        sys.exit(1)

    # Define the prediction and observation windows in days
    #days_pw = 180  # Prediction window
    #days_ow = 730  # Observation window
    #timestamp = "20250520_053753"
    # Define the path and file name
    #data_path = "base_datos"
    #data_name = "Variables_HC.txt"
    #label_name='Etiqueta.txt'

    # Define the prediction and observation windows in days
    days_pw = int(sys.argv[1])  # Prediction window
    days_ow = int(sys.argv[2])  # Observation window
    timestamp = sys.argv[3]  # Timestamp
    # Define the path and file name
    data_path = sys.argv[4]
    data_name = sys.argv[5]  # Data file name
    label_name= sys.argv[6]  # Label file name


    main(days_pw, days_ow, timestamp, data_path, data_name, label_name)