#!/usr/bin/env python3
import sys
import os
import time
import pandas as pd
from datetime import datetime
import utils_early_disease
import utils_general_porpose


def get_timestamp():
    """Genera un timestamp con formato de fecha y hora."""
    #return datetime.now().strftime("%Y%m%d_%H%M%S")

def load_and_process_polisomnography_data(path, filename):
    """Carga y procesa los datos de polisomnografía."""
    data = utils_early_disease.cargar_fecha_poli(os.path.join(path, filename))
    print("Polisomnography date data loaded")
    
    data = utils_early_disease.ordenar_datos_por_fecha(data, "fecha_poli")
    print("Polisomnography date data sorted by date")
    
    data = utils_early_disease.eliminar_duplicados(data, "cc")
    print("Duplicates removed from polisomnography date data")
    
    return data

def load_and_process_sleep_study_data(path, filename):
    """Carga y procesa los datos del estudio del sueño."""
    data = utils_early_disease.load_sleep_study_data(path, filename)
    print("Sleep study data loaded")
    
    lista_de_texto, lista_de_digitos = utils_early_disease.saber_si_digito_texto(data["o2 / CPAP"], valor_incluido="1")
    lista_valores = lista_de_digitos + lista_de_texto
    data = utils_early_disease.excluir_incluir_pacientes(data, "o2 / CPAP", lista_valores)
    print("Sleep study data cleansed, only patients without titration studies are included")
    
    data = data[data["excluir_incluir"] == "incluir"]
    data = utils_early_disease.eliminar_duplicados(data, "cc")
    print("Duplicates removed from sleep study data")
    
    return data

def merge_and_clean_data(data_poli, data_sleep):
    """Une y limpia los datos de polisomnografía y estudio del sueño."""
    data = utils_early_disease.mergue_datasets([data_sleep, data_poli], "cc")
    print("Polisomnography and sleep study data merged")
    
    data = utils_early_disease.eliminar_datos_faltantes_col(data, "fecha_poli")
    print("Patients without polisomnography date removed")
    
    data = utils_early_disease.agregar_variable_objetivo(data, "IAH")
    print("Label added for cases (IAH > 5) and controls (IAH <= 5)")
    
    data = data[["cc", "IAH", "o2 / CPAP", "fecha_poli", "label_apnea"]]
    print("Columns relevant for early prediction selected")
    
    return data

def load_and_merge_idcc_data(path, filename, data):
    """Carga y une los datos de ID y CC con el dataset principal."""
    data_idcc = utils_early_disease.load_idcc_data(os.path.join(path, filename))
    print("ID and CC dataset loaded")
    
    data = utils_early_disease.mergue_datasets([data_idcc, data], "cc")
    print("ID and CC dataset merged with the main dataset")
    
    data = utils_early_disease.eliminar_datos_faltantes_col(data, "label_apnea")
    print("Patients without label removed")
    
    data = data.rename({"idCliente": "IdCliente"}, axis="columns")
    print("IdCliente column renamed")
    
    return data

def process_ehr_data(path, filename, data, days_pw, days_ow):
    """Procesa los datos de EHR y los une con el dataset principal."""
    start_time = time.time()
    data_ehr = utils_early_disease.load_ehr_data(path, filename)
    print(f"EHR data loaded in {(time.time() - start_time) / 60:.2f} minutes")
    
    data_ehr = utils_early_disease.convertir_fecha_nacimiento(data_ehr, "FecNacimiento")
    print("FecNacimiento transformed into datetime type")
    
    for col in ["FecIngreso", "FechaConsulta"]:
        data_ehr = utils_early_disease.convertir_datos_fecha(data_ehr, col)
    print("FecIngreso and FechaConsulta transformed into datetime type")
    
    datos_unicos_ehr = data_ehr[["IdCliente", "FecNacimiento", "Sexo"]]
    datos_unicos_ehr_pacientes = utils_early_disease.agrupar_pacientes_por_fechaNacimiento_sexo(datos_unicos_ehr, "IdCliente", "FecNacimiento", "Sexo")
    
    datos_sec_con = data_ehr[["IdCliente", "IdConsulta", "FechaConsulta", "Diagnosticos_Consulta", "DesPlanYConcepto"]]
    datos_sec_consulta_pacientes = utils_early_disease.make_sequences_ehr(datos_sec_con, "IdCliente", ["FechaConsulta", "Diagnosticos_Consulta", "DesPlanYConcepto"], "dic_datos_consulta")
    print("EHR data per patient joined")
    
    datos_ehr_pacientes = utils_early_disease.mergue_datasets([datos_unicos_ehr_pacientes, datos_sec_consulta_pacientes], "IdCliente")
    data = utils_early_disease.mergue_datasets([datos_ehr_pacientes, data], "IdCliente")
    print("EHR data merged with the main dataset")
    
    data = utils_early_disease.eliminar_datos_faltantes_col(data, "label_apnea")
    print("Patients without label removed")
    
    data["edad_poli"] = data.apply(lambda x: utils_early_disease.edad_dia_poli(x["FecNacimiento"], x["fecha_poli"]).years * -1, axis=1)
    print("Date of birth transformed into age according to the date of the polisomnography")
    
    data["secuencia_recortada"] = data.apply(lambda x: utils_early_disease.recortar_historia(x["dic_datos_consulta"], x["fecha_poli"], days_pw, days_ow), axis=1)
    print(f"Patients records cut: {days_pw} days, which is when prediction window starts")
    print(f"Patients records cut: {days_ow} days, which is when observation window ends")
    
    data["vacios_poli"] = data["secuencia_recortada"].apply(lambda x: "not empty" if len(x) > 0 else "empty")
    data = data[data["vacios_poli"] == "not empty"]
    print("Patients without EHR data before the prediction window removed")
    
    data = data[data["label_apnea"] < 3]
    print("Patients with label 3 removed")

    new_data = utils_early_disease.calculate_info_dates(data, days_pw)
    data = utils_early_disease.mergue_datasets([data, new_data], "IdCliente")
    #data = data[data["num_app_included"] > 1]
    print("prediction_window_start, number of appointments to included, total appointments were added to the dataset")

    #TODO: es importante hacer una comparación entre las variables que se usan normalmente en Fenotipado computacional y las que se cargan o usan para el modelo
    
    return data, data_ehr

def save_dataset(data, directory, filename_prefix, timestamp):
    """Guarda el dataset en formato CSV y JSON con versionado por fecha y hora."""
    #timestamp = get_timestamp()
    directory = utils_general_porpose.create_directory(directory)
    
    csv_path = os.path.join(directory, f"{filename_prefix}_{timestamp}.csv")
    data_save = data[["IdCliente", "IAH", "Sexo", "Cantidad_Atenciones", "fecha_poli", "label_apnea", "edad_poli", "last_appointment","prediction_window_start", "end_observation_window", "num_app_included", "total_app", "lista_consultas", "lista_recorte"]]
    data_save.to_csv(csv_path, index=False)
    print(f"Dataset saved in CSV format: {csv_path}")

    #convert dataset as list of dictionary to save in json file
    
    data  = utils_early_disease.make_listDictionary_patients(data)
    print("data transformed into list of dictionaries")
    
    json_path = os.path.join(directory, f"{filename_prefix}_{timestamp}.json")
    utils_general_porpose.save_json(data, json_path)
    print(f"Dataset saved in JSON format: {json_path}")

def main(path_data, name_poli_data, name_sleepS_data, name_idcc, name_ehr_data, days_pw, days_ow):
    """Pipeline principal para transformar los datos."""
    data_poli = load_and_process_polisomnography_data(path_data, name_poli_data)
    data_sleep = load_and_process_sleep_study_data(path_data, name_sleepS_data)
    data = merge_and_clean_data(data_poli, data_sleep)
    data = load_and_merge_idcc_data(path_data, name_idcc, data)
    data, data_ehr = process_ehr_data(path_data, name_ehr_data, data, days_pw, days_ow)
    save_dataset(data, "./cases_controls", "cases_controls", timestamp)
    print("Cases_controls step finished successfully")

# Ejecución del pipeline
if __name__ == "__main__":
    if len(sys.argv) != 9:
        print("Usage: python data_transformation_pipeline.py <path_data> <name_poli_data> <name_sleepS_data> <name_idcc> <name_ehr_data> <days_pw> <days_ow>")
        sys.exit(1)

    path_data = sys.argv[1]
    name_poli_data = sys.argv[2]
    name_sleepS_data = sys.argv[3]
    name_idcc = sys.argv[4]
    name_ehr_data = sys.argv[5]
    days_pw = int(sys.argv[6])
    days_ow = int(sys.argv[7])
    timestamp = sys.argv[8]
        
    #path_data = "./raw_data/"
    #name_poli_data = "fecha_cedula_clinica_suenio_may 31 2023.csv"
    #name_sleepS_data = "base principal ajustada 11mayo2021.csv"
    #name_idcc = "3636_idClientes.csv"
    #name_ehr_data = "Vista_Minable_3636.csv"
    #days_pw = 180
    #days_ow = 730
    
    main(path_data, name_poli_data, name_sleepS_data, name_idcc, name_ehr_data, days_pw, days_ow)