#!/usr/bin/env python3
import time
import os
import pandas as pd
from datetime import datetime
import utils_early_disease
import utils_general_porpose

def get_timestamp():
    """Genera un timestamp con formato de fecha y hora."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

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

def process_ehr_data(path, filename, data, num_dias):
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
    
    data["secuencia_recortada"] = data.apply(lambda x: utils_early_disease.recortar_historia(x["dic_datos_consulta"], x["fecha_poli"], num_dias), axis=1)
    print(f"Patients records cut: {num_dias} days")
    
    data["vacios_poli"] = data["secuencia_recortada"].apply(lambda x: "not empty" if len(x) > 0 else "empty")
    data = data[data["vacios_poli"] == "not empty"]
    print("Patients without EHR data before the prediction window removed")
    
    data = data[data["label_apnea"] < 3]
    print("Patients with label 3 removed")
    
    return data

def save_dataset(data, directory, filename_prefix):
    """Guarda el dataset en formato CSV y JSON con versionado por fecha y hora."""
    timestamp = get_timestamp()
    directory = utils_general_porpose.create_directory(directory)
    
    csv_path = os.path.join(directory, f"{filename_prefix}_{timestamp}.csv")
    data.to_csv(csv_path, index=False)
    print(f"Dataset saved in CSV format: {csv_path}")
    
    json_path = os.path.join(directory, f"{filename_prefix}_{timestamp}.json")
    utils_general_porpose.save_json(data, json_path)
    print(f"Dataset saved in JSON format: {json_path}")

def main(path_data, name_poli_data, name_sleepS_data, name_idcc, name_ehr_data, num_dias):
    """Pipeline principal para transformar los datos."""
    data_poli = load_and_process_polisomnography_data(path_data, name_poli_data)
    data_sleep = load_and_process_sleep_study_data(path_data, name_sleepS_data)
    data = merge_and_clean_data(data_poli, data_sleep)
    data = load_and_merge_idcc_data(path_data, name_idcc, data)
    data = process_ehr_data(path_data, name_ehr_data, data, num_dias)
    save_dataset(data, "./dataframes", "dataset")

# Ejecución del pipeline
if __name__ == "__main__":
    path_data = "/zine/data/salud/computational_pipe_v2/raw_data/"
    name_poli_data = "fecha_cedula_clinica_suenio_may 31 2023.csv"
    name_sleepS_data = "base principal ajustada 11mayo2021.csv"
    name_idcc = "3636_idClientes.csv"
    name_ehr_data = "Vista_Minable_3636.csv"
    num_dias = 180
    
    main(path_data, name_poli_data, name_sleepS_data, name_idcc, name_ehr_data, num_dias)