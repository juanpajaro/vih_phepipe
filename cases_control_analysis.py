#!/usr/bin/env python3
import os
import utils_general_porpose
import utils_early_disease
import pandas as pd
import datetime
from pandas import Timestamp
import ast

path = os.getcwd()
print(path)


#Funtion to load a dataframe using pandas
name_file = "/dataframes/dataset.csv"
def load_dataframe(path, name_file):
    data = pd.read_csv(path + name_file)
    data["fecha_poli"] = pd.to_datetime(data["fecha_poli"], errors = "coerce")
    return data

data = load_dataframe(path, name_file)

print(data.head())
print(data.columns)
print(data["fecha_poli"].head())
print(type(data["fecha_poli"].head()))
print(data["fecha_poli"].iloc[0])
print(type(data["fecha_poli"].iloc[0]))
print(data.info())

print(data.info())

def imprimir_fechas_consulta(secuencia_paciente):
    print("llegue aqui")
    secuencia_paciente = ast.literal_eval(secuencia_paciente)
    print("siguiente paso")
    print(type(secuencia_paciente))
    print("tercer paso")
    print(secuencia_paciente[0])

    fecha_consulta = []
    for i in secuencia_paciente:        
        fecha_consulta.append(i.get("FechaConsulta"))

    return fecha_consulta

def view_cut_patient(data, id_patient, num_dias):
    sample_data = data.loc[data["IdCliente"] == id_patient]
    #print(sample_data)    
    fecha_poli = sample_data["fecha_poli"].iloc[0]
    print(fecha_poli)
  
    fecha_menos_seis_meses = utils_early_disease.calcular_fecha_antes_poli(fecha_poli, num_dias)
    fecha_menos_seis_meses = fecha_poli - datetime.timedelta(days = num_dias)
    secuencia_paciente = sample_data["dic_datos_consulta"].iloc[0]

    #print(secuencia_paciente)
    lista_consultas = imprimir_fechas_consulta(secuencia_paciente)
    secuencia_recortada = sample_data["secuencia_recortada"].iloc[0]
    #print(secuencia_recortada)
    lista_recorte = imprimir_fechas_consulta(secuencia_recortada)
       
    return fecha_poli, fecha_menos_seis_meses, lista_consultas, lista_recorte

print(data.loc[data["IdCliente"] == 13558])

#id_patient = 13558
id_patient = 1619
num_dias = 180
fecha_poli, fecha_menos_seis_meses, lista_consultas, lista_recorte = view_cut_patient(data, id_patient, num_dias)
print("id_patient: {}".format(id_patient))
print("diagnosis date {}".format(fecha_poli))
print("prediction window start: {}".format(fecha_menos_seis_meses))
#print(lista_consultas)
print("dates to include as the observation window:")
for i in lista_recorte:
    len(lista_recorte)
    print(i)
print("all consulting dates of the patient")
for i in lista_consultas:
    len(lista_consultas)
    print(i)

print("id_patient: {}, diagnosis date {}, prediction window start: {}, diag_predict {}, diag_total {}".format(id_patient, fecha_poli, fecha_menos_seis_meses, len(lista_recorte), len(lista_consultas)))

# Function to calculate the id_patient, diagnosis date, prediction window start, diag_predict and diag_total for all patients
def calculate_dates(data, num_dias):    
    # Iterate over all patients
    # Create a new dataframe to store the results
    new_data = pd.DataFrame(columns=["id_patient", "diagnosis date", "prediction window start", "diag_predict", "diag_total"])
    # Iterate over all patients
    for id_patient in data["IdCliente"]:
        fecha_poli, fecha_menos_seis_meses, lista_consultas, lista_recorte = view_cut_patient(data, id_patient, num_dias)
        # Append the new row to the dataframe
        print("id_patient: {}, diagnosis date {}, prediction window start: {}, diag_predict {}, diag_total {}".format(id_patient, fecha_poli, fecha_menos_seis_meses, len(lista_recorte), len(lista_consultas)))        
        new_row = {"id_patient": id_patient, "diagnosis date": fecha_poli, "prediction window start": fecha_menos_seis_meses, "diag_predict": len(lista_recorte), "diag_total": len(lista_consultas)}
        new_data = pd.concat([new_data, pd.DataFrame([new_row])], ignore_index=True)  

    return new_data
# Calculate the dates for all patients
num_dias = 180
new_data = calculate_dates(data, num_dias)
# Print the new dataframe
print(new_data.head())
print(new_data.columns)
# Save the dataframe to a csv file
data.to_csv(path + "/dataframes/dataset_with_dates.csv", index=False)

