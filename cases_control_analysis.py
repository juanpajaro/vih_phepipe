#!/usr/bin/env python3
import os
import utils_general_porpose
import utils_early_disease
import pandas as pd
import datetime

path = os.getcwd()
print(path)


#Funtion to load a dataframe using pandas
name_file = "/dataframes/dataset.csv"
def load_dataframe(path, name_file):
    data = pd.read_csv(path + name_file)
    return data

data = load_dataframe(path, name_file)

print(data.head())
print(data.columns)

def imprimir_fechas_consulta(secuencia_paciente):
    secuencia_paciente = eval(secuencia_paciente)
    print(type(secuencia_paciente))
    fecha_consulta = []
    for i in secuencia_paciente:        
        fecha_consulta.append(i.get("FechaConsulta"))

    return fecha_consulta

def view_cut_patient(data, id_patient, num_dias):
    sample_data = data.loc[data["IdCliente"] == id_patient]
    print(sample_data)    
    fecha_poli = sample_data["fecha_poli"].iloc[0]
    print("fecha_poli: {}".format(fecha_poli))
    print(type(fecha_poli))
    fecha_poli = datetime.datetime.strptime(fecha_poli, '%Y-%m-%d')
    print("fecha_poli: {}".format(fecha_poli))
    print(type(fecha_poli))
    #fecha_menos_seis_meses = utils_early_disease.calcular_fecha_antes_poli(fecha_poli, num_dias)
    fecha_menos_seis_meses = fecha_poli - datetime.timedelta(days = num_dias)
    secuencia_paciente = sample_data["dic_datos_consulta"].iloc[0]
    lista_consultas = imprimir_fechas_consulta(secuencia_paciente)
    secuencia_recortada = sample_data["secuencia_recortada"].iloc[0]
    lista_recorte = imprimir_fechas_consulta(secuencia_recortada)
       
    return fecha_poli, fecha_menos_seis_meses, lista_consultas, lista_recorte

id_patient = 1619
num_dias = 180
fecha_poli, fecha_menos_seis_meses, lista_consultas, lista_recorte = view_cut_patient(data, id_patient, num_dias)
print("id_patient: {}".format(id_patient))
print("diagnosis date {}".format(fecha_poli))
print("prediction window start: {}".format(fecha_menos_seis_meses))
#print(lista_consultas)
print("dates to include as the observation window:")
for i in lista_recorte:
    print(i)
print("all consulting dates of the patient")
for i in lista_consultas:
    print(i)
