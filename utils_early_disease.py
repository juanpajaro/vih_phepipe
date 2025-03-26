#!/usr/bin/env python3
import pandas as pd
import numpy as np
from collections import OrderedDict
import functools as ft
import re
import datetime
from dateutil import relativedelta

#Funcion para calcular la fecha antes de la polisomnografia. Recibe un parametro de días para calcular 
#por cada registro
def calcular_fecha_antes_poli(fecha_poli, num_dias):
    fecha_poli = datetime.datetime.strptime(str(fecha_poli), '%Y-%m-%d %H:%M:%S')
    fecha_menos_seis_meses = fecha_poli - datetime.timedelta(days = num_dias)
    return fecha_menos_seis_meses

#Funcion para recortar una lista de fecha seis meses atras de la fecha de entrada
def recortar_historia(secuencia_paciente, fecha_poli, num_dias):
    secuencia_recortada = {}
    lista_secuencia = []    
    fecha_antes_poli = calcular_fecha_antes_poli(fecha_poli, num_dias)
    #for index, value in secuencia_paciente.items():
    for i in secuencia_paciente:
        fecha = i.get("FechaConsulta")
        if fecha < fecha_antes_poli:
            diagnosticos = i.get("Diagnosticos_Consulta")
            texto = i.get("DesPlanYConcepto")
            fecha_menor = i.get("FechaConsulta")
            secuencia_recortada = {"FechaConsulta":fecha_menor, "Diagnosticos_Consulta":diagnosticos, "DesPlanYConcepto":texto}                
            lista_secuencia.append(secuencia_recortada)
    return lista_secuencia

def agrupar_pacientes_por_fechaNacimiento_sexo(datos, idPaciente, fechaNacimiento, sexo):
    datos = datos.groupby([idPaciente, fechaNacimiento, sexo]).size().reset_index(name='count')
    return datos

"""
def convertir_datos_dicc_fecha(datos, columna_key, listado_columnas):
    dicc_fechas = {}
    for index, row in datos.iterrows():
        dicc_fechas[row[columna_key]] = {}
        for columna in row.index:
            if columna in listado_columnas:
                dicc_fechas[row[columna_key]].update({columna:row[columna]})
    return dicc_fechas    
"""

def agrupar_pacientes_por_datos_atention(datos, columna_id, columna_key, lisdado_columnas, nombre_columna):
    grouped = datos.groupby(columna_id).apply(lambda x: {row[columna_key]:{columna:row[columna] for columna in lisdado_columnas} for _, row in x.iterrows()})
    datos = grouped.reset_index().rename(columns={"index": "index_col", 0: nombre_columna})
    return datos

def make_sequences_ehr(datos, columna_id, lisdado_columnas, nombre_columna):
    grouped = datos.groupby(columna_id).apply(lambda x: [{columna:row[columna] for columna in lisdado_columnas} for _, row in x.iterrows()])
    datos = grouped.reset_index().rename(columns={"index": "index_col", 0: nombre_columna})
    return datos
    #return grouped


"""
def agrupar_pacientes_por_datos_atencion(datos):
    grouped = datos.groupby('IdCliente').apply(lambda x: {row["FecIngreso"]:{"Temperatura":row['Temperatura'],
                                                           "Frecuencia_Cardiaca":row["Frecuencia_Cardiaca"],
                                                           'Frecuencia_Respiratoria':row['Frecuencia_Respiratoria'], 
                                                           'Presion_Sistolica':row['Presion_Sistolica'],
                                                           'Presion_Diastolica':row['Presion_Diastolica'], 
                                                           'Saturacion':row['Saturacion'], 
                                                           'Diagnosticos_Ingreso':row['Diagnosticos_Ingreso'], 
                                                           'ValIMC':row['ValIMC'],
                                                           'ValSuperficieCorporal':row['ValSuperficieCorporal']} for _, row in x.iterrows()})
    datos = grouped.reset_index().rename(columns={"index": "index_col", 0: "dic_datos_atencion"})
    return datos
"""

def sort_dict_by_date(the_dict, date_format):
    # Python dicts do not hold their ordering so we need to make it an
    # ordered dict, after sorting.
    return OrderedDict(sorted(the_dict.items(), key=lambda x: (x[0], date_format)))


"""
['IdCliente', 'IdConsulta', 'FechaConsulta', 'Talla_Consulta',
       'Peso_Consulta', 'Temperatura_Consulta', 'FC_Consulta', 'FR_Consulta',
       'PS_Consulta', 'PD_Consulta', 'Saturacion_Consulta',
       'DesEnfermedadActual', 'DesExamenFisico', 'DesMotivoCon', 'DesPlanTrat',
       'DesPlanYConcepto', 'DesValoracionSistemas', 'Diagnosticos_Consulta',
       'Medicamentos_Aplicados', 'Medicamentos_Formulados', 'Procedimientos',
       'NotasAtencion']

"""

"""
def agrupar_pacientes_por_datos_consulta(datos):
    grouped = datos.groupby('IdCliente').apply(lambda x: {row["FechaConsulta"]:{"Talla_Consulta":row['Talla_Consulta'],
                                                           "Peso_Consulta":row["Peso_Consulta"],
                                                           'Temperatura_Consulta':row['Temperatura_Consulta'],                                                            
                                                           'FR_Consulta':row['FR_Consulta'], 
                                                           'PS_Consulta':row['PS_Consulta'], 
                                                           'PD_Consulta':row['PD_Consulta'],
                                                           'Saturacion_Consulta':row['Saturacion_Consulta'],
                                                           'DesEnfermedadActual':row['DesEnfermedadActual'],
                                                           'DesExamenFisico':row['DesExamenFisico'],
                                                           'DesMotivoCon':row['DesMotivoCon'],
                                                           'DesPlanTrat':row['DesPlanTrat'],
                                                           'DesPlanYConcepto':row['DesPlanYConcepto'],
                                                           'DesValoracionSistemas':row['DesValoracionSistemas'],
                                                           'Diagnosticos_Consulta':row['Diagnosticos_Consulta'],
                                                           'Medicamentos_Aplicados':row['Medicamentos_Aplicados'],
                                                           'Medicamentos_Formulados':row['Medicamentos_Formulados'],
                                                           'Procedimientos':row['Procedimientos'],
                                                           'NotasAtencion':row['NotasAtencion']} for _, row in x.iterrows()})
    datos = grouped.reset_index().rename(columns={"index": "index_col", 0: "dic_datos_consulta"})
    return datos
"""

def mergue_datasets(lista_dataframes, columnName):
    dfs = lista_dataframes    
    df_final = ft.reduce(lambda left, right: pd.merge(left, right, how ="left", on=columnName), dfs)
    return df_final

#Funcion para agregar la variable objetivo a los datos de la clinica del sueño
def agregar_variable_objetivo(datos, columna):
    #Condición para agregar el label
    #si el indice de apnea (IAH) es por debajo de 5 (sin apnea)
    #si el indice de apnea (IAH) es superior a 5 y su CPAP no tiene datos se coloca 1 (con apnea)
    #aquellos que no tengan valor en el indice de apnea (IAH) se coloca 3 (sin definir)
    for index, value in datos[columna].items():
        if value <= 5:
            datos.loc[index, "label_apnea"] = 0
        elif value > 5:
            datos.loc[index, "label_apnea"] = 1
        else:
            datos.loc[index, "label_apnea"] = 3

    return datos


#se eliminan los registros nulos desde la fecha de la polisomnografia
def eliminar_datos_faltantes_col(vista_minable, columna_con_datos_faltantes):
    vista_minable.dropna(subset=[columna_con_datos_faltantes], inplace= True)
    return vista_minable



#Prueba de la función anterior
"""lista_prueba_paciente = [["2012-09-10 12:15:21", "code_X, cod_30, cod_50, cod_70"], 
                            ["2013-06-09 12:00:54", "code_Y, cod_R, cod_20"], 
                            ["2014-09-04 15:00:08", "code_5, cod_4"], 
                            ["2019-03-04 15:00:08", "code_23, cod_45"]]
fecha_poli = str(vista_minable["fecha_poli"][0])
num_dias = 180
print(recortar_seis_meses_historia(lista_prueba_paciente, fecha_poli, num_dias))"""

#Función para definir las palabras o numeros que identifican a pacientes en la columna "o2/cpap" para que puedan ser marcado como excluidos
#esta función puede modificarse si se desea agregar más valores que definan estudios basales
def saber_si_digito_texto(columna, valor_incluido = None):    
    lista_de_texto = []
    lista_de_digitos = []
    for value in columna:
        if pd.isna(value) == False:
        #if value is not None:
            value = str(value)
            if value.isdigit():
                #value = int(value)                
                if value not in lista_de_digitos and value != valor_incluido:
                    lista_de_digitos.append(value)        
            elif value not in lista_de_texto:
                lista_de_texto.append(value)
    return lista_de_texto, lista_de_digitos

"""    
lista_prueba = ["ecto2", 15, 9, "prueba", None]
lista_de_texto, lista_de_digitos = saber_si_digito_texto(lista_prueba)
print(lista_de_digitos)
print(lista_de_texto)
"""

def excluir_incluir_pacientes(datos, columna, lista_valores):
    for index, value in datos[columna].items():
        if value in lista_valores:
            datos.loc[index, "excluir_incluir"] = "excluir"
        else:
            datos.loc[index, "excluir_incluir"] = "incluir"    
    return datos

#Función para eliminar registros duplicados
#Revisar con la clinical del sueño que registros pueden ser eliminados y cuales no para realizar la asignación automatica
def eliminar_duplicados(datos, columna):
    datos = datos.drop_duplicates(subset= columna, keep="first")
    return datos

#Función que permite ordenar los datos por fecha
def ordenar_datos_por_fecha(datos, columna_fecha):
    datos = datos.sort_values(by=[columna_fecha])
    return datos

#Función para determinar el numero de duplicados en una columna
def contar_duplicados_columna(datos, columna):
    boolean = datos[columna].duplicated()
    return boolean.value_counts()

#Función para contar pacientes, consultas y diagnosticos
def contar_pacientes_consultas(datos, columna_pacientes, columna_consultas):
    num_pacientes = len(datos[columna_pacientes].unique())
    num_consultas = len(datos[columna_consultas].unique())    
    media_consultas_pacientes = num_consultas/ num_pacientes
    return num_pacientes, num_consultas, media_consultas_pacientes

#función para cargar la base de datos que contiene la fecha de la polisomnografia
def cargar_fecha_poli(ruta_datos_fecha):
        df_fecha_poli = pd.read_csv(ruta_datos_fecha, header= None)
        df_fecha_poli = df_fecha_poli.rename({0: "fecha_poli", 1: "cc"}, axis= "columns")
        df_fecha_poli["fecha_poli"] = pd.to_datetime(df_fecha_poli["fecha_poli"], errors = "coerce")

        return df_fecha_poli

def load_sleep_study_data(path_data, name_sleepS_data):
    
    """datos_clinica_sueno = pd.read_csv(path_data + name_sleepS_data, sep = ";", usecols= ["sexo", "cc","Edad", "Peso", "Talla", "IMC", "IAH", "Ronquido", "Somnolencia", "Fatiga", 
            "Despertares falta aire", "Apneas presenciadas", "cefalea", "mov piernas", "Epworth", "HTA", "HTTP", "RGE", 
            "EPOC", "ARTRITIS", "DM", "ASMA", "RINOSIN", "HIPOTIROIDISMO", "CARDIOPATIA", "BYPASS", "EnfermedadCoronaria", "INSUF MITRAL", 
            "ICC", "REEMPVALV", "MARCAPASO", "FA", "Latencia", "lat rem", "eficiencia", "eventos", "mov piernas2", "arousal", 
            "bruxismo", "saturacion mín", "sat máx ", "o2 / CPAP", "Perímetro cuello", "perímetroabdom", "PresionArterialSistolicaNoche",
            "PresionArterialDiastolicaNoche", "PresionArtericalSistolicaManana", "PresionArterialDiastolicaManana", "Medicamentos"]
                                , decimal = ",")"""
    datos_clinica_sueno = pd.read_csv(path_data + name_sleepS_data, sep = ";", decimal = ",", usecols=["cc", "IAH", "o2 / CPAP"], dtype={"cc":"str","IAH":float, "o2 / CPAP":"str"})
    #datos_clinica_sueno = pd.read_csv(path_data + name_sleepS_data, sep = ";", decimal = ",")
    #manejo de datos vacios o nulos
    """values = {'Ronquido': 0, 'Somnolencia':0, 'Fatiga':0, 'Despertares falta aire':0, 'Apneas presenciadas':0, 'cefalea':0,
            'mov piernas':0, 'Epworth':0, 'HTA':0, 'HTTP':0, 'RGE':0, 'EPOC':0, 'ARTRITIS':0, 'DM':0, 'ASMA':0, 'RINOSIN':0,
                'HIPOTIROIDISMO':0, 'CARDIOPATIA':0, 'BYPASS':0, 'ENF,CORON':0, 'INSUF MITRAL':0, 'ICC':0, 'REEMPVALV':0, 
                'MARCAPASO':0, 'FA':0, "bruxismo":0}"""
            
    #datos_clinica_sueno = datos_clinica_sueno.fillna(value = values)

    #se eliminan datos vacios de las columnas IAH, Peso, Talla, Edad, Sexo
    #datos_clinica_sueno = datos_clinica_sueno.dropna(subset= ["IAH"])

    return datos_clinica_sueno

#Funcion cargar base de datos que contiene la relación cedula con idCliente
def load_idcc_data(path_id_sahi):
    datos_id = pd.read_csv(path_id_sahi)
    datos_id = datos_id.rename({"Documento": "cc"}, axis = "columns")
    datos_id = datos_id.drop_duplicates(subset = ["cc"], keep = "first")
    return datos_id

def edad_dia_poli(end_date, start_date):
    # Get the relativedelta between two dates
    delta = relativedelta.relativedelta(end_date, start_date)
    return delta

def load_ehr_data(path_data, name_ehr_data):
    """Loads EHR data from a CSV file.

    # Arguments
        path_data: str, path to the data file.
        name_ehr_data: str, name of the data file.

    # Returns
        A pandas DataFrame.
    """
    return pd.read_csv(path_data + name_ehr_data)

def convertir_fecha_nacimiento(datos, nombreVariable):
    datos[nombreVariable] = pd.to_datetime(datos[nombreVariable], format='%d/%m/%Y')
    return datos

def convertir_datos_fecha(datos, nombreVariable):
    datos[nombreVariable] = pd.to_datetime(datos[nombreVariable], format='%d/%m/%Y %H:%M:%S')    
    return datos

#function to convert a dataframe pandas into a list of dictionaries
def make_listDictionary_patients(dataframe):
    patient_list = []
    for index, row in dataframe.iterrows():        
        #print(row["IdCliente"])
        id_cliente = row["IdCliente"]
        label = row["label_apnea"]
        #print(row["secuencia_recortada"])
        num_consulta = 0       
        dictionary_patient = {"id_cliente":id_cliente, "label":label}
        dictonary_consulta = {}
        for i in row["secuencia_recortada"]:
            num_consulta += 1
            fecha_consulta = i.get("FechaConsulta")
            diagnosticos = i.get("Diagnosticos_Consulta")
            texto = i.get("DesPlanYConcepto")        
            #print(dictionary_patient)
            dictonary_consulta["consulta_"+str(num_consulta)] = {"fecha":str(fecha_consulta), "diagnosticos":diagnosticos, "texto":texto}    
            #print(dictonary_consulta)
            #print(num_consulta)
            dictionary_final = dict(list(dictionary_patient.items()) + [("consultas", dictonary_consulta)])
        #print(dictionary_final)
        patient_list.append(dictionary_final)

    print("tamaño de la lista pacientes {}".format(len(patient_list)))
    return patient_list

def clean_without_alnum(string):
    list_clean = []
    string_list = string.split(" ")
    for term in string_list:
        term = "".join(e for e in term if e.isalnum())
        list_clean.append(term)
    return " ".join(list_clean)

def get_string_dataset(data, name_col):
    all_patients_string = []
    for index, value in data[name_col].items():
        stream = "---"
        patient = "patient:"
        id_patient = "IdCliente: {}".format(index)
        encabezado = stream + "\n" + patient + "\n" + id_patient
        #print(encabezado)
        num_consulta = 0
        all_consult = []
        for k, v in value.items():
            num_consulta += 1
            consulta = "consulta_{}:".format(num_consulta)
            fecha = "  FechaConsulta: {}".format(k)
            diag_string = v.get("Diagnosticos_Consulta")
            #diag_clean = "".join(e for e in diag_string if e.isalnum())
            diagnosticos = "  Diagnosticos_Consulta: '{}'".format(diag_string)
            concept_string = str(v.get("DesPlanYConcepto"))
            concept_string_clean = clean_without_alnum(concept_string)
            conceptos = "  DesPlanYConcepto: '{}'".format(concept_string_clean)        
            string_consult = "\n" + consulta + "\n" + fecha + "\n" + diagnosticos + "\n" + conceptos
            #print(string_consult)
            all_consult.append(string_consult)
        final_string = encabezado + "".join(all_consult)
        all_patients_string.append(final_string)
    return all_patients_string

#Function to review the patient information
def review_patient_info(patient_list):
    for patient in patient_list:
        #print(index)    
        print("id_cliente: {}, label {}".format(patient.get("id_cliente"), patient.get("label")))
        consultas_paciente = patient.get("consultas")
        #print(type(consultas_paciente))
        for num_consulta, values in consultas_paciente.items():
            print("{}:".format(num_consulta))
            print("fecha: {}, diagnosticos: {}, texto: {}".format(values.get("fecha"), values.get("diagnosticos"), values.get("texto")))
    return None

