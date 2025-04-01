#!/usr/bin/env python3
import time
#import pandas as pd
import utils_early_disease
import os
#import utils_train_models
import utils_general_porpose
#import yaml

class DatasetTransformation:
    def __new__(cls, *args, **kwargs):
        print("DatasetTransformation object created")
        return super().__new__(cls)
    
    #path_data = "/mnt/g/My Drive/clinical_phenotypes_OSA_frontiers/golden_standar/"
    #name_ehr_data = "Vista_Minable_3636.csv"
    #name_poli_data = "fecha_cedula_clinica_suenio_may 31 2023.csv"
    #name_sleepS_data = "base principal ajustada 11mayo2021.csv"
    #name_idcc = "3636_idClientes.csv"
    #num_dias = 180
      
    def __init__(self, path_data, name_poli_data, name_sleepS_data, name_idcc, name_ehr_data, num_dias):
        self.path_data = path_data
        self.name_poli_data = name_poli_data
        self.name_sleepS_data = name_sleepS_data
        self.name_idcc = name_idcc
        self.name_ehr_data = name_ehr_data
        self.num_dias = num_dias
        print("DatasetTransformation object instantiated")

    def __repr__(self):
        return "filter_name: {}, path_data: {}".format(type(self).__name__, self.path_data)

    #This function transform four dataset into a single early prediction dataset, and return it
    def run_transformation_pipe(self):
        #Load polisomnography date data
        data_poliDate = utils_early_disease.cargar_fecha_poli(self.path_data + self.name_poli_data)
        print("polisomnography date data loaded")

        #Sort polisomnography date data by date
        columna_fecha = "fecha_poli"
        data_poliDate = utils_early_disease.ordenar_datos_por_fecha(data_poliDate, columna_fecha)
        print("polisomnography date data sorted by date")

        #Remove duplicates from polisomnography date data
        duplicated_column = "cc"
        data_poliDate = utils_early_disease.eliminar_duplicados(data_poliDate, duplicated_column)
        print("duplicates removed from polisomnography date data")

        #Load sleep study data
        data_sleepStudy = utils_early_disease.load_sleep_study_data(self.path_data, self.name_sleepS_data)
        print("sleep study data loaded")

        #Cleanse sleep study data
        #el valor de uno (1) en la columna [o2/CPAP] indica que no es un estudio de titulación puede ser incluído dentro de la investigación
        lista_de_texto, lista_de_digitos = utils_early_disease.saber_si_digito_texto(data_sleepStudy["o2 / CPAP"], valor_incluido = "1")
        columna = "o2 / CPAP"
        lista_valores = lista_de_digitos + lista_de_texto
        data_sleepStudy = utils_early_disease.excluir_incluir_pacientes(data_sleepStudy, columna, lista_valores)
        print("sleep study data cleansed, only patients without titration studies are included")

        #remove duplicates from sleep study data
        data_sleepStudy = data_sleepStudy[data_sleepStudy["excluir_incluir"] == "incluir"]
        data_sleepStudy = utils_early_disease.eliminar_duplicados(data_sleepStudy, "cc")
        print("duplicates removed from sleep study data")

        #Merge polisomnography date data and sleep study data
        lista_dataframes = [data_sleepStudy, data_poliDate]
        data = utils_early_disease.mergue_datasets(lista_dataframes, "cc")
        print("polisomnography date data and sleep study data merged")

        #Remove patients without polisomnography date
        columna_datos_faltantes = "fecha_poli"
        data = utils_early_disease.eliminar_datos_faltantes_col(data, columna_datos_faltantes)
        print("patients without polisomnography date removed")

        #Add label for cases and controls
        column_cases = "IAH"
        data = utils_early_disease.agregar_variable_objetivo(data, column_cases)
        print("label added for cases (IAH > 5) and controls (IAH <= 5)")

        #Get columns that are relevant for the early prediction of disease.
        data = data[["cc","IAH", "o2 / CPAP", "fecha_poli", "label_apnea"]]
        print("columns relevant for the early prediction of disease selected")

        #Load id and cc dataset
        data_idcc = utils_early_disease.load_idcc_data(self.path_data + self.name_idcc)
        print("id and cc dataset loaded")

        #Merge id and cc dataset with the previous dataset
        lista_dataframes = [data_idcc, data]
        data = utils_early_disease.mergue_datasets(lista_dataframes, "cc")
        print("id and cc dataset merged with the previous dataset")

        #Remove patients without label
        column_label = "label_apnea"    
        data = utils_early_disease.eliminar_datos_faltantes_col(data, column_label)
        print("patients without label removed")

        #Rename the IdCliente column
        data = data.rename({"idCliente": "IdCliente"}, axis = "columns")
        print("IdCliente column renamed")

        #Load ehr data
        start_time = time.time()
        data_ehr = utils_early_disease.load_ehr_data(self.path_data, self.name_ehr_data)
        end_time = time.time()
        print("Tiempo de ejecución en s: ", (end_time - start_time)/60)
        print("ehr data loaded")

        #transform FecNacimiento into datetime type
        data_ehr = utils_early_disease.convertir_fecha_nacimiento(data_ehr, "FecNacimiento")
        print("FecNacimiento transformed into datetime type")

        #transform FecIngreso and FechaConsulta into datetime type
        lista_de_columnas = ["FecIngreso", "FechaConsulta"]
        for NomVariable in lista_de_columnas:
            data_ehr = utils_early_disease.convertir_datos_fecha(data_ehr, NomVariable)
        print("FecIngreso and FechaConsulta transformed into datetime type")

        #Join ehr per patient
        datos_unicos_ehr = data_ehr[["IdCliente", "FecNacimiento", "Sexo"]]
        datos_sec_con = data_ehr[["IdCliente", "IdConsulta", "FechaConsulta", 'Talla_Consulta', 'Peso_Consulta', 'Temperatura_Consulta',
        'FC_Consulta', 'FR_Consulta', 'PS_Consulta', 'PD_Consulta',
        'Saturacion_Consulta', 'DesEnfermedadActual', 'DesExamenFisico',
        'DesMotivoCon', 'DesPlanTrat', 'DesPlanYConcepto',
        'DesValoracionSistemas', 'Diagnosticos_Consulta',
        'Medicamentos_Aplicados', 'Medicamentos_Formulados', 'Procedimientos',
        'NotasAtencion']]
        #print("only the follow variables are selected: {}".format(datos_unicos_ehr + datos_sec_con))
        
        #se agrupan los registros por pacientes con los datos unicos para obtener una sola fecha nacimiento y un solo sexo
        datos_unicos_ehr_pacientes = utils_early_disease.agrupar_pacientes_por_fechaNacimiento_sexo(datos_unicos_ehr, "IdCliente", "FecNacimiento", "Sexo")
        

        #Group EHR data per patient as a sequence    
        lista_columnas = ['FechaConsulta', 'Diagnosticos_Consulta', 'DesPlanYConcepto']
        columna_id = "IdCliente"
        nombre_columna_final = "dic_datos_consulta"
        datos_sec_consulta_pacientes = utils_early_disease.make_sequences_ehr(datos_sec_con, columna_id, lista_columnas, nombre_columna_final)
        print("ehr data per patient joined")
        print("Only the follow variables are selected for this version of the clinical phenotyping pipeline in OSA: {}".format(lista_columnas))

        #sort by date
        """datos_sec_consulta_pacientes["dic_datos_consulta"] = datos_sec_consulta_pacientes.apply(lambda x: utils_early_disease.sort_dict_by_date(x["dic_datos_consulta"], "%Y-%m-%d %H:%M:%S"), axis = 1)
        print("ehr data per patient sorted by date of events")"""

        #Join ehr data
        lista_dataframes = [datos_unicos_ehr_pacientes, datos_sec_consulta_pacientes]
        datos_ehr_pacientes = utils_early_disease.mergue_datasets(lista_dataframes, "IdCliente")
        #print("ehr data per patient joined with previous dataset")

        #el orden en la cual se carguen los dataframe importa
        lista_dataframes = [datos_ehr_pacientes, data]
        data = utils_early_disease.mergue_datasets(lista_dataframes, "IdCliente")
        print("ehr data per patient joined with previous dataset")

        #Remove patients without label
        column_label = "label_apnea"
        data = utils_early_disease.eliminar_datos_faltantes_col(data, column_label)
        print("patients without label removed")

        #Transform date of birth into age according to the date of the polisomnography
        for index, row in data[["fecha_poli", "FecNacimiento"]].iterrows():        
            delta = utils_early_disease.edad_dia_poli(row["FecNacimiento"], row["fecha_poli"])
            edad = delta.years *(-1)
            data.loc[index, "edad_poli"] = edad
        print("date of birth transformed into age according to the date of the polisomnography")
        
        #Cut the patients records
        #dependiendo de la fecha de recorte de la polisomnografía, se define el  numero de pacientes
        #num_dias = 180
        data["secuencia_recortada"] = data.apply(lambda x: utils_early_disease.recortar_historia(x["dic_datos_consulta"], x["fecha_poli"], self.num_dias), axis = 1)
        print("patients records cut: {} days".format(self.num_dias))

        #TODO: this has to be a function in utils_early_disease.py
        #Remove patients without ehr data before the prediction window
        for index, row in data["secuencia_recortada"].items():
            if len(row) == 0:
                dic_vacio = "empty"
            else:
                dic_vacio = "not empty"
            data.loc[index, "vacios_poli"] = dic_vacio

        #TODO: this has to be a function in utils_early_disease.py
        data = data[data["vacios_poli"] == "not empty"]
        print("patients without ehr data before the prediction window removed")
        print("data transformation process finished")
        
        #TODO: this has to be a function in utils_early_disease.py
        data = data[data["label_apnea"] < 3]
        print("patients with label 3 removed")

        #current directory for save the dataset and json file
        current_path = os.getcwd()

        #Make a code to save the dataset
        directory_name = "/dataframes/"
        path_frame_directory = current_path + directory_name
        path_frame_directory = utils_general_porpose.create_directory(path_frame_directory)
        #save the dataset
        data.to_csv(path_frame_directory + "dataset.csv", index = False)
        print("dataset saved in csv format")


        #save the dataset as json file
        data  = utils_early_disease.make_listDictionary_patients(data)
        print("data transformed into list of dictionaries")

        #The directory exists or not?
        
        path_save = current_path + "/early_data/"
        path_save = utils_general_porpose.create_directory(path_save)

        #TODO: Review this part of the code
        #Save the dataset    
        #This part of the code is for save the dataset in a yml file
        """if os.path.exists(path_save):
            #if some version of dataset exists, then get the name list of the dataset
            list_dataframes = utils_train_models.extract_name_model(path_save, ".yaml")
            #extract the last version of the dataset
            last_version = utils_train_models.extract_last_version_model(list_dataframes)
            #get the number of the new version
            version = str(utils_train_models.counter_version(last_version))
            #save the dataset
            path_version = path_save + "early_prediction_dataset_v" + version + ".yaml"
            #data.to_csv(path_version, index = False)
                    
            with open(path_version, "w", encoding="latin-1") as f:
                yaml.dump_all(patients_prueba_yaml, f)
        else:
            #data.to_csv(path_save + "early_prediction_dataset_v1.csv", index = False)
            with open(path_save + "early_prediction_dataset_v1.yaml", "w", encoding="latin-1") as f:
                yaml.dump_all(patients_prueba_yaml, f)"""

        #TODO: Review this part of the code    
        #Save the dataset as json file
        if os.path.exists(path_save):
            #if some version of dataset exists, then get the name list of the dataset
            list_dataframes = utils_general_porpose.extract_name_model(path_save, ".json")
            #extract the last version of the dataset
            last_version = utils_general_porpose.extract_last_version_model(list_dataframes)
            #get the number of the new version
            version = str(utils_general_porpose.counter_version(last_version))
            #save the dataset
            path_version = path_save + "early_prediction_data" + version + ".json"
            #data.to_csv(path_version, index = False)
            utils_general_porpose.save_json(data, path_version)

        print("dataset saved")        
        return None


#print("module dataset transformation imported")

""""path_data = "./raw_data/"
name_ehr_data = "Vista_Minable_3636.csv"
name_poli_data = "fecha_cedula_clinica_suenio_may 31 2023.csv"
name_sleepS_data = "base principal ajustada 11mayo2021.csv"
name_idcc = "3636_idClientes.csv"
num_dias = 180

dataset_transformation = DatasetTransformation(path_data, name_poli_data, name_sleepS_data, name_idcc, name_ehr_data, num_dias)
dataset_transformation
dataset_transformation.run_transformation_pipe()
"""

