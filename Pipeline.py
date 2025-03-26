#!/usr/bin/env python3

from dataset_transformation import DatasetTransformation
from utils_class_split import DatasetSplit
from conceptse import ClinicalExtraction

path_data = "/home/pajaro/pipeline_project_v2/computational_pipe_v2/raw_data/"
name_ehr_data = "Vista_Minable_3636.csv"
name_poli_data = "fecha_cedula_clinica_suenio_may 31 2023.csv"
name_sleepS_data = "base principal ajustada 11mayo2021.csv"
name_idcc = "3636_idClientes.csv"
num_dias = 180

dataset_transformation = DatasetTransformation(path_data, name_poli_data, name_sleepS_data, name_idcc, name_ehr_data, num_dias)
dataset_transformation
dataset_transformation.run_transformation_pipe()

split = DatasetSplit("/early_data/early_prediction_data1.json")
split
split.run_pipe_ds()

extraction = ClinicalExtraction("/early_data/early_prediction_data1.json", "/map/map_icd10_umls.csv", "/destination_umls_es")
extraction.run_pipe_ec()
print("concept extraction ended...")