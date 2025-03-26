#!/usr/bin/env python3
import utils_clinical_concept_extraction
import utils_general_porpose
import os

#path_data = "/mnt/g/My Drive/clinical_phenotypes_OSA_frontiers/golden_standar/"

#name_col_sequence = "secuencia_recortada"
#value_diagnostic = "Diagnosticos_Consulta"
#value_clinical_note = "DesPlanYConcepto"

#name_file = "/genDiagnostico.csv"

#path_destination = "/home/pajaro/pipeline_project_v2/computational_pipe_v2"
#name_database = "/destination_umls_es_v1"

#name_col_history_patient = "history_patient"

#Function to extract clinical concepts from the clinical history patient
def main():    
    current_path = utils_general_porpose.get_current_path()
    #print("current path is {}".format(current_path))
    
    data = utils_general_porpose.load_json(current_path, "/datasets/early_prediction_data3.json")
    #print("data loaded")

    patients = utils_clinical_concept_extraction.send_patient_info(data[:10])
    #print("patients info convert into sequences")

    lista_example_umls_icd = utils_general_porpose.load_mapping_icd(current_path, "/map/map_icd10_umls.csv")
    #print("mapping icd10-umls loaded")
    
    lista_NoIdentificados, lista_identificados = utils_clinical_concept_extraction.list_codes_identified(lista_example_umls_icd)
    #print("list of codes identified and not identified created")

    target_rules = utils_clinical_concept_extraction.load_target_rules(lista_identificados)
    #print("target rules loaded")

    clinical_pipe = utils_clinical_concept_extraction.load_clinical_NLPpipe(current_path, "/destination_umls_es", target_rules)
    #print("medspacy nlp-pipeline loaded")

    patient_seq = utils_clinical_concept_extraction.extract_clinical_concepts(patients[:10], clinical_pipe)
    #print("clinical concepts extracted")

    path_save = current_path + "/concepts/"
    path_save = utils_general_porpose.create_directory(path_save)
    #print("save the clinical concepts in {}".format(path_save))

    if os.path.exists(path_save):
        #if some version of dataset exists, then get the name list of the dataset
        list_dataframes = utils_general_porpose.extract_name_model(path_save, ".json")
        #extract the last version of the dataset
        last_version = utils_general_porpose.extract_last_version_model(list_dataframes)
        #get the number of the new version
        version = str(utils_general_porpose.counter_version(last_version))
        #save the dataset
        path_version = path_save + "clinical_concepts" + version + ".json"
        #data.to_csv(path_version, index = False)
        utils_general_porpose.save_json(patient_seq, path_version)

    print("clinical concepts extraction process finished")

if __name__ == "__main__":
    main()