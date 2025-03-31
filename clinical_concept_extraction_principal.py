import os
import multiprocessing
import utils_general_porpose
import utils_clinical_concept_extraction
import utils_split_dataset

def load_data(path_data_train, current_path, umlstoicd_path, qumls_path):
    data_train = utils_general_porpose.load_json(current_path, path_data_train)

    patients = utils_split_dataset.send_patient_info(data_train[:10])
    patients_maxLength = utils_split_dataset.make_patient_list_with_maxlength(patients, 1000000)

    lista_example_umls_icd = utils_general_porpose.load_mapping_icd(current_path, umlstoicd_path)
    _, lista_identificados = utils_clinical_concept_extraction.list_codes_identified(lista_example_umls_icd)
    target_rules = utils_clinical_concept_extraction.load_target_rules(lista_identificados)
    clinical_pipe = utils_clinical_concept_extraction.load_clinical_NLPpipe(current_path, qumls_path, target_rules)

    return patients_maxLength, clinical_pipe

def process_patient(patient, clinical_pipe):
    return utils_clinical_concept_extraction.extract_clinical_concepts([patient], clinical_pipe)

def run_parallel_extraction(patients, clinical_pipe, num_processes):
    with multiprocessing.Pool(num_processes) as pool:
        results = pool.map(lambda patient: process_patient(patient, clinical_pipe), patients)
    return results

def save_data(current_path, patient_seq, dictionary_entities):
    path_save = os.path.join(current_path, "concepts")
    path_save = utils_general_porpose.create_directory(path_save)

    if os.path.exists(path_save):
        list_dataframes = utils_general_porpose.extract_name_model(path_save, ".json")
        last_version = utils_general_porpose.extract_last_version_model(list_dataframes)
        version = str(utils_general_porpose.counter_version(last_version))

        path_version = os.path.join(path_save, f"clinical_concepts{version}.json")
        utils_general_porpose.save_json(patient_seq, path_version)

        path_entities = os.path.join(path_save, f"dictionary_concepts{version}.json")
        utils_general_porpose.save_json(dictionary_entities, path_entities)
