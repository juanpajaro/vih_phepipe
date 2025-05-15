#!/usr/bin/env python3
import sys
import os
import multiprocessing
import utils_clinical_concept_extraction
import utils_general_porpose
import utils_split_dataset

# Global clinical pipeline
clinical_pipe = None
cat_semantic = None

def load_data(path_data_train, current_path, umlstoicd_path, qumls_path, simi, lista_cat):
    # Load the data
    data_train = utils_general_porpose.load_json(current_path, path_data_train)
    
    # Extract patient information
    patients = utils_split_dataset.send_patient_info(data_train)
    
    # Create a list of patients with a maximum length
    patients_maxLength = utils_split_dataset.make_patient_list_with_maxlength(patients, 1000000)
    
    # Load mapping and clinical NLP pipeline
    lista_example_umls_icd = utils_general_porpose.load_mapping_icd(current_path, umlstoicd_path)
    lista_NoIdentificados, lista_identificados = utils_clinical_concept_extraction.list_codes_identified(lista_example_umls_icd)
    target_rules = utils_clinical_concept_extraction.load_target_rules(lista_identificados)

    # Initialize the clinical pipeline
    global clinical_pipe
    clinical_pipe = utils_clinical_concept_extraction.load_clinical_NLPpipe(current_path, qumls_path, target_rules, simi)
    print("Clinical pipeline initialized.")

    global cat_semantic
    cat_semantic = lista_cat
    print("Category semantic initialized.")

    return patients_maxLength


def extract_concepts(patients_list):
    # Extract clinical concepts
    patients_seq = []
    dictionary_entities = {}
    #print("lista_cat_semantic: ", cat_semantic)
    #print(type(cat_semantic))

    for patient in patients_list:
        id_cliente = patient.get("id_cliente")
        label = patient.get("label")
        seq = patient.get("seq")        
        doc = clinical_pipe(seq)
        list_entities = []
        list_codes = []        

        #matches = [ent for ent in doc.ents if ent._.semtypes in cat_semantic]
        #matches = []
        for ent in doc.ents:
            #print("ent: ", ent)
            #print("ent._.semtypes: ", ent._.semtypes)
            #if ent._.semtypes in cat_semantic:
                #print("ent: ", ent)
                #print("ent._.semtypes: ", ent._.semtypes)

                #matches.append(ent)
                
        #print("matches: ", matches)
        
        #for ent in matches:               
            if ent._.description == "" and ent._.semtypes in cat_semantic:
                #print(ent.text)
                entity = ent.text.split()
                list_entities.append("_".join(entity))
                #print(ent.label_)
                list_codes.append(ent.label_)                
                dictionary_entities[ent.label_] = "_".join(entity)
            
            elif ent._.description != "" and "{icd}" in cat_semantic:
                print(ent._.Diagnostic)
                print(ent._.description)
                print(ent._.cui_code)
                entity = ent._.description.split()
                list_entities.append("_".join(entity))
                #print(ent.text)
                list_codes.append(ent._.cui_code)                
                dictionary_entities[ent._.cui_code] = "_".join(entity)
        
        list_entities_str = " ".join(list_entities)
            #print(list_entities_str)
        list_codes_str = " ".join(list_codes)
            #print(list_codes_str)

        dict_patient = {"id_cliente":id_cliente, "label":label, "entities":list_entities_str, "codes":list_codes_str}
        patients_seq.append(dict_patient)

    #print("clinical concepts extracted")
    return patients_seq, dictionary_entities
    
def save_data(patients_seq, dictionary_entities, current_path, timestamp):
#save the dataset
    path_save = current_path + "/concepts/"
    path_save = utils_general_porpose.create_directory(path_save)

    if os.path.exists(path_save):
        #if some version of dataset exists, then get the name list of the dataset
        list_dataframes = utils_general_porpose.extract_name_model(path_save, ".json")
        #extract the last version of the dataset
        last_version = utils_general_porpose.extract_last_version_model(list_dataframes)
        #get the number of the new version
        version = str(utils_general_porpose.counter_version(last_version))
        #save the dataset
        path_version = path_save + "clinical_concepts_" + timestamp + ".json"
        #data.to_csv(path_version, index = False)
        utils_general_porpose.save_json(patients_seq, path_version)
        #save the dictionary of entities
        path_entities = path_save + "dictionary_concepts_" + timestamp + ".json"
        utils_general_porpose.save_json(dictionary_entities, path_entities)
        
if __name__ == "__main__":
    if len(sys.argv) != 9:
        print("Usage: python clinical_concept_extraction_pipeline.py <path_data_train> <current_path> <umlstoicd_path> <qumls_path> <num_processes>")
        sys.exit(1)

    path_data_train = sys.argv[1]
    current_path = sys.argv[2]
    umlstoicd_path = sys.argv[3]
    qumls_path = sys.argv[4]
    num_processes = int(sys.argv[5])
    timestamp = sys.argv[6]
    simi = float(sys.argv[7])
    lista_tipos_semanticos = sys.argv[8].split(",")
    
    #Example local paths
    """
    path_data_train = "cases_controls/cases_controls_20250402_205502.json"
    current_path = "/home/pajaro/compu_Pipe_V3/"
    umlstoicd_path = "/map/map_icd10_umls.csv"
    qumls_path = "/destination_umls_es"
    simi = 0.8
    lista_cat = [{"T047"},{"T184"}]
    """
    #print("lista: ", lista_tipos_semanticos)

    lista_cat = utils_clinical_concept_extraction.buscar_terminos_en_diccionario(lista_tipos_semanticos)
    print("list of semantic type included: ", lista_cat)

    #print("lista_cat: ", lista_cat)
    
    n_workers = multiprocessing.cpu_count()
    print(f"Using {n_workers} workers...")

    # Load data
    patients_maxLength = load_data(path_data_train, current_path, umlstoicd_path, qumls_path, simi, lista_cat)
    print("Data loaded.")

    # Split the data into chunks for parallel processing
    chunk_size = len(patients_maxLength) // n_workers
    print(f"Chunk size: {chunk_size}")
    print(f"Number patienst: {len(patients_maxLength)}")
    chunks = [patients_maxLength[i:i + chunk_size] for i in range(0, len(patients_maxLength), chunk_size)]

    # If there are any remaining patients, add them to the last chunk
    if len(patients_maxLength) % n_workers != 0:
        chunks[-1].extend(patients_maxLength[len(chunks) * chunk_size:])

    # Run parallel extraction
    print("Running parallel extraction...")
    with multiprocessing.Pool(processes=n_workers) as pool:
        results = pool.map(extract_concepts, chunks)
    
    #patients_seq, dictionary_entities = extract_concepts(patients_maxLength[2:3])
    #print("patients_seq: ", patients_seq)

    # Merging results from multiprocessing
    patient_seq = []
    dictionary_entities = {}
    for result in results:
        if result:
            seq, entities = result[0], result[1]
            patient_seq.extend(seq)
            dictionary_entities.update(entities)
            
    # Save data
    print("Saving data...")
    save_data(patient_seq, dictionary_entities, current_path, timestamp)
    print("Extraction clinical concept execution completed.")

    

