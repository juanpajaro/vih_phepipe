import sys
import os
import multiprocessing
from clinical_concept_extraction_principal import load_data, run_parallel_extraction, save_data

def run_pipeline(path_data_train, current_path, umlstoicd_path, qumls_path, num_processes):
    print("Loading data...")
    patients_maxLength, clinical_pipe = load_data(path_data_train, current_path, umlstoicd_path, qumls_path)

    print("Running parallel extraction...")
    extracted_results = run_parallel_extraction(patients_maxLength, clinical_pipe, num_processes)

    # Merging results from multiprocessing
    patient_seq = []
    dictionary_entities = {}
    for result in extracted_results:
        if result:
            seq, entities = result[0], result[1]
            patient_seq.extend(seq)
            dictionary_entities.update(entities)

    print("Saving data...")
    save_data(current_path, patient_seq, dictionary_entities)
    print("Pipeline execution completed.")

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python clinical_concept_extraction_pipeline.py <path_data_train> <current_path> <umlstoicd_path> <qumls_path> <num_processes>")
        sys.exit(1)

    path_data_train = sys.argv[1]
    current_path = sys.argv[2]
    umlstoicd_path = sys.argv[3]
    qumls_path = sys.argv[4]
    num_processes = int(sys.argv[5])

    run_pipeline(path_data_train, current_path, umlstoicd_path, qumls_path, num_processes)
