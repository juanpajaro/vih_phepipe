#!/usr/bin/env python3
import os
import spacy
from multiprocessing import Pool, cpu_count
from conceptse import ClinicalExtraction

name_data = "/early_data/early_prediction_data1.json"
extraction = ClinicalExtraction(name_data, "/map/map_icd10_umls.csv", "/destination_umls_es")

def load_data():
    extraction.load_data()

def ejecutar_extraccion(worker_id):
    print(f"Ejecutando extracción en worker {worker_id}")
    extraction.run_pipe_ec()
    return f"Worker {worker_id} completado."

def ejecutar_extraccion_normal():
    print("Ejecutando extracción normal")
    extraction.run_pipe_ec()

def save_data():
    extraction.save_data()

if __name__ == "__main__":
    n_workers = cpu_count()
    print(f"Usando {n_workers} workers...")
    load_data()
    ejecutar_extraccion_normal()

    """
    with Pool(processes=n_workers) as pool:
        results = pool.map(ejecutar_extraccion, range(n_workers))
    for r in results:
        print(r)
    """
    
    save_data()

print("concept extraction ended...")