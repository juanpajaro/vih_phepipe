#!/usr/bin/env python3
import os
from multiprocessing import Pool, cpu_count
from conceptse import ClinicalExtraction

def ejecutar_extraccion(worker_id):
    print(f"Ejecutando extracción en worker {worker_id}")
    name_data = "/early_data/early_prediction_data1.json"
    extraction = ClinicalExtraction(name_data, "/map/map_icd10_umls.csv", "/destination_umls_es")
    extraction.run_pipe_ec()
    return f"Worker {worker_id} completado."

if __name__ == "__main__":
    n_workers = cpu_count()
    print(f"Usando {n_workers} workers...")
    with Pool(processes=n_workers) as pool:
        results = pool.map(ejecutar_extraccion, range(n_workers))
    for r in results:
        print(r)

print("concept extraction ended...")