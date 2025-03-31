#!/bin/bash
#SBATCH --job-name=clinical_extraction
#SBATCH --output=logs/clinical_extraction_%j.out
#SBATCH --error=logs/clinical_extraction_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

source /zine/apps/anaconda_salud/etc/profile.d/conda.sh
conda activate 1cphe

echo "Starting clinical concept extraction job..."

# Define paths
PATH_DATA_TRAIN="/early_data/early_prediction_data1.json"
CURRENT_PATH="/zine/data/salud/compu_Pipe_V3/"
UMLS_TO_ICD_PATH="/map/map_icd10_umls.csv"
QUMLS_PATH="/destination_umls_es"
NUM_PROCESSES=8

echo "El ambiente activado es: "$CONDA_DEFAULT_ENV

# Crear carpeta de logs si no existe
mkdir -p logs

# Run the pipeline
srun python3 clinical_concept_extraction_pipeline_v2.py $PATH_DATA_TRAIN $CURRENT_PATH $UMLS_TO_ICD_PATH $QUMLS_PATH $NUM_PROCESSES
