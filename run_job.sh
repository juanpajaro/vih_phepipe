#!/bin/bash
#SBATCH --job-name=clinical_extraction
#SBATCH --output=clinical_extraction_%j.out
#SBATCH --error=clinical_extraction_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=standard

source /zine/apps/anaconda_salud/etc/profile.d/conda.sh
conda activate 1cphe

echo "Starting clinical concept extraction job..."

# Define paths
PATH_DATA_TRAIN="/zine/data/salud/compu_Pipe_V3/early_data/early_prediction_data1.json"
CURRENT_PATH="/zine/data/salud/compu_Pipe_V3"
UMLS_TO_ICD_PATH="/map/map_icd10_umls.csv"
QUMLS_PATH="/destination_umls_es"
NUM_PROCESSES=8

# Run the pipeline
srun python3 clinical_concept_extraction_pipeline.py $PATH_DATA_TRAIN $CURRENT_PATH $UMLS_TO_ICD_PATH $QUMLS_PATH $NUM_PROCESSES
