#!/bin/bash
#SBATCH --job-name=clinical_extraction
#SBATCH --output=clinical_extraction_%j.out
#SBATCH --error=clinical_extraction_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=standard

module load python/3.8

echo "Starting clinical concept extraction job..."

# Define paths
PATH_DATA_TRAIN="/path/to/train.json"
CURRENT_PATH="/path/to/current"
UMLS_TO_ICD_PATH="/path/to/umls_to_icd.csv"
QUMLS_PATH="/path/to/quickumls"
NUM_PROCESSES=8

# Run the pipeline
srun python3 clinical_concept_extraction_pipeline.py $PATH_DATA_TRAIN $CURRENT_PATH $UMLS_TO_ICD_PATH $QUMLS_PATH $NUM_PROCESSES
