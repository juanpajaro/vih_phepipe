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

CURRENT_DATE=$(date +"%Y%m%d_%H%M%S")
echo "Current date: $CURRENT_DATE"

# Define paths
PATH_DATA_TRAIN="cases_controls/cases_controls_20250402_150855.json"
CURRENT_PATH="/zine/data/salud/compu_Pipe_V3/"
UMLS_TO_ICD_PATH="/map/map_icd10_umls.csv"
QUMLS_PATH="/destination_umls_es"
NUM_PROCESSES=8
SIMILARITY_THRESHOLD=0.8
#LISTA_CAT=('{"T047"}' '{"T184"}')

# Convierte la lista en una cadena separada por comas
#LIST_AS_STRING=$(IFS=,; echo "${LISTA_CAT[*]}")
LIST_AS_STRING='{"T047"},{"T184"}'


echo "El ambiente activado es: "$CONDA_DEFAULT_ENV

# Crear carpeta de logs si no existe
mkdir -p logs

# Run the pipeline
srun python3 clinical_concept_extraction_pipeline_v2.py $PATH_DATA_TRAIN $CURRENT_PATH $UMLS_TO_ICD_PATH $QUMLS_PATH $NUM_PROCESSES "$CURRENT_DATE" $SIMILARITY_THRESHOLD "$LIST_AS_STRING"
