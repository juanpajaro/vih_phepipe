#!/bin/bash
#SBATCH --job-name=cp
#SBATCH --output=logs/cp_out_%j.txt
#SBATCH --error=logs/cp_err_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Get the current date and time
CURRENT_DATE=$(date +"%Y%m%d_%H%M%S")
echo "Current date: $CURRENT_DATE"

# Load the conda environment
source /zine/apps/anaconda_salud/etc/profile.d/conda.sh
conda activate 1cphe
echo "Starting computational phenotyping job..."
echo "El ambiente activado es: "$CONDA_DEFAULT_ENV

#RUN CASES CONTROL DEFINITION
# Define paths
PATH_DATA="/zine/data/salud/computational_pipe_v2/raw_data/"
#PATH_DATA="./raw_data/"
NAME_POLI_DATA="fecha_cedula_clinica_suenio_may 31 2023.csv"
NAME_SLEEP_DATA="base principal ajustada 11mayo2021.csv"
NAME_IDCC="3636_idClientes.csv"
NAME_EHR_DATA="Vista_Minable_3636.csv"
DAYSPW=180
DAYSOW=730

#Create folder for logs if it doesn't exist
mkdir -p logs

srun python3 data_transformation_pipeline.py "$PATH_DATA" "$NAME_POLI_DATA" "$NAME_SLEEP_DATA" "$NAME_IDCC" "$NAME_EHR_DATA" $DAYSPW $DAYSOW "$CURRENT_DATE"

#RUN CLINICAL CONCEPT EXTRACTION
# Define paths
PATH_DATA_TRAIN="cases_controls/cases_controls_${CURRENT_DATE}.json"
CURRENT_PATH="/zine/data/salud/compu_Pipe_V3/"
UMLS_TO_ICD_PATH="/map/map_icd10_umls.csv"
QUMLS_PATH="/destination_umls_es"
NUM_PROCESSES=8
# Run the pipeline
srun python3 clinical_concept_extraction_pipeline_v2.py $PATH_DATA_TRAIN $CURRENT_PATH $UMLS_TO_ICD_PATH $QUMLS_PATH $NUM_PROCESSES "$CURRENT_DATE"




