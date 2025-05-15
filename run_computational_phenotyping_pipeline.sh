#!/bin/bash
#SBATCH --job-name=cp
#SBATCH --output=logs/cp_out_%j.txt
#SBATCH --error=logs/cp_err_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
##SBATCH -w hpc02-w002

# Get the current date and time
CURRENT_DATE=$(date +"%Y%m%d_%H%M%S")
echo "Current date: $CURRENT_DATE"

# Load the conda environment
source /zine/apps/anaconda_salud/etc/profile.d/conda.sh
conda activate 1cphe
echo "Starting computational phenotyping job..."
echo "El ambiente activado es: "$CONDA_DEFAULT_ENV

#RUN DATA TRANSFORMATION STEP
# Define paths
PATH_RAW_DATA="/zine/data/salud/computational_pipe_v2/raw_data/"
#PATH_DATA="./raw_data/"
NAME_POLI_DATA="fecha_cedula_clinica_suenio_may 31 2023.csv"
NAME_SLEEP_DATA="base principal ajustada 11mayo2021.csv"
NAME_IDCC="3636_idClientes.csv"
NAME_EHR_DATA="Vista_Minable_3636.csv"
DAYSPW=180
DAYSOW=730

#Create folder for logs if it doesn't exist
mkdir -p logs

srun python3 data_transformation.py "$PATH_RAW_DATA" "$NAME_POLI_DATA" "$NAME_SLEEP_DATA" "$NAME_IDCC" "$NAME_EHR_DATA" $DAYSPW $DAYSOW "$CURRENT_DATE"

# Define paths
PATH_DATA_TRAIN="data_transformation/data_t_${CURRENT_DATE}.json"
CURRENT_PATH="/zine/data/salud/compu_Pipe_V3/"
#CURRENT_PATH="/home/pajaro/compu_Pipe_V3/"
UMLS_TO_ICD_PATH="/map/map_icd10_umls.csv"
QUMLS_PATH="/destination_umls_es"
NUM_PROCESSES=8
SIMILARITY_THRESHOLD=0.8
#LISTA_CAT=("Disease or Syndrome")
#LISTA_CAT=("icd_10", "Disease or Syndrome", "Sign or Symptom")
#LISTA_CAT=None
DICTIONARY_ICD_LOCAL=True

# Convierte la lista en una cadena separada por comas
LIST_AS_STRING=$(IFS=,; echo "${LISTA_CAT[*]}")
#LIST_AS_STRING='{"T047"},{"T184"}'


# Crear carpeta de logs si no existe
#mkdir -p logs

# Run the pipeline
srun python3 clinical_concept_extraction.py $PATH_DATA_TRAIN $CURRENT_PATH $UMLS_TO_ICD_PATH $QUMLS_PATH $NUM_PROCESSES "$CURRENT_DATE" $SIMILARITY_THRESHOLD "$LIST_AS_STRING" $DICTIONARY_ICD_LOCAL
#python3 clinical_concept_extraction_pipeline_v2.py $PATH_DATA_TRAIN $CURRENT_PATH $UMLS_TO_ICD_PATH $QUMLS_PATH $NUM_PROCESSES "$CURRENT_DATE" $SIMILARITY_THRESHOLD "$LIST_AS_STRING"

#RUN DATA SPLITTING STEP
# Define paths
#PATH_DATA="/zine/data/salud/compu_Pipe_V3/"
#PATH_DATA="/home/pajaro/compu_Pipe_V3/"
FILENAME="/concepts/clinical_concepts_${CURRENT_DATE}.json"
TRAIN_SIZE=0.8
srun python3 split_data.py "$CURRENT_PATH" "$FILENAME" $TRAIN_SIZE "$CURRENT_DATE"
#python3 split_data_pipeline.py "$PATH_DATA" "$FILENAME" $TRAIN_SIZE "$CURRENT_DATE"

#RUN LSTM TRAINING STEP
conda deactivate
conda activate tf_envs_v2
echo "El ambiente activado es: "$CONDA_DEFAULT_ENV

# Define paths
#PATH_DATA="/zine/data/salud/compu_Pipe_V3/"
#PATH_DATA="/home/pajaro/compu_Pipe_V3/"
MAX_TOKEN=1000
MAX_LEN=730

# Run the model
#p_report=$(python3 train_lstm.py "$CURRENT_DATE" "$PATH_DATA" $MAX_TOKEN $MAX_LEN)
#p_report=$(srun python3 train_lstm.py "$CURRENT_DATE" "$PATH_DATA" $MAX_TOKEN $MAX_LEN)
srun python3 train_lstm_loop.py "$CURRENT_DATE" "$CURRENT_PATH" $MAX_TOKEN $MAX_LEN "$LIST_AS_STRING"
#python3 train_lstm_loop.py "$CURRENT_DATE" "$PATH_DATA" $MAX_TOKEN $MAX_LEN