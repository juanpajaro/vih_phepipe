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

#CURRENT_DATE="20250520_053753"
#echo "Current date: $CURRENT_DATE"

# Load the conda environment
source /zine/apps/anaconda_salud/etc/profile.d/conda.sh
conda activate 1cphe
echo "Starting computational phenotyping job..."
echo "El ambiente activado es: "$CONDA_DEFAULT_ENV

#RUN DATA TRANSFORMATION STEP
# Define paths
DAYSPW=180
DAYSOW=545
PATH_RAW_DATA="/zine/data/salud/vih_phepipe/base_datos/"
DATA_NAME="Variables_HC.txt"
LABEL_NAME='Etiqueta.txt'

#Create folder for logs if it doesn't exist
#mkdir -p logs

srun python3 wo_data_preparation.py $DAYSPW $DAYSOW "$CURRENT_DATE" "$PATH_RAW_DATA" "$DATA_NAME" "$LABEL_NAME" 

# Define paths
PATH_DATA_TRAIN="data_transformation/data_t_${CURRENT_DATE}.json"
CURRENT_PATH="/zine/data/salud/vih_phepipe/"
UMLS_TO_ICD_PATH="/map/map_icd10_umls.csv"
QUMLS_PATH="/destination_umls_es"
NUM_PROCESSES=8
SIMILARITY_THRESHOLD=0.8
LISTA_CAT=("Disease or Syndrome")
#LISTA_CAT=("Disease or Syndrome", "Sign or Symptom")
DICTIONARY_ICD_LOCAL="icd"

# Convierte la lista en una cadena separada por comas
LIST_AS_STRING=$(IFS=,; echo "${LISTA_CAT[*]}")
#LIST_AS_STRING='{"T047"},{"T184"}'


# Crear carpeta de logs si no existe
#mkdir -p logs

# Run the pipeline
srun python3 clinical_concept_extraction.py $PATH_DATA_TRAIN $CURRENT_PATH $UMLS_TO_ICD_PATH $QUMLS_PATH $NUM_PROCESSES "$CURRENT_DATE" $SIMILARITY_THRESHOLD "$LIST_AS_STRING" "$DICTIONARY_ICD_LOCAL"
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
MAX_LEN=$DAYSOW
#MAX_LEN=30

# Run the model
#p_report=$(python3 train_lstm.py "$CURRENT_DATE" "$PATH_DATA" $MAX_TOKEN $MAX_LEN)
#p_report=$(srun python3 train_lstm.py "$CURRENT_DATE" "$PATH_DATA" $MAX_TOKEN $MAX_LEN)
srun python3 train_lstm_loop.py "$CURRENT_DATE" "$CURRENT_PATH" $MAX_TOKEN $MAX_LEN "$LIST_AS_STRING" "$DICTIONARY_ICD_LOCAL" $DAYSPW $DAYSOW
#python3 train_lstm_loop.py "$CURRENT_DATE" "$PATH_DATA" $MAX_TOKEN $MAX_LEN

# Run the attention model
srun python3 train_attention.py "$CURRENT_DATE" "$CURRENT_PATH" $MAX_TOKEN "$LIST_AS_STRING" "$DICTIONARY_ICD_LOCAL" $DAYSPW $DAYSOW

#Run logistic regression model
srun python3 train_logistic.py "$CURRENT_DATE" "$CURRENT_PATH" $MAX_LEN "$LIST_AS_STRING" "$DICTIONARY_ICD_LOCAL" $DAYSPW $DAYSOW