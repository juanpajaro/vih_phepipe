#!/bin/bash
#SBATCH --job-name=cases_control_definition
#SBATCH --output=logs/cases_control_definition_%j.out
#SBATCH --error=logs/cases_control_definition_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

source /zine/apps/anaconda_salud/etc/profile.d/conda.sh
conda activate 1cphe
echo "El ambiente activado es: "$CONDA_DEFAULT_ENV

echo "Starting cases control definition job..."

CURRENT_DATE=$(date +"%Y%m%d_%H%M%S")
echo "Current date: $CURRENT_DATE"

#TODO: Piensa en como pegar un nombre que se vaya versionando

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

# Run the pipeline
srun python3 data_transformation_pipeline.py "$PATH_DATA" "$NAME_POLI_DATA" "$NAME_SLEEP_DATA" "$NAME_IDCC" "$NAME_EHR_DATA" $DAYSPW $DAYSOW "$CURRENT_DATE"
#echo PATH_DATA $PATH_DATA
#python3 data_transformation_pipeline.py "$PATH_DATA" "$NAME_POLI_DATA" "$NAME_SLEEP_DATA" "$NAME_IDCC" "$NAME_EHR_DATA" $DAYSPW $DAYSOW "$CURRENT_DATE"