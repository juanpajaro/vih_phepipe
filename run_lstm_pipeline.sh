#!/bin/bash
#SBATCH --job-name=lstm_pipeline
#SBATCH --output=logs/lstm_out_%j.txt
#SBATCH --error=logs/lstm_err_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
##source /zine/apps/anaconda_salud/etc/profile.d/conda.sh
##conda activate 1cphe
echo "El ambiente activado es: "$CONDA_DEFAULT_ENV
echo "Starting LSTM pipeline job..."
##CURRENT_DATE=$(date +"%Y%m%d_%H%M%S")
CURRENT_DATE="20250408_144607"
echo "Current date: $CURRENT_DATE"
# Define paths
#PATH_DATA="/zine/data/salud/compu_Pipe_V3/"
PATH_DATA="/home/pajaro/compu_Pipe_V3/"
MAX_TOKEN=1000
MAX_LEN=4
p_report=$(python3 train_lstm.py "$CURRENT_DATE" "$PATH_DATA" $MAX_TOKEN $MAX_LEN)
# Extraer solo las líneas que contienen PARAM1 y PARAM2
param1=$(echo "$p_report" | grep '^PARAM1=' | cut -d'=' -f2)
param2=$(echo "$p_report" | grep '^PARAM2=' | cut -d'=' -f2)
echo ""$CURRENT_DATE","$PATH_DATA",$MAX_TOKEN,$MAX_LEN,$param1,$param2" >> performance_report.csv
echo "Parámetros capturados correctamente: $param1,$param2"