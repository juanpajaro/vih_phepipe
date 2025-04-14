#!/bin/bash
#SBATCH --job-name=lstm_pipeline
#SBATCH --output=logs/lstm_out_%j.txt
#SBATCH --error=logs/lstm_err_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
source /zine/apps/anaconda_salud/etc/profile.d/conda.sh
conda activate 1cphe
echo "El ambiente activado es: "$CONDA_DEFAULT_ENV
echo "Starting LSTM pipeline job..."

# Set current date
#CURRENT_DATE=$(date +"%Y%m%d_%H%M%S")
CURRENT_DATE="20250408_144607"
echo "Current date: $CURRENT_DATE"

# Define paths
#PATH_DATA="/zine/data/salud/compu_Pipe_V3/"
PATH_DATA="/home/pajaro/compu_Pipe_V3/"
MAX_TOKEN=1000
MAX_LEN=4

# Run the model
#p_report=$(python3 train_lstm.py "$CURRENT_DATE" "$PATH_DATA" $MAX_TOKEN $MAX_LEN)
p_report=$(srun python3 train_lstm.py "$CURRENT_DATE" "$PATH_DATA" $MAX_TOKEN $MAX_LEN)

# Extract PARAM1 and PARAM2
param1=$(echo "$p_report" | grep '^PARAM1=' | cut -d'=' -f2)
param2=$(echo "$p_report" | grep '^PARAM2=' | cut -d'=' -f2)

# Create results folder if it doesn't exist
results_dir="performance_reports"
mkdir -p "$results_dir"

# Determine next version number
base_name="pipeline"
last_version=$(ls "$results_dir"/${base_name}_v*.csv 2>/dev/null | sed -E "s/.*${base_name}_v([0-9]+)\.csv/\1/" | sort -n | tail -n 1)
next_version=$(( (${last_version:-0}) + 1 ))

# Set versioned filename inside results folder
report_file="$results_dir/${base_name}_v${next_version}.csv"

# Write the output
echo ""$CURRENT_DATE","$PATH_DATA",$MAX_TOKEN,$MAX_LEN,$param1,$param2" >> "$report_file"

echo "Parámetros capturados correctamente: $param1,$param2"
echo "Reporte guardado como: $report_file"