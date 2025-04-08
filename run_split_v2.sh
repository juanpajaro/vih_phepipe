#!/bin/bash
#SBATCH --job-name=split_data
#SBATCH --output=logs/split_data_%j.out
#SBATCH --error=logs/split_data_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

source /zine/apps/anaconda_salud/etc/profile.d/conda.sh
conda activate 1cphe
echo "El ambiente activado es: "$CONDA_DEFAULT_ENV
echo "Starting data splitting job..."
CURRENT_DATE=$(date +"%Y%m%d_%H%M%S")
echo "Current date: $CURRENT_DATE"
# Define paths
PATH_DATA="/zine/data/salud/compu_Pipe_V3/"
#PATH_DATA="/home/pajaro/compu_Pipe_V3/"
FILENAME="/concepts/clinical_concepts_20250404_171428.json"
TRAIN_SIZE=0.8
srun python3 split_data_pipeline.py "$PATH_DATA" "$FILENAME" $TRAIN_SIZE "$CURRENT_DATE"
#python3 split_data_pipeline.py "$PATH_DATA" "$FILENAME" $TRAIN_SIZE "$CURRENT_DATE"