#!/bin/bash
#SBATCH --job-name=split
#SBATCH --output=logs/split_%j.out
#SBATCH --error=logs/split_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

source /zine/apps/anaconda_salud/etc/profile.d/conda.sh
conda activate 1cphe
echo "El ambiente activado es: "$CONDA_DEFAULT_ENV

echo "Starting split job..."

CURRENT_DATE=$(date +"%Y%m%d_%H%M%S")
echo "Current date: $CURRENT_DATE"

# Define paths
PATH="/zine/data/salud/compu_Pipe_V3/"
FILENAME="concepts/concepts_${CURREN_DATE}.json/"
TRAIN_SIZE=0.8

# Run the pipeline
srun python3 split_data_pipeline.py $PATH $FILENAME $TRAIN_SIZE "$CURRENT_DATE"