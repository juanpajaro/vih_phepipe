#!/bin/bash
#SBATCH --job-name=feature
#SBATCH --output=logs/fa_out_%j.txt
#SBATCH --error=logs/fa_err_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

NAME_MODEL=lstm_v66.h5
TOKENIZER=vectorizer_obj.pkl
DATA_TEST=X_test.npy

source /zine/apps/anaconda_salud/etc/profile.d/conda.sh
conda activate shap_v20
echo "El ambiente activado es: "$CONDA_DEFAULT_ENV

srun python3 run_shap_analysis.py $NAME_MODEL -t $TOKENIZER -se $DATA_TEST -f

