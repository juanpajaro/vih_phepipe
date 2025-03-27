#!/bin/bash
#SBATCH --job-name=slurm_test
#SBATCH --output=logs/test_out_%j.txt
#SBATCH --error=logs/test_err_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:01:00
#SBATCH --mem=512M

# Activar Conda
source /zine/apps/anaconda_salud/etc/profile.d/conda.sh
conda activate 1cphe

# Crear carpeta de logs si no existe
mkdir -p logs

# Ejecutar script Python
srun python test_script.py
