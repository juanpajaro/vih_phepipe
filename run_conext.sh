#!/bin/bash
#SBATCH --job-name=concept_extraction
#SBATCH --output=logs/out_%j.txt
#SBATCH --error=logs/err_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8     # Ajusta este valor si es necesario
##SBATCH --time=00:30:00
#SBATCH --mem=8G

# Creamos vaiables de entorno
export PATH="/zine/apps/anaconda_salud/bin:$PATH"

# Iniciamos CONDA
conda init
source /zine/apps/anaconda_salud/etc/profile.d/conda.sh
conda activate phepi_v3

# Crear carpeta de logs si no existe
mkdir -p logs

# Ejecutar script Python
srun pipe_clinicalExt.py
