#!/bin/bash
#SBATCH --job-name=concept_extraction
#SBATCH --output=logs/out_%j.txt
#SBATCH --error=logs/err_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8     # Ajusta este valor si es necesario
##SBATCH --time=00:30:00
#SBATCH --mem=8G

# Creamos vaiables de entorno
#export PATH="/zine/apps/anaconda_salud/bin:$PATH"

# Iniciamos CONDA
#conda init

source /zine/apps/anaconda_salud/etc/profile.d/conda.sh
conda activate 1cphe

echo "El ambiente activado es: "$CONDA_DEFAULT_ENV

# Crear carpeta de logs si no existe
mkdir -p logs

# Ejecutar script Python
srun python /zine/data/salud/compu_Pipe_V3/pipe_clinicalExt.py
