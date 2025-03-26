#!/bin/bash
echo "inicio Bash de activacion de CONDA"

# Creamos variables de entorno
export PATH="/zine/apps/anaconda_salud/bin:$PATH"
echo "creamos variable entorno para CONDS"

# Inciamos CONDA
conda init
echo " "
echo "Iniciamos CONDA"

# Cargamos el nuevo archivo bashrc
source /zine/apps/anaconda_salud/etc/profile.d/conda.sh
echo "Reiniciamos sistema equivalente al /.bashrc"
echo " "

# Activamos el entorno
conda activate extraction_venv
echo "**********************************************"
echo "Activamos el ambiente:"
echo "El ambiente activado es: "$CONDA_DEFAULT_ENV
echo "**********************************************"
echo " "
#concept extraction
echo "**********************************************"
echo "corriendo concepts extraction"
python3 conceptse.py
echo "**********************************************"
echo "concept extraction finished"


