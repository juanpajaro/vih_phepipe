#!/bin/bash
echo "inicio Bash de activacion de CONDA"

# Creamos variables de entorno
#export PATH=/mnt/disco_aux/paula/conda/anaconda3/bin:$PATH
#export PATH="/data/software/CONDA/bin:$PATH"
export PATH="/zine/apps/anaconda_salud/bin:$PATH"
echo "creamos variable entonor para CONDS"

# Inciamos CONDA
conda init
echo " "
echo "Iniciamos CONDA"
 # Cargamos el nuevo archivo bashrc
#source ~/.bashrc
#source /mnt/disco_aux/paula/conda/anaconda3/etc/profile.d/conda.sh
#source /data/software/CONDA/etc/profile.d/conda.sh
source /zine/apps/anaconda_salud/etc/profile.d/conda.sh
echo "Reiniciamos sistema equivalente al /.bashrc"
echo " "
# Configuramos Conda para no iniciar con env=base
#conda config --set auto_activate_base

# Activamos el entorno
#conda activate renv
#conda activate venvPhePi
conda activate phepi_v3
#/mnt/disco_aux/paula/conda/anaconda3/bin/conda activate renv
#/mnt/disco_aux/paula/conda/anaconda3/envs
#conda list
echo "**********************************************"
echo "Activamos el ambiente:"
echo "El ambiente activado es: "$CONDA_DEFAULT_ENV
echo "**********************************************"
echo " "
#ejecutar
echo "**********************************************"
echo "corriendo dataset_transformation..."
echo "**********************************************"
#python3 dataset_transformation.py
echo "**********************************************"
echo "corriendo datset_split..."
#python3 utils_class_split.py
echo "**********************************************"
echo "corriendo concepts extraction"
#python3 conceptse.py
echo "**********************************************"
echo "corriendo representation-learning and training-evaluation models"
#python3 class_pipe.py
echo "**********************************************"
echo "corriendo feature atributtion"
#python3 run_shap_analysis.py lstm_model_v17.h5
