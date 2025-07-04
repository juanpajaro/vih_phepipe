import utils_reports
import utils_general_porpose
import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
# Load the dataset
    model_version = "lstm_v490"
    current_dir = os.getcwd()
    #print("Current directory:", current_dir)    
    d_train = utils_general_porpose.load_json(current_dir, "/train/train_20250704_115703.json")
    df_train = pd.DataFrame(d_train)
    print(df_train.info())
    print(df_train["label"].value_counts())

    utils_reports.graficar_frecuencias_columna(df_train, columna="label", save_fig=True, output_dir="g_split", fig_name="frecuencias_pacientes_train_"+model_version+".png", titulo="number of patients in train "+model_version)

    d_test = utils_general_porpose.load_json(current_dir, "/test/test_20250704_115703.json")
    df_test = pd.DataFrame(d_test)
    print(df_test.info())

    utils_reports.graficar_frecuencias_columna(df_test, columna="label", save_fig=True, output_dir="g_split", fig_name="frecuencias_pacientes_test_"+model_version+".png", titulo="numero de pacientes test "+model_version)

if __name__ == "__main__":
    main()
