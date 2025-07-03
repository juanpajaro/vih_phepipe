#!/usr/bin/env python3
import utils_reports
import ast
import pandas as pd
import os
import utils_general_porpose
import json
import matplotlib.pyplot as plt
#import seaborn as sns
import sys

def cargar_json_result(file_path):
    """
    Carga un archivo JSON de inconsistencias y retorna la lista y su longitud.

    Parámetros:
    - file_path_inc (str): Ruta completa al archivo de inconsistencias.

    Retorna:
    - inc (list): Lista de inconsistencias.
    - inc_len (int): Cantidad de inconsistencias.
    """
    try:
        with open(file_path, 'r') as file:
            data_j = file.read()
            data = json.loads(data_j)
        data_len = len(data)
        return data_len
    except Exception as e:
        print(f"Error al cargar inconsistencias: {e}")
        return None, 0

def cargar_y_listar_columna(ruta_csv, columna):
    """
    Carga un DataFrame desde un archivo CSV y retorna una lista con los valores de una columna específica,
    filtrando solo aquellos que terminan con '.h5'.

    Parámetros:
    - ruta_csv (str): Ruta del archivo CSV.
    - columna (str): Nombre de la columna cuyos valores se desean listar.

    Retorna:
    - lista de valores de la columna que terminan en '.h5', o None si ocurre un error.
    """
    try:
        df = pd.read_csv(ruta_csv)
        if columna in df.columns:
            lista_filtrada = [v for v in df[columna].tolist() if isinstance(v, str) and v.endswith('.h5')]
            return lista_filtrada
        else:
            print(f"La columna '{columna}' no existe en el archivo.")
            return None
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        return None
    
def obtener_dataframe_modelos(ruta_performance):
    """
    Itera sobre los modelos en el archivo de performance, carga los datos asociados a cada modelo
    (si existe el archivo correspondiente) y retorna un DataFrame con la información de cada modelo y su cantidad de pacientes.

    Retorna:
    - DataFrame con columnas: model_name, max_len, folder_recortado, file_path, patient_len
    """


    lista_modelos = cargar_y_listar_columna(ruta_performance, "model_name")
    resultados = []
    print("lista_modelos:", lista_modelos)
    
    
    for modelo in lista_modelos:
        result = utils_reports.buscar_performance_por_version(
            ruta_performance, "model_name", modelo)
        
        if result is None:
            continue

        if modelo.startswith("lstm"):

                print("result:", result)
                print(type(result))
                date = result["date"]
                f_score = result["f1_test"]
                semantic_categories = result["semantic_categories"]

                dic_vec = ast.literal_eval(result["vectorization_hyperparameters"])
                max_len = dic_vec.get("max_len", None)

                dic_hyperparameters = ast.literal_eval(result["model_hyperparameters"])
                embedding_dim = dic_hyperparameters.get("embedding_dim", None)
                block_layers = dic_hyperparameters.get("block_layers", None)
                hidden_units = dic_hyperparameters.get("hidden_units", None)
                learning_rate = dic_hyperparameters.get("learning_rate", None)
                epochs = dic_hyperparameters.get("epochs", None)
                batch_size = dic_hyperparameters.get("batch_size", None)

                loss = result["loss"]
                accuracy = result["accuracy"]
                precision_train = result["precision_train"]
                recall_train = result["recall_train"]
                precision_test = result["precision_test"]
                recall_test = result["recall_test"]
                
                folder = result['path_vectorization']
                print("folder:", folder)
                folder_recortado = utils_reports.recortar_folder(folder, "tokens", incluir_carpeta=False)
                print("folder_recortado:", folder_recortado)

                file_path = f"/home/pajaro/compu_Pipe_V3/data_transformation/data_t_{folder_recortado}.csv"
                if not os.path.isfile(file_path):
                    print("No se encontró el archivo:", file_path)
                    continue

                df = utils_reports.load_data(file_path)
                patient_len = len(df)

                file_path = f"/home/pajaro/compu_Pipe_V3/concepts/"
                file_name = f"clinical_concepts_{folder_recortado}.json"    
                data = utils_general_porpose.load_json(file_path, file_name)
                df = pd.DataFrame(data)
                p_concepts_len = len(df)

                dict_c = utils_general_porpose.load_json(file_path, "dictionary_concepts_"+folder_recortado+".json")
                dict_c_len = len(dict_c)

                file_path_inc = f"/home/pajaro/compu_Pipe_V3/concepts/inconsistencies_{folder_recortado}.json"
                if not os.path.isfile(file_path_inc):
                    print("No se encontró el archivo:", file_path_inc)
                    continue

                inc_len = cargar_json_result(file_path_inc)

                file_path_train = f"/home/pajaro/compu_Pipe_V3/train/train_{folder_recortado}.json"
                if not os.path.isfile(file_path_train):
                    print("No se encontró el archivo:", file_path_train)
                    continue

                # Cargar el archivo de entrenamiento para obtener la cantidad de pacientes
                train_len = cargar_json_result(file_path_train)

                file_path_test = f"/home/pajaro/compu_Pipe_V3/test/test_{folder_recortado}.json"
                if not os.path.isfile(file_path_test):
                    print("No se encontró el archivo:", file_path_test)
                    continue
                
                # Cargar el archivo de prueba para obtener la cantidad de pacientes
                test_len = cargar_json_result(file_path_test)

                resultados.append({
                    "date": date,
                    "max_len": max_len,
                    "p_dt_len": patient_len,
                    "semantic_categories": semantic_categories,
                    "p_concepts_len": p_concepts_len,
                    "dict_len": dict_c_len,
                    "inc_len": inc_len,
                    "train_len": train_len,
                    "test_len": test_len,
                    "model_name": modelo,
                    "embedding_dim": embedding_dim,
                    "block_layers": block_layers,
                    "hidden_units": hidden_units,
                    "learning_rate": learning_rate,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "loss": loss,
                    "accuracy": accuracy,
                    "precision_train": precision_train,
                    "recall_train": recall_train,
                    "precision_test": precision_test,
                    "recall_test": recall_test,                     
                    "f_score": f_score
                })
        elif modelo.startswith("log"):
            print(modelo)

            #elif modelo.startswith("attention"):
                #print(modelo)

    return pd.DataFrame(resultados)
    
    

def comparacion_OW_desempenio(df, patient_col="p_dt_len", fscore_col="f_score", save_fig=False, fig_name="comparacion_OW_desempenio.png"):
    """
    Dibuja una gráfica comparativa entre el tamaño de pacientes (patient_len) y el desempeño (f_score),
    ordenando de menor a mayor según el f_score.

    Parámetros:
    - df: DataFrame con los resultados.
    - patient_col: Nombre de la columna con el tamaño de pacientes.
    - fscore_col: Nombre de la columna con el f_score.
    - save_fig: Si True, guarda la figura en 'g_reports'.
    - fig_name: Nombre del archivo de la figura.
    """
    # Ordenar el DataFrame por f_score de menor a mayor
    df_sorted = df.sort_values(by=fscore_col, ascending=True)

    plt.figure(figsize=(8, 6))
    plt.scatter(df_sorted[patient_col], df_sorted[fscore_col], color='royalblue', alpha=0.7)
    plt.title("Comparación entre tamaño de pacientes y desempeño (f_score)")
    plt.xlabel("Cantidad de pacientes (patient_len)")
    plt.ylabel("Desempeño (f_score)")
    plt.tight_layout()

    for i, row in df_sorted.iterrows():
        plt.annotate(row["model_name"], (row[patient_col], row[fscore_col]), fontsize=8, alpha=0.7)

    if save_fig:
        output_dir = "g_p_reports"
        os.makedirs(output_dir, exist_ok=True)
        fig_path = os.path.join(output_dir, fig_name)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figura guardada en: {fig_path}")
    else:
        plt.show()

def comparacion_maxlen_vs_pdtlen(df, maxlen_col="max_len", pdtlen_col="p_dt_len", save_fig=False, fig_name="comparacion_maxlen_vs_pdtlen.png"):
    """
    Dibuja una gráfica de línea suavizada entre max_len y p_dt_len.

    Parámetros:
    - df: DataFrame con los resultados.
    - maxlen_col: Nombre de la columna con el valor de max_len.
    - pdtlen_col: Nombre de la columna con el valor de p_dt_len.
    - save_fig: Si True, guarda la figura en 'g_p_reports'.
    - fig_name: Nombre del archivo de la figura.
    """
    import seaborn as sns

    # Ordenar por max_len para una línea más clara
    df_sorted = df.sort_values(by=maxlen_col, ascending=True)

    plt.figure(figsize=(8, 6))
    sns.lineplot(x=df_sorted[maxlen_col], y=df_sorted[pdtlen_col], marker="o", linewidth=2, color="royalblue")
    sns.regplot(x=df_sorted[maxlen_col], y=df_sorted[pdtlen_col], scatter=False, lowess=True, color="red", line_kws={"lw":2, "alpha":0.7})

    plt.title("Relación entre max_len y cantidad de pacientes (p_dt_len)")
    plt.xlabel("max_len")
    plt.ylabel("Cantidad de pacientes (p_dt_len)")
    plt.tight_layout()

    if save_fig:
        output_dir = "g_p_reports"
        os.makedirs(output_dir, exist_ok=True)
        fig_path = os.path.join(output_dir, fig_name)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figura guardada en: {fig_path}")
    else:
        plt.show()


def cargar_dataframe(ruta_csv):
    """
    Carga un DataFrame desde un archivo CSV.

    Parámetros:
    - ruta_csv (str): Ruta al archivo CSV.

    Retorna:
    - df (DataFrame): DataFrame cargado desde el archivo, o None si ocurre un error.
    """
    
    try:
        df = pd.read_csv(ruta_csv, sep=",", encoding="utf-8")
        return df
    except Exception as e:
        print(f"Error al cargar el archivo CSV: {e}")
        return None
    
def top_20_por_f1(df, f1_col="f1_test"):
    """
    Retorna un nuevo DataFrame con los 20 registros con mayor valor en la columna f1_test.

    Parámetros:
    - df: DataFrame original.
    - f1_col: Nombre de la columna con el valor de f1_test.

    Retorna:
    - DataFrame con los 20 mejores registros según f1_test.
    """
    df_sorted = df.sort_values(by=f1_col, ascending=False)
    return df_sorted.head(20)

def grafica_top20_f1(df_top20, f1_col="f1_test", model_col="model_name", loss_col="loss", save_fig=False, fig_name="top20_f1.png"):
    """
    Genera una gráfica de línea donde el eje X es el nombre del modelo y el eje Y muestra dos líneas:
    una para f1_test y otra para loss (aproximando loss a cero si es NaN o N/a).

    Parámetros:
    - df_top20: DataFrame con los 20 mejores modelos según f1.
    - f1_col: Nombre de la columna con el valor de f1.
    - model_col: Nombre de la columna con el nombre del modelo.
    - loss_col: Nombre de la columna con el valor de loss.
    - save_fig: Si True, guarda la figura en 'g_p_reports'.
    - fig_name: Nombre del archivo de la figura.
    """

    # Ordenar para que el mejor quede a la derecha
    df_plot = df_top20.sort_values(by=f1_col, ascending=True).copy()

    # Procesar loss para asegurar que sea numérico y reemplazar NaN/N/a por 0
    def safe_loss(val):
        try:
            if pd.isna(val) or str(val).lower() in ["n/a", "nan"]:
                return 0.0
            return float(val)
        except Exception:
            return 0.0

    df_plot["loss_clean"] = df_plot[loss_col].apply(safe_loss)

    plt.figure(figsize=(12, 6))
    x_labels = df_plot[model_col].astype(str)

    plt.plot(x_labels, df_plot[f1_col], marker='o', color='royalblue', linewidth=2, label='f1_test')
    plt.plot(x_labels, df_plot["loss_clean"], marker='o', color='tomato', linewidth=2, label='loss')

    plt.xticks(rotation=90, ha='right', fontsize=14)
    plt.xlabel("Model name")
    plt.ylabel("Value")
    plt.title("Top model performance in HIV: f1 vs loss")
    plt.legend()
    plt.tight_layout()

    # Mostrar el valor de f1 y loss en cada punto
    for x, y, loss in zip(x_labels, df_plot[f1_col], df_plot["loss_clean"]):
        plt.text(x, y, f"{y:.3f}", ha='center', va='top', fontsize=12, rotation=90, color='royalblue')
        plt.text(x, loss, f"{loss:.3f}", ha='center', va='bottom', fontsize=12, rotation=90, color='tomato')

    if save_fig:
        output_dir = "g_p_reports"
        os.makedirs(output_dir, exist_ok=True)
        fig_path = os.path.join(output_dir, fig_name)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figura guardada en: {fig_path}")
    else:
        plt.show()

def main():

    if len(sys.argv) != 2:
        print("Uso: python tab_resume.py <ruta_al_archivo_performance>")
        sys.exit(1)
    #ruta_performance = "/home/pajaro/compu_Pipe_V3/performance_zine/performance_report.csv"

    ruta_performance = sys.argv[1]
    #df_modelos = obtener_dataframe_modelos(ruta_performance)
    #print(df_modelos)
    #df_modelos.to_csv("df_modelos.csv", index=False)
    #comparacion_OW_desempenio(df_modelos, save_fig=True, fig_name="comparacion_OW_desempenio.png")
    #comparacion_maxlen_vs_pdtlen(df_modelos, save_fig=True, fig_name="comparacion_maxlen_vs_pdtlen.png")
    df_modelos = cargar_dataframe(ruta_performance)
    #print(df_modelos.info())
    df_top20 = top_20_por_f1(df_modelos)
    print(df_top20["model_name"])
    grafica_top20_f1(df_top20, save_fig=True, fig_name="performance_models_hiv.png")

    
if __name__ == "__main__":
    main()
