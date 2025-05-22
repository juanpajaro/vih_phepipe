import utils_reports
import ast
import pandas as pd
import os

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

    for modelo in lista_modelos:
        result = utils_reports.buscar_performance_por_version(
            ruta_performance, "model_name", modelo)
        
        if result is None:
            continue

        print("result:", result)
        print(type(result))
        date = result["date"]
        f_score = result["f1_test"]

        dic_vec = ast.literal_eval(result["vectorization_hyperparameters"])
        max_len = dic_vec.get("max_len", None)
        
        folder = result['path_vectorization']
        folder_recortado = utils_reports.recortar_folder(folder, "tokens", incluir_carpeta=False)

        file_path = f"/home/pajaro/compu_Pipe_V3/data_transformation/data_t_{folder_recortado}.csv"
        if not os.path.isfile(file_path):
            print("No se encontró el archivo:", file_path)
            continue

        df = utils_reports.load_data(file_path)
        patient_len = len(df)

        resultados.append({
            "model_name": modelo,
            "max_len": max_len,            
            "patient_len": patient_len,
            "date": date,
            "f_score": f_score
        })

    return pd.DataFrame(resultados)

def main():
    ruta_performance = "/home/pajaro/compu_Pipe_V3/performance_zine/performance_report.csv"
    df_modelos = obtener_dataframe_modelos(ruta_performance)
    print(df_modelos)
    df_modelos.to_csv("df_modelos.csv", index=False)
    
if __name__ == "__main__":
    main()
