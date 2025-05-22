import pandas as pd
import os
import matplotlib.pyplot as plt

def load_data(file_path):
    datos = pd.read_csv(file_path)
    datos = datos.rename(columns={"fecha_poli": "fecha_diagnostico"})
    return datos

def graficar_frecuencias_columna(df, columna, save_fig=False, output_dir="g_reports", fig_name="frecuencias_columna.png", titulo=None):
    """
    Cuenta los valores únicos de una columna y dibuja una gráfica de frecuencias.

    Parámetros:
    - df: DataFrame de pandas.
    - columna: nombre de la columna a analizar.
    - save_fig: si True, guarda la figura en la carpeta indicada por output_dir.
    - output_dir: carpeta donde se guardará la figura.
    - fig_name: nombre del archivo de la figura.
    - titulo: título de la figura (opcional).
    """
    conteo = df[columna].value_counts()
    plt.figure(figsize=(10, 5))
    ax = conteo.plot(kind='bar')
    if titulo is None:
        titulo = f"Frecuencia de valores en '{columna}'"
    plt.title(titulo)
    plt.xlabel(columna)
    plt.ylabel("Frecuencia")
    plt.tight_layout()

    # Agregar los números encima de las barras
    for p in ax.patches:
        ax.annotate(str(int(p.get_height())),
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Agregar leyenda en la parte superior derecha con el total de registros
    total = conteo.sum()
    legend_text = f"Total: {total}"
    plt.legend([legend_text], loc='upper right', frameon=True)

    if save_fig:
        os.makedirs(output_dir, exist_ok=True)
        fig_path = os.path.join(output_dir, fig_name)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figura guardada en: {fig_path}")
    else:
        plt.show()

def buscar_performance_por_version(
    ruta_csv, columna_busqueda, valor_busqueda
):
    """
    Busca una fila en un CSV que coincida con un valor en una columna de búsqueda,
    y retorna los valores de varias columnas de interés.

    Parámetros:
    - ruta_csv (str): Ruta del archivo CSV.
    - columna_busqueda (str): Nombre de la columna donde se buscará el valor.
    - valor_busqueda (str): Valor que se desea buscar.

    Retorna:
    dict: Diccionario con los valores de las columnas de interés o None si no se encuentra.
    """
    columnas_interes = [
        'date', 'semantic_categories', 'num_classes', 'vectorize_technique', 'vectorization_hyperparameters',
        'path_vectorization', 'model_name', 'model_hyperparameters', 'accuracy', 'loss',
        'precision_train', 'recall_train', 'f1_train', 'precision_test', 'recall_test', 'f1_test'
    ]
    try:
        df = pd.read_csv(ruta_csv)
        #print("df.columns:", df.columns)
        #print("df.shape:", df.shape)
        fila = df[df[columna_busqueda] == valor_busqueda]
        if not fila.empty:
            resultado = {col: fila.iloc[0][col] if col in fila.columns else None for col in columnas_interes}
            return resultado
        else:
            print("Valor no encontrado.")
            return None
    except FileNotFoundError:
        print("Archivo no encontrado.")
    except KeyError as e:
        print(f"Columna no encontrada: {e}")
    except Exception as e:
        print(f"Ocurrió un error: {e}")
    return None

def recortar_folder(ruta_completa, nombre_carpeta, incluir_carpeta=True):
    """
    Recorta una ruta a partir del nombre de una carpeta.

    Parámetros:
    ruta_completa (str): Ruta completa del archivo o directorio.
    nombre_carpeta (str): Nombre de la carpeta que se quiere identificar.
    incluir_carpeta (bool): Si True, incluye la carpeta en el resultado; si False, la excluye.

    Retorna:
    str: Ruta recortada o None si la carpeta no se encuentra.
    """
    partes = ruta_completa.split(os.sep)
    try:
        indice = partes.index(nombre_carpeta)
        if incluir_carpeta:
            return os.sep.join(partes[:indice + 1])
        else:
            return os.sep.join(partes[indice + 1:])
    except ValueError:
        print("Carpeta no encontrada en la ruta.")
        return None

def load_data(file_path):
    datos = pd.read_csv(file_path)
    datos = datos.rename(columns={"fecha_poli": "fecha_diagnostico"})
    return datos