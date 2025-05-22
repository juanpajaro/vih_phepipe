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