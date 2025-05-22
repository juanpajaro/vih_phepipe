from collections import Counter
import utils_reports
import utils_general_porpose
import os
import pandas as pd
import matplotlib.pyplot as plt

def count_top_words(datasets, top_n=20, exclude_words=None):
    """
    Counts the frequency of words across a series of datasets and returns the top N most frequent words.

    Args:
        datasets (list): A list of strings, where each string represents a dataset.
        top_n (int): The number of top frequent words to return.
        exclude_words (set): A set of words to exclude from the count.

    Returns:
        list: A list of tuples with the top N most frequent words and their counts.
    """
    if exclude_words is None:
        exclude_words = set()

    word_counter = Counter()

    for dataset in datasets:
        # Split the dataset into words and normalize to lowercase
        words = dataset.lower().split()
        # Filter out excluded words
        filtered_words = [word for word in words if word not in exclude_words]
        # Update the word counter with the current dataset
        word_counter.update(filtered_words)

    return word_counter.most_common(top_n)

def dividir_pos_neg_df(df, label_col="label"):
    """
    Divide el DataFrame en dos: negativos y positivos según la columna de etiqueta.
    Retorna ambos DataFrames.

    Parámetros:
    - df: DataFrame de pandas.
    - label_col: nombre de la columna de etiquetas (por defecto 'label').

    Returns:
    - neg_apnea: DataFrame con casos negativos.
    - pos_apnea: DataFrame con casos positivos.
    """
    neg_apnea = df[df[label_col] == 0]
    pos_apnea = df[df[label_col] == 1]
    print(f"Negativos apnea: {neg_apnea.shape}")
    print(f"Positivos apnea: {pos_apnea.shape}")
    return neg_apnea, pos_apnea

def imprimir_top_words(df, column="entities", top_n=20, exclude_words=None, save_fig=False, fig_name="top_words.png", tipo=""):
    """
    Imprime las top N palabras más frecuentes en una columna de texto de un DataFrame y guarda la gráfica si se indica.

    Parámetros:
    - df: DataFrame de pandas.
    - column: nombre de la columna de texto.
    - top_n: número de palabras más frecuentes a mostrar.
    - exclude_words: conjunto de palabras a excluir.
    - save_fig: si True, guarda la gráfica en la carpeta 'g_reports'.
    - fig_name: nombre del archivo de la figura.
    - tipo: texto adicional para indicar si es 'pos' o 'neg' en el título de la gráfica.
    """
    if exclude_words is None:
        exclude_words = {"enfermedad", "paciente"}
    top_words = count_top_words(df[column], top_n=top_n, exclude_words=exclude_words)
    print(f"Top {top_n} conceptos in the '{column}' column:")
    for word, count in top_words:
        print(f"{word}: {count}")

    # Ordenar palabras y conteos de menor a mayor para la gráfica horizontal
    words, counts = zip(*top_words)
    counts, words = zip(*sorted(zip(counts, words), reverse=False))

    plt.figure(figsize=(12, 6))
    bars = plt.barh(words, counts)
    titulo = f"Top {top_n} conceptos clinicos más frecuentes en '{column}'"
    if tipo:
        titulo += f" ({tipo})"
    plt.title(titulo)
    plt.xlabel("Frecuencia")
    plt.ylabel("Conceptos")
    plt.tight_layout()

    # Agregar los valores al final de cada barra
    for bar in bars:
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 str(int(bar.get_width())),
                 va='center', ha='left', fontsize=10, fontweight='bold')

    if save_fig:
        output_dir = "g_consistency"
        os.makedirs(output_dir, exist_ok=True)
        fig_path = os.path.join(output_dir, fig_name)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figura guardada en: {fig_path}")
    else:
        plt.show()


def main():
    # Load the dataset
    current_dir = os.getcwd()
    #print("Current directory:", current_dir)
    #data = utils_general_porpose.load_json(current_dir, "/concepts/clinical_concepts_20250404_171428.json")
    data = utils_general_porpose.load_json(current_dir, "/concepts/clinical_concepts_20250520_053753.json")
    df = pd.DataFrame(data)
    print(df.info())

    neg, pos = dividir_pos_neg_df(df, label_col="label")
    tipo = "pos"
    imprimir_top_words(pos, column="entities", top_n=20, exclude_words={"enfermedad", "paciente"}, save_fig=True, fig_name="top_concepts_"+tipo+".png", tipo="pos")
    tipo = "neg"
    imprimir_top_words(neg, column="entities", top_n=20, exclude_words={"enfermedad", "paciente"}, save_fig=True, fig_name="top_concepts_"+tipo+".png", tipo="neg")

    incon = utils_general_porpose.load_json(current_dir, "/concepts/inconsistencies_20250520_053753.json")
    incon = pd.DataFrame(incon)
    print(incon.info())

    utils_reports.graficar_frecuencias_columna(incon, columna="label", save_fig=True, output_dir="g_consistency", fig_name="frecuencias_pacientes_inconsistencia.png", titulo="numero de pacientes con diagnostico Apnea antes de la polisomnografia")
    utils_reports.graficar_frecuencias_columna(df, columna="label", save_fig=True, output_dir="g_consistency", fig_name="frecuencias_pacientes_sin_inconsistencias.png", titulo="numero de pacientes con diagnostico Apnea antes de la polisomnografia sin inconsistencias")

    imprimir_top_words(incon, column="entities", top_n=20, exclude_words={"enfermedad", "paciente"}, save_fig=True, fig_name="top_concepts_inconsistencias.png", tipo="inconsistencias")

if __name__ == "__main__":
    main()