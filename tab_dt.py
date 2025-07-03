import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime as dt
import os  # <-- Añade esta línea

def load_data(file_path):
    datos = pd.read_csv(file_path)
    datos = datos.rename(columns={"Fecha_Dx_Prueba": "fecha_diagnostico"})
    return datos

def graficar_eventos_pacientes_df(df, n_pacientes=10, paciente_label_col="label_apnea", save_fig=False, fig_name="grafica_eventos.png", model_version=""):
    """
    Grafica eventos temporales para pacientes desde un DataFrame.
    
    Requiere las siguientes columnas:
    - 'fecha_diagnostico'
    - 'last_appointment'
    - 'prediction_window_start'
    - 'end_observation_window'
    - 'evento_futuro' (0 o 1)
    - paciente_label_col: nombre de la columna con el nombre del paciente

    Parámetros:
    - df: pandas DataFrame con las columnas anteriores.
    - n_pacientes: número de pacientes a graficar.
    - paciente_label_col: nombre de la columna que contiene el nombre/etiqueta del paciente
    - save_fig: si True, guarda la figura en la carpeta 'g_reports'
    - fig_name: nombre del archivo de la figura
    """

    # Asegurar que las fechas son datetime
    date_cols = ["fecha_diagnostico", "last_appointment", "prediction_window_start", "end_observation_window"]
    for col in date_cols:
        if df[col].dtype == "O" or not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col])

    # Cortar a los primeros n pacientes
    df_plot = df.iloc[:n_pacientes].copy()
    df_plot = df_plot.reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 1 + n_pacientes))

    for idx, row in df_plot.iterrows():
        y = idx
        nombre_paciente = row[paciente_label_col] if paciente_label_col in df.columns else f"Paciente {idx + 1}"

        # Línea gris de observación
        ax.plot([row["end_observation_window"], row["last_appointment"]], [y, y], color="gray", linewidth=2)

        # Puntos clave y fechas verticales encima de los puntos
        eventos = {
            "fecha_diagnostico": ("blue", "Dx"),
            "last_appointment": ("black", "Últ."),
            "prediction_window_start": ("red", "Inicio ventana"),
            "end_observation_window": ("red", "Fin ventana"),
        }

        for evento, (color, _) in eventos.items():
            fecha = row[evento]
            ax.plot(fecha, y, 'o', color=color, markersize=8)
            ax.text(fecha, y + 0.18, fecha.strftime("%Y-%m-%d"),
                    fontsize=8, ha='center', va='bottom', rotation=90, color=color, fontweight='bold')

        # Ventana de observación (línea roja)
        ax.plot([row["prediction_window_start"], row["end_observation_window"]], [y, y], color="red", linewidth=6)

        # Evento futuro (dicotómico)
        if "evento_futuro" in df.columns:
            color_dicotomica = "green" if row["evento_futuro"] == 1 else "red"
            punto_evento = row["last_appointment"] + pd.Timedelta(days=30)
            ax.plot(punto_evento, y, 'o', color=color_dicotomica)
            ax.text(punto_evento, y + 0.18, str(row["evento_futuro"]), fontsize=8, ha='center', va='bottom', rotation=90)

    # Etiquetas de eje Y con nombres personalizados
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels(df_plot[paciente_label_col] if paciente_label_col in df.columns else [f"Paciente {i+1}" for i in range(len(df_plot))])

    # Eje X
    ax.xaxis_date()
    fig.autofmt_xdate()

    # Estética
    ax.set_title("timeline per patients {}".format(model_version))
    ax.set_xlabel("Date")
    ax.set_ylabel("Label patient")

    plt.tight_layout()
    
    if save_fig:
        output_dir = "g_reports"
        os.makedirs(output_dir, exist_ok=True)
        fig_path = os.path.join(output_dir, fig_name)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figura guardada en: {fig_path}")
    else:
        plt.show()

def graficar_frecuencias_label(df, columna, save_fig=False, fig_name="frecuencias_label_apnea.png", model_version=""):
    """
    Cuenta los valores únicos de una columna y dibuja una gráfica de frecuencias, 
    agregando una leyenda solo con el valor total.

    Parámetros:
    - df: DataFrame de pandas.
    - columna: nombre de la columna a analizar.
    - save_fig: si True, guarda la figura en la carpeta 'g_reports'.
    - fig_name: nombre del archivo de la figura.
    """
    conteo = df[columna].value_counts()
    total = len(df)

    plt.figure(figsize=(10, 5))
    ax = conteo.plot(kind='bar')
    plt.title("Frequency of Patients with/without Apnea {}".format(model_version))
    plt.xlabel(columna)
    plt.ylabel("Frequency")
    plt.tight_layout()

    # Agregar los números encima de las barras
    for p in ax.patches:
        ax.annotate(str(int(p.get_height())),
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Agregar solo el total en la leyenda
    legend_text = f"Total: {total}"
    plt.gca().legend([legend_text], loc='upper right', frameon=True)

    if save_fig:
        output_dir = "g_reports"
        os.makedirs(output_dir, exist_ok=True)
        fig_path = os.path.join(output_dir, fig_name)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figura guardada en: {fig_path}")
    else:
        plt.show()

def distribucion_por_sexo(df, column_to_split="label_apnea", column_to_plot="Sex", 
                          title1="Sex distribution in patients with apnea = 1", 
                          title2="Sex distribution in patients without apnea = 0", 
                          save_fig=False, fig_name="distribucion_por_sexo.png", model_version=""):
    """
    Separa un DataFrame según los valores únicos de una columna y genera dos gráficos comparativos,
    mostrando los valores sobre las barras.

    Args:
        df (pd.DataFrame): El DataFrame de entrada.
        column_to_split (str): La columna según la cual se separará el DataFrame.
        column_to_plot (str): La columna que se graficará.
        title1 (str): Título del primer gráfico.
        title2 (str): Título del segundo gráfico.
        save_fig (bool): Si True, guarda la figura en la carpeta 'g_reports'.
        fig_name (str): Nombre del archivo de la figura.
    """
    unique_values = df[column_to_split].unique()
    if len(unique_values) < 2:
        print("La columna para dividir debe tener al menos dos valores únicos.")
        return

    df_1 = df[df[column_to_split] == unique_values[0]]
    df_2 = df[df[column_to_split] == unique_values[1]]

    plt.figure(figsize=(14, 6))

    # Primer gráfico
    plt.subplot(1, 2, 1)
    ax1 = sns.countplot(x=column_to_plot, data=df_1)
    plt.title(title1+" {}".format(model_version))
    plt.xlabel(column_to_plot)
    plt.ylabel("Count")
    for p in ax1.patches:
        ax1.annotate(f'{int(p.get_height())}', 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 5), 
                     textcoords='offset points')

    # Segundo gráfico
    plt.subplot(1, 2, 2)
    ax2 = sns.countplot(x=column_to_plot, data=df_2)
    plt.title(title2+" {}".format(model_version))
    plt.xlabel(column_to_plot)
    plt.ylabel("Conunt")
    for p in ax2.patches:
        ax2.annotate(f'{int(p.get_height())}', 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 5), 
                     textcoords='offset points')

    plt.tight_layout()
    if save_fig:
        output_dir = "g_reports"
        os.makedirs(output_dir, exist_ok=True)
        fig_path = os.path.join(output_dir, fig_name)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figura guardada en: {fig_path}")
    else:
        plt.show()

def distribucion_por_edad(df, column_to_split="label_apnea", column_to_plot="edad_poli", bins=5,
                          title1="Distribución de Edad con Apnea = 1",
                          title2="Distribución de Edad sin Apnea = 0",
                          save_fig=False, fig_name="distribucion_por_edad.png"):
    """
    Separa un DataFrame según los valores únicos de una columna, agrupa los valores continuos
    en intervalos (bins) y genera dos gráficos comparativos.

    Args:
        df (pd.DataFrame): El DataFrame de entrada.
        column_to_split (str): La columna según la cual se separará el DataFrame.
        column_to_plot (str): La columna continua que se graficará.
        bins (int): Número de intervalos (bins) para agrupar los valores continuos.
        title1 (str): Título del primer gráfico.
        title2 (str): Título del segundo gráfico.
        save_fig (bool): Si True, guarda la figura en la carpeta 'g_reports'.
        fig_name (str): Nombre del archivo de la figura.
    """
    unique_values = df[column_to_split].unique()
    if len(unique_values) < 2:
        print("La columna para dividir debe tener al menos dos valores únicos.")
        return

    df_1 = df[df[column_to_split] == unique_values[0]].copy()
    df_2 = df[df[column_to_split] == unique_values[1]].copy()

    df_1['binned'] = pd.cut(df_1[column_to_plot], bins=bins)
    df_2['binned'] = pd.cut(df_2[column_to_plot], bins=bins)

    plt.figure(figsize=(14, 6))

    # Primer gráfico
    plt.subplot(1, 2, 1)
    ax1 = sns.countplot(x='binned', data=df_1, order=sorted(df_1['binned'].unique()))
    plt.title(title1)
    plt.xlabel(f"{column_to_plot} (agrupado)")
    plt.ylabel("Conteo")
    plt.xticks(rotation=45)
    for p in ax1.patches:
        ax1.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 5),
                     textcoords='offset points')

    # Segundo gráfico
    plt.subplot(1, 2, 2)
    ax2 = sns.countplot(x='binned', data=df_2, order=sorted(df_2['binned'].unique()))
    plt.title(title2)
    plt.xlabel(f"{column_to_plot} (agrupado)")
    plt.ylabel("Conteo")
    plt.xticks(rotation=45)
    for p in ax2.patches:
        ax2.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 5),
                     textcoords='offset points')

    plt.tight_layout()
    if save_fig:
        output_dir = "g_reports"
        os.makedirs(output_dir, exist_ok=True)
        fig_path = os.path.join(output_dir, fig_name)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figura guardada en: {fig_path}")
    else:
        plt.show()

def main():
    # Cargar datos

    file_path = "/home/pajaro/vih_phepipe/data_transformation/data_t_20250520_053753.csv"
    df = load_data(file_path)
    #df.info()
    df_explo = df[["fecha_diagnostico", "last_appointment", "prediction_window_start", "end_observation_window"]]
    print(df_explo.head())

    model_version = "lstm_v80"

    # Graficar eventos y guardar figura
    #graficar_eventos_pacientes_df(df, n_pacientes=10, save_fig=True, fig_name="eventos_paciente_{}.png".format(model_version), model_version=model_version)

    # Graficar frecuencias con leyenda y guardar figura
    #graficar_frecuencias_label(df, "label", save_fig=True, fig_name="frecuencias_label_apnea_{}.png".format(model_version), model_version=model_version)

    # Graficar distribución por sexo y guardar figura
    #distribucion_por_sexo(df, column_to_split="label", column_to_plot="Sexo", save_fig=True, fig_name="distribucion_por_sexo_{}.png".format(model_version), model_version=model_version)

    # Graficar distribución por edad y guardar figura
    #distribucion_por_edad(df, column_to_split="label_apnea", column_to_plot="edad_poli", bins=5, save_fig=True, fig_name="distribucion_por_edad_{}.png".format(model_version))

if __name__ == "__main__":
    main()