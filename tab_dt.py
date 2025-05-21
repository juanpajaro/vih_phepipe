
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt

def load_data(file_path):
    datos = pd.read_csv(file_path)
    datos = datos.rename(columns={"fecha_poli": "fecha_diagnostico"})
    return datos

def graficar_eventos_pacientes_df(df, n_pacientes=10, paciente_label_col = "label_apnea"):
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

        # Puntos clave
        eventos = {
            "fecha_diagnostico": ("blue", "Dx"),
            "last_appointment": ("black", "Últ."),
        }

        for evento, (color, _) in eventos.items():
            fecha = row[evento]
            ax.plot(fecha, y, 'o', color=color)
            ax.text(fecha, y + 0.1, fecha.strftime("%Y-%m-%d"), fontsize=8, ha='center', va='bottom', rotation=45)

        # Ventana de observación
        ax.plot([row["prediction_window_start"], row["end_observation_window"]], [y, y], color="red", linewidth=6)

        # Evento futuro (dicotómico)
        if "evento_futuro" in df.columns:
            color_dicotomica = "green" if row["evento_futuro"] == 1 else "red"
            punto_evento = row["last_appointment"] + pd.Timedelta(days=30)
            ax.plot(punto_evento, y, 'o', color=color_dicotomica)
            ax.text(punto_evento, y + 0.1, str(row["evento_futuro"]), fontsize=8, ha='center', va='bottom')

    # Etiquetas de eje Y con nombres personalizados
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels(df_plot[paciente_label_col] if paciente_label_col in df.columns else [f"Paciente {i+1}" for i in range(len(df_plot))])

    # Eje X
    ax.xaxis_date()
    fig.autofmt_xdate()

    # Estética
    ax.set_title("Línea de tiempo de eventos por paciente")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Paciente")

    plt.tight_layout()   
    #plt.savefig("grafica_"+version+"_"+str(i)+".png", dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # Cargar datos

    file_path = "/home/pajaro/compu_Pipe_V3/data_transformation/data_t_20250520_053753.csv"
    df = load_data(file_path)
    df.info()

    # Graficar eventos
    graficar_eventos_pacientes_df(df, n_pacientes=5)

if __name__ == "__main__":
    main()
