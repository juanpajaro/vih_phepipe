# computational_pipe_v2
Computational Phenotyping Pipeline to identify clinical phenotypes on Obstructive Sleep Apnea

## Descripción de la Enfermedad
La apnea del sueño es un trastorno del sueño caracterizado por pausas repetidas en la respiración o períodos de respiración superficial durante el sueño. Estas pausas pueden durar desde unos pocos segundos hasta minutos y pueden ocurrir 30 veces o más por hora. La apnea del sueño puede ser obstructiva (la más común), central (menos común), o una combinación de ambas.

Apnea Obstructiva del Sueño (AOS): Ocurre cuando los músculos de la garganta se relajan excesivamente, bloqueando parcialmente o completamente las vías respiratorias superiores.

![apnea_imagen](/imagenes/Apnea_imagen_1.jpg)

## Principales Síntomas
Ronquidos fuertes
Pausas observadas en la respiración durante el sueño
Despertares abruptos con jadeo o sensación de ahogo
Sueño inquieto
Somnolencia diurna excesiva
Dificultad para concentrarse
Dolores de cabeza matutinos
Irritabilidad o cambios de humor
Boca seca o dolor de garganta al despertar

## Forma de Diagnóstico
El diagnóstico de la apnea del sueño generalmente se realiza a través de una evaluación clínica y pruebas específicas. Los métodos más comunes incluyen:

- Historial Médico y Examen Físico: El médico evalúa los síntomas y realiza un examen físico, a menudo enfocándose en las vías respiratorias, boca, nariz y garganta.
- Polisomnografía Nocturna: Es la prueba más completa y precisa, realizada en un laboratorio del sueño. Monitorea varias funciones corporales durante el sueño, incluyendo la actividad cerebral (EEG), movimientos oculares, actividad muscular, ritmo cardíaco, respiración y niveles de oxígeno en la sangre.
- Pruebas de Sueño en el Hogar: En algunos casos, se puede utilizar un equipo portátil para monitorear la respiración y otros parámetros en el hogar.

## Ejemplos de Scripts

### Setup

#### Setup de virtual environment (conda)
```bash
. scripts/venv-setup.sh
```

#### Crear ambiente 
si fue ejecutado con "Setup de virtual environment" no es necesario ejecutar este comando
```bash
. scripts/venv-create.sh
```

#### Actualizar dependencias
cuando de alguna forma cambian las dependencias, se debe ejecutar este comando, eliminara las innecesarias y actualizara las que se encuentran en el archivo environment.yml
```bash
. scripts/venv-update-deps.sh
```

### Comandos de aplicación

#### Extracción de Características
```bash
. scripts/app-concept-extraction.sh
```

#### Entrenar modelos secuenciales
```bash
. scripts/app-run-model-validation.sh
```

#### Entrenar modelos no secuenciales
```bash
. scripts/app-run-no-seq-models.sh
```

#### Análisis SHAP
El comando run_shap_analysis.py ejecuta un análisis integral del modelo, proporcionando tanto una valoración global de la importancia de las características a lo largo del dataset como un análisis local en un registro específico. Además, se dispone de opciones avanzadas para renombrar los archivos del tokenizer y las secuencias, definir rutas personalizadas para el rendimiento del modelo, especificar la carpeta que contiene los modelos e indicar si estos se encuentran en una subcarpeta.

Sintaxis:
python run_shap_analysis.py nombre-del-modelo.h5 
    [--tokenizer-name NOMBRE_TOKENIZER] 
    [--sequences-name NOMBRE_SECUENCIAS] 
    [--codes-file-url RUTA_MAPA_CSV] 
    [--performance-report-path RUTA_PERFORMANCE_REPORT] 
    [--models-folder CARPETA_MODELOS] 
    [--use-model-subfolder] 
    [--save-shap-values] 
    [--force-recalculate] 
    [--default-record INDICE] 
    [--max-features-num MAX_FEATURES]

| Argumento                              | Descripción                                                                                                             | Valor por Defecto           |
|----------------------------------------|-------------------------------------------------------------------------------------------------------------------------|-----------------------------|
| nombre-del-modelo.h5                   | Archivo del modelo a analizar (obligatorio).                                                                          | -                           |
| --tokenizer-name NOMBRE_TOKENIZER       | Nombre del archivo del objeto tokenizer para procesar el texto.                                                         | tokenizer_obj.pkl           |
| --sequences-name NOMBRE_SECUENCIAS      | Archivo que contiene las secuencias de prueba utilizadas en la evaluación del modelo.                                  | X_test_sequences.pkl        |
| --codes-file-url RUTA_MAPA_CSV          | Ruta al archivo CSV que mapea los códigos ICD10 a UMLS.                                                                 | map/map_icd10_umls.csv        |
| --performance-report-path RUTA_PERFORMANCE_REPORT| Ruta al archivo CSV que contiene el informe de desempeño del modelo.                                               | ./performance_report.csv      |
| --models-folder CARPETA_MODELOS         | Carpeta donde se almacenan los modelos.                                                                               | models                      |
| --use-model-subfolder                   | Activa el uso de una subcarpeta específica para localizar el modelo si se encuentra en un directorio dedicado.            | False                       |
| --save-shap-values                      | Guarda de forma automática los valores SHAP calculados en un archivo.                                                   | True                        |
| --force-recalculate                     | Forza la recalculación de los valores SHAP, ignorando resultados previos.                                               | False                       |
| --default-record INDICE                 | Índice del registro del dataset utilizado para el análisis local.                                                      | 3                           |
| --max-features-num MAX_FEATURES         | Número máximo de características a mostrar en las visualizaciones generadas.                                           | 100                         |

Ejemplos de Uso:

1. Ejecución básica (utiliza los valores por defecto para el registro, número de características y archivos asociados):
```bash
python run_shap_analysis.py lstm_model_v57.h5
```

2. Especificar un registro distinto para el análisis local:
```bash
python run_shap_analysis.py lstm_model_v57.h5 --default-record 10
```

3. Ajustar el número máximo de características a visualizar:
```bash
python run_shap_analysis.py lstm_model_v57.h5 --max-features-num 50
```

4. Renombrar archivos del tokenizer y de las secuencias:
```bash
python run_shap_analysis.py lstm_model_v57.h5 --tokenizer-name nuevo_tokenizer.pkl --sequences-name nuevas_secuencias.pkl
```

5. Definir rutas personalizadas para el mapeo, informe de desempeño y carpeta de modelos, además de utilizar una subcarpeta para el modelo:
```bash
python run_shap_analysis.py lstm_model_v57.h5 --codes-file-url data/map_icd10_umls_updated.csv --performance-report-path reports/performance.csv --models-folder modelos --use-model-subfolder
```

6. Forzar la recalculación de los valores SHAP y guardar explícitamente los resultados:
```bash
python run_shap_analysis.py lstm_model_v57.h5 --force-recalculate --save-shap-values
```

7. Ejemplo completo con múltiples opciones personalizadas:
```bash
python run_shap_analysis.py lstm_model_v57.h5 --tokenizer-name custom_tokenizer.pkl --sequences-name custom_sequences.pkl --codes-file-url data/custom_map.csv --performance-report-path reports/custom_performance.csv --models-folder custom_models --use-model-subfolder --default-record 5 --max-features-num 20 --force-recalculate
```
#### Ejecución total
```bash
. condor_submmit pipe_condor.sub
```

#### Ejecución por pasos del fenotipado computacional
![pasos](/imagenes/flujoDatosEnglish.svg)

- modificar el paso que se desee en el archivo pipe.sh
```bash
. vim pipe.sh
```
- volver a correr el flujo
```bash
. condor_submmit pipe_condor.sub
```