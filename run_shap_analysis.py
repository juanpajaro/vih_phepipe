import os
import sys
import warnings

import click

# Configuring warnings to minimize output noise
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

# Adding the './src' directory to the system path to ensure project modules can be imported
sys.path.append(os.path.join(os.getcwd(), "src"))
click.echo("INFO: SHAP Analysis (more info in README)")

# Default paths and filenames for required assets
default_performance_report_path = "./performance_report.csv"
default_codes_file_url = "map/map_icd10_umls.csv"
default_models_folder = "models"
default_tokenizer_name = "tokenizer_obj.pkl"
default_sequences_name = "X_test_sequences.pkl"

ERROR_MESSAGE_TEMPLATE = "ERROR: {}"

# This command performs a comprehensive analysis using SHAP (SHapley Additive exPlanations) to explain the model's predictions.
# The process involves the following steps:

# 1. Validate the provided model name and confirm the existence of essential assets such as the tokenizer and test sequences.
# 2. Load the performance report to retrieve the corresponding model metadata.
# 3. Initialize and build the model based on the retrieved configuration and assets.
# 4. Instantiate a SHAP explainer for the model.
# 5. Calculate SHAP values, with an option to force re-calculation if required.
# 6. Generate and save both global and local analysis reports, including:
#    - Global feature importance visualization.
#    - A violin plot showing the distribution of SHAP values.
#    - Local analysis for a specific record.

# Parameters:
#     model_name (str): The filename of the model to be analyzed.
#     tokenizer_name (str): The filename of the tokenizer object used for text processing.
#     sequences_name (str): The filename containing the test sequences for model evaluation.
#     codes_file_url (Path): Path to the CSV file mapping ICD10 codes to UMLS codes.
#     performance_report_path (str): Path to the CSV file that contains the performance report.
#     models_folder (str): Directory where the model files are stored.
#     use_model_subfolder (bool): Flag indicating whether the model is stored in a dedicated subfolder.
#     save_shap_values (bool): Flag to enable automatic saving of the calculated SHAP values.
#     force_recalculate (bool): Flag to force the re-calculation of SHAP values, disregarding previously computed results.
#     max_features_num (int): Number of top features to display in the analysis report.
#     default_record (int): Index of the record to be used for local feature importance analysis.
#     no_reverse_sequences (bool): If True, sequences will not be reversed. Default is False, meaning sequences are reversed by default.
#     clean_sequences (bool): If True, empty sequences will be cleaned. Default is True, meaning empty sequences are cleaned by default.

# Examples:
#     python run_shap_analysis.py my_model.h5 --default-record 5 --max-features-num 15
#     python run_shap_analysis.py my_model.h5 --clean-sequences --no-reverse-sequences

# This command will execute SHAP analysis on 'my_model.h5', performing local analysis on record number 5 and displaying the top 15 important features.
#
@click.command("app:shap:analysis")
@click.argument("model_name", type=click.STRING)
@click.option("-t", "--tokenizer-name", type=click.STRING, default=default_tokenizer_name, help="Tokenizer file name.")
@click.option(
    "-se", "--sequences-name", type=click.STRING, default=default_sequences_name, help="X test Sequences file name."
)
@click.option(
    "-c",
    "--codes-file-url",
    type=click.Path(),
    default=default_codes_file_url,
    help="Path to the ICD10 to UMLS mapping CSV file.",
)
@click.option(
    "-p",
    "--performance-report-path",
    type=click.Path(),
    default=default_performance_report_path,
    help="Path to the performance report CSV file.",
)
@click.option(
    "-mo",
    "--models-folder",
    type=click.Path(),
    default=default_models_folder,
    help="Path to the models folder. This folder contains the model files used for analysis.",
)
@click.option(
    "-umf",
    "--use-model-subfolder",
    is_flag=True,
    flag_value=True,
    default=False,
    help="Enable using a model subfolder (set this flag if the model file is located in a dedicated subfolder).",
)
@click.option(
    "-s",
    "--save-shap-values",
    is_flag=True,
    flag_value=True,
    default=True,
    help="Automatically save calculated SHAP values to a file.",
)
@click.option(
    "-f",
    "--force-recalculate",
    is_flag=True,
    flag_value=True,
    default=False,
    help="If activated, forces the recalculation of SHAP values, ignoring previous results.",
)
@click.option(
    "-num",
    "--max-features-num",
    type=click.INT,
    default=20,
    help="Number of features to display in the report.",
)
@click.option(
    "--default-record",
    type=click.INT,
    default=3,
    help="Record number to analyze for local feature importance.",
)
@click.option(
    "-nrs",
    "--no-reverse-sequences",
    is_flag=True,
    flag_value=True,
    default=False,
    help="Do not reverse sequences. (default: reverse sequences)",
)
@click.option(
    "-cs",
    "--clean-sequences",
    is_flag=True,
    flag_value=True,
    default=True,
    help="Clean empty sequences. (default: True)",
)
def run_shap_analysis(
    model_name,
    tokenizer_name,
    sequences_name,
    codes_file_url,
    performance_report_path,
    models_folder,
    use_model_subfolder,
    save_shap_values,
    force_recalculate,
    max_features_num,
    default_record,
    no_reverse_sequences,
    clean_sequences,
):
    """
    Execute SHAP analysis on a specified machine learning model.

    This command performs a comprehensive analysis using SHAP (SHapley Additive exPlanations) to explain the model's predictions.
    The process involves the following steps:

    1. Validate the provided model name and confirm the existence of essential assets such as the tokenizer and test sequences.
    2. Load the performance report to retrieve the corresponding model metadata.
    3. Initialize and build the model based on the retrieved configuration and assets.
    4. Instantiate a SHAP explainer for the model.
    5. Calculate SHAP values, with an option to force re-calculation if required.
    6. Generate and save both global and local analysis reports, including:
       - Global feature importance visualization.
       - A violin plot showing the distribution of SHAP values.
       - Local analysis for a specific record.

    Parameters:
        model_name (str): The filename of the model to be analyzed.
        tokenizer_name (str): The filename of the tokenizer object used for text processing.
        sequences_name (str): The filename containing the test sequences for model evaluation.
        codes_file_url (Path): Path to the CSV file mapping ICD10 codes to UMLS codes.
        performance_report_path (str): Path to the CSV file that contains the performance report.
        models_folder (str): Directory where the model files are stored.
        use_model_subfolder (bool): Flag indicating whether the model is stored in a dedicated subfolder.
        save_shap_values (bool): Flag to enable automatic saving of the calculated SHAP values.
        force_recalculate (bool): Flag to force the re-calculation of SHAP values, disregarding previously computed results.
        max_features_num (int): Number of top features to display in the analysis report.
        default_record (int): Index of the record to be used for local feature importance analysis.
        no_reverse_sequences (bool): If True, sequences will not be reversed. Default is False, meaning sequences are reversed by default.
        clean_sequences (bool): If True, empty sequences will be cleaned. Default is True, meaning empty sequences are cleaned by default.

    Examples:
        python run_shap_analysis.py my_model.h5 --default-record 5 --max-features-num 15
        python run_shap_analysis.py my_model.h5 --clean-sequences --no-reverse-sequences

    This command will execute SHAP analysis on 'my_model.h5', performing local analysis on record number 5 and displaying the top 15 important features.
    """
    if not model_name:
        click.echo("ERROR: you must specify a model name")
        exit()

    click.echo(f"INFO: analysis will be executed for record {default_record}")
    click.echo(f"INFO: showing the first {max_features_num} most important features")

    from pathlib import Path
    import pandas as pd

    df_performance_report = pd.read_csv(performance_report_path)
    record = df_performance_report[df_performance_report["model_name"] == model_name]

    if record.shape[0] == 0:
        click.echo(f"ERROR: No model exists with name '{model_name}'")
        exit()

    record = record.to_dict(orient="records")[0]
    folder = os.path.join(models_folder, Path(record["model_name"]).stem if use_model_subfolder else "")

    report_folder = os.path.join(models_folder, Path(record["model_name"]).stem)

    os.makedirs(report_folder, exist_ok=True)
    click.echo(f"INFO: the report will be dropped in: {report_folder}")

    assets = [tokenizer_name, sequences_name]
    assets = [os.path.join(record["path_vectorization"], asset) for asset in assets]

    for asset in assets:
        if not os.path.exists(asset):
            click.echo("ERROR: not found asset: '{}'".format(asset))
            exit()

    click.echo()
    click.echo("INFO: initializing libraries...")

    from app.model.aix_wrapper import AIXWrapper
    from app.model.errors import AIXModelNotFoundError
    from app.model.model_wrapper import ModelWrapper

    click.echo("INFO: preparing SHAP analysis")
    click.echo('INFO: instantiating model "{}"'.format(model_name))

    model = ModelWrapper(
        name=model_name,
        folder=folder,
        model_path=os.path.join(folder, model_name),
        vectorizer_path=assets[0],
        vectorizer_technique=record["vectorize_technique"],
        vectorization_hyperparameters=record["vectorization_hyperparameters"],
        model_hyperparameters=record["model_hyperparameters"],
        sequences_path=assets[1],
        reverse_sequences=not no_reverse_sequences,
        clean_empty_sequences=clean_sequences,
    )

    try:
        model.build_model()
    except AIXModelNotFoundError as e:
        click.echo(ERROR_MESSAGE_TEMPLATE.format(e.strerror))
        exit()

    except Exception as e:
        click.echo(ERROR_MESSAGE_TEMPLATE.format(e.args[0]))
        exit()

    click.echo("INFO: instantiating SHAP explainer")
    aix = AIXWrapper(model=model, folder=report_folder, translator_file=codes_file_url)

    click.echo("INFO: calculating SHAP values")
    click.echo("INFO: model sequences shape {}".format(model.sequences.shape))

    try:
        aix.calculate(save_values=save_shap_values, force=force_recalculate)
    except Exception as e:
        click.echo(ERROR_MESSAGE_TEMPLATE.format(e.args[0]))
        exit()

    click.echo("INFO: generating report")

    aix.generate_analysis_v2(
        max_display=max_features_num,
        title=f"Feature Attribution\n({aix.model.name})",
        ylabel="UMLS Codes",
        save=True,
    )
    aix.show_violin(
        max_display=max_features_num,
        title="SHAP Values Distribution by Feature",
        ylabel="Features",
        show_n=True,
        asc=False,
        save=True,
    )
    aix.generate_local_analysis(default_record, max_display=max_features_num, save=True)


if __name__ == "__main__":
    run_shap_analysis()
