from class_pipe import Pipe

path_data_train = "/concepts/train/train.json"
path_data_test = "/concepts/test/test.json"
file_vectors_parameters = "/parameters/"
name_file_ngram_params = "list_ngram_params.json"
name_file_seq_params = "list_seq_params.json"
data_name = "Codes_EHR_OSA"
file_data = "/vectors/"
path_var = "codes"
file_models_parameters = "/models_parameters/"
name_file_hyper_params_mlp = "list_hyper_params_mlp.json"
name_file_hyper_params_lr = "list_hyper_params_logistic_regression.json"
name_file_hyper_params_lstm = "list_hyper_params_lstm.json"

pipe = Pipe(
    path_data_train,
    path_data_test,
    file_vectors_parameters,
    name_file_ngram_params,
    name_file_seq_params,
    data_name,
    file_data,
    path_var,
    file_models_parameters,
    name_file_hyper_params_mlp,
    name_file_hyper_params_lr,
    name_file_hyper_params_lstm,
)

pipe.run_seq_models()
