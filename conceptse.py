import utils_general_porpose
print("por aqui...")
import utils_clinical_concept_extraction
print("mas adelante...")
#import utils_general_porpose
import os
import utils_split_dataset
#import nltk
#print("importando nltk")
#nltk.download("stopwords")
#print("descargamos stropwords")

class ClinicalExtraction:
    def __new__(cls, *args, **kwargs):
        print("ClinicalConceptsExtraction object created")
        return super().__new__(cls)

    #def __init__(self, path_data_train, path_data_test, umlstoicd_path, qumls_path):
    def __init__(self, path_data_train, umlstoicd_path, qumls_path):
        self.path_data_train = path_data_train
        #self.path_data_test = path_data_test
        self.umlstoicd_path = umlstoicd_path
        self.qumls_path = qumls_path
        #self.name_data = name_data
        #self.directory_name = directory_name
        self.current_path = None
        #self.data = None
        self.patients_maxLength = None
        self.clinical_pipe = None

        self.patients = None
        self.patient_seq = None
        self.dictionary_entities = None
        print("ClinicalConceptsExtraction object instantiated")

    def __repr__(self):        
        return "filter_name: {}, p_data_train: {}, p_data_test: {}, path_umlstoicd: {}, path_qumls: {}".format(type(self).__name__, self.path_data_train, self.path_data_test, self.umlstoicd_path, self.qumls_path)
    
    def load_data(self):
        #_list_path_save = ["/concepts/train/", "/concepts/test/"]
        #_name_data = ["train", "test"]

        self.current_path = utils_general_porpose.get_current_path()
        data_train = utils_general_porpose.load_json(self.current_path, self.path_data_train)
        #self.data_test = utils_general_porpose.load_json(self.current_path, self.path_data_test)

        #self.patients = utils_clinical_concept_extraction.send_patient_info(self.data[:10])

        patients = utils_split_dataset.send_patient_info(data_train)
        self.patients_maxLength = utils_split_dataset.make_patient_list_with_maxlength(patients, 1000000)

        lista_example_umls_icd = utils_general_porpose.load_mapping_icd(self.current_path, self.umlstoicd_path)
        lista_NoIdentificados, lista_identificados = utils_clinical_concept_extraction.list_codes_identified(lista_example_umls_icd)
        target_rules = utils_clinical_concept_extraction.load_target_rules(lista_identificados)
        self.clinical_pipe = utils_clinical_concept_extraction.load_clinical_NLPpipe(self.current_path, self.qumls_path, target_rules)

    def run_pipe_ec(self):        
        
        self.patient_seq, self.dictionary_entities = utils_clinical_concept_extraction.extract_clinical_concepts(self.patients_maxLength, self.clinical_pipe)

        return print("clinical concepts extraction process finished")
    
    def save_data(self):
    
    #save the dataset
        path_save = self.current_path + "/concepts/"
        path_save = utils_general_porpose.create_directory(path_save)

        if os.path.exists(path_save):
            #if some version of dataset exists, then get the name list of the dataset
            list_dataframes = utils_general_porpose.extract_name_model(path_save, ".json")
            #extract the last version of the dataset
            last_version = utils_general_porpose.extract_last_version_model(list_dataframes)
            #get the number of the new version
            version = str(utils_general_porpose.counter_version(last_version))
            #save the dataset
            path_version = path_save + "clinical_concepts" + version + ".json"
            #data.to_csv(path_version, index = False)
            utils_general_porpose.save_json(self.patient_seq, path_version)
            #save the dictionary of entities
            path_entities = path_save + "dictionary_concepts" + version + ".json"
            utils_general_porpose.save_json(self.dictionary_entities, path_entities)

        
        
    
    
#python3
#from utils_class_conceptsExtraction import ClinicalExtraction

"""
extraction = ClinicalExtraction("/datasets/train/train.json", "/datasets/test/test.json", "/map/map_icd10_umls.csv", "/destination_umls_es")
extraction.run_pipe_ec()
print("concept extraction ended...")"
"""
