#!/usr/bin/env python3
#print("1. estamos aqui")
import pandas as pd
#print("2. segunda llegada")
import spacy
#print("2.1 segunda primera llegada")
import medspacy
#print("2.2 segunda segunda llegada")
from quickumls.spacy_component import SpacyQuickUMLS
#print("2.3 segunda tercera llegada")
from medspacy.util import DEFAULT_PIPE_NAMES
#print("3. llegamos aqui")
from medspacy.ner import TargetRule
#print("4 cuarta estacion")
from spacy.tokens import Span
#print("5 quinta estacion")
from medspacy.context import ConTextRule
#print("6 final del viacruciz")

#function to transform the dictionay ssequence patient data into strigns
def extract_sequence_from_dictionary(data, name_col_sequence, value_diagnostic, value_clinical_note, name_col_id_patient):
    lista_codes_icd10 = []

    for index, value in data[name_col_sequence].items():
        id_patient = value.get(name_col_id_patient)
        data.loc[index, "id_patient"] = id_patient
        list_seq = []
        for i in range(1, len(value)-1):
            num_consulta = "consulta_{}".format(i)
            codes_diagnostic = (str(value[num_consulta][value_diagnostic])).replace(",", " ")            
            text_diagnostic = str(value[num_consulta][value_clinical_note])
            list_seq.append(codes_diagnostic + " " + text_diagnostic.lower())            
            data.loc[index, "history_patient"] = " ".join(list_seq)
            lista_codes_icd10.append(codes_diagnostic)
            
    set_codes = set(lista_codes_icd10)
    return data, set_codes
    #return data


#función para cargar los nombres de los diagnosticos según el codigo
def load_code_dictionary(path_cod, name_file):
    abs_path = path_cod + name_file
    codigos_diagnosticos = pd.read_csv(abs_path, sep = ";")
    codigos_diagnosticos = codigos_diagnosticos[["IdDiagnostico", "NomDiagnostico", "CodCie9"]]
    dict_codigos = pd.Series(codigos_diagnosticos["NomDiagnostico"].values, index = codigos_diagnosticos["CodCie9"]).to_dict()

    return dict_codigos

#TODO: #Rule about using a sleep apnea vocabulary


#Function for add as target rules to the pipe
def load_target_rules(lista_identificados):
    Span.set_extension("description", default="")
    Span.set_extension("cui_code", default="")
    target_rules = []
    for item in lista_identificados:
        #target_rules.append(TargetRule(item["icd_husi"], "icd_husi", attributes = {"description":item["description"], "code":item["icd_10"], "cui_code":item["cui_code"]}))
        target_rules.append(TargetRule(item["icd_husi"], "icd_husi", attributes = {"description":item["description"], "cui_code":item["cui_code"]}))
    print("target rules loaded")
    return target_rules

#TODO: modify the context component for negated entity fot spanish language
#A list that have the rules for entity negation in spanish language for the context component
context_rule = [
    ConTextRule(literal='ausencia de', category='NEGATED_EXISTENCE', pattern=None, direction='FORWARD'),
    ConTextRule(literal='suficiente para descartar', category='NEGATED_EXISTENCE', pattern=[{'LOWER': {'IN': ['suficiente', 'adecuado']}}, {'LOWER': 'para'}, {'LOWER': 'descartar'}, {'LOWER': {'IN': ['él', 'ella', 'ellos', 'paciente', 'pt']}, 'OP': '?'}, {'LOWER': 'out'}, {'LOWER': {'IN': ['contra', 'a favor']}, 'OP': '?'}], direction='FORWARD'),
    ConTextRule(literal='suficiente para descartar al paciente', category='NEGATED_EXISTENCE', pattern=[{'LOWER': {'IN': ['suficiente', 'adecuado']}}, {'LOWER': 'para'}, {'LOWER': 'descartar'}, {'LOWER': 'al'}, {'LOWER': {'IN': ['paciente', 'pt']}}, {'LOWER': 'out'}, {'LOWER': {'IN': ['contra', 'a favor']}, 'OP': '?'}], direction='FORWARD'),
    ConTextRule(literal='cualquier otro', category='NEGATED_EXISTENCE', pattern=None, direction='FORWARD'),
    ConTextRule(literal='aparte de', category='NEGATED_EXISTENCE', pattern=[{'LOWER': 'aparte'}, {'LOWER': {'IN': ['de']}}], direction='TERMINATE'),
    ConTextRule(literal='se descartan', category='NEGATED_EXISTENCE', pattern=[{'LOWER': {'IN': ['son', 'es', 'fue']}}, {'LOWER': 'descartados'}], direction='BACKWARD'),
    ConTextRule(literal='como causa de', category='NEGATED_EXISTENCE', pattern=[{'LOWER': 'como'}, {'LOWER': {'IN': ['una', 'un', 'la']}}, {'LOWER': {'IN': ['causa', 'etiología', 'fuente', 'razón']}}, {'LOWER': {'IN': ['para', 'de']}}], direction='TERMINATE'),
    ConTextRule(literal='como tiene', category='NEGATED_EXISTENCE', pattern=None, direction='TERMINATE'),
    ConTextRule(literal='según sea necesario', category='HYPOTHETICAL', pattern=None, direction='FORWARD'),
    ConTextRule(literal='así como cualquier', category='NEGATED_EXISTENCE', pattern=None, direction='FORWARD'),
    ConTextRule(literal='a excepción de', category='NEGATED_EXISTENCE', pattern=None, direction='TERMINATE'),
    ConTextRule(literal='familiar', category='FAMILY', pattern=[{'LOWER': {'IN': ['tía', 'tías', 'hermano', 'hermanos', 'niño', 'niños', 'primo', 'primos', 'papá', 'papás', 'hija', 'hijas', 'fam', 'familia', 'familias', 'padre', 'padres', 'nieta', 'nietos', 'nieta...']}, 'OP': '?'}], direction='FORWARD'),
    ConTextRule(literal='porque', category='HYPOTHETICAL', pattern=[{'LOWER': {'IN': ['porque', 'ya que']}}], direction='TERMINATE'),
    ConTextRule(literal='riesgos de', category='HYPOTHETICAL', pattern=[{'LOWER': {'IN': ['riesgos de', 'riesgo de']}}], direction='FORWARD'),
    ConTextRule(literal='pero', category='NEGATED_EXISTENCE', pattern=[{'LOWER': {'IN': ['pero', 'sin embargo', 'no obstante', 'aun así', 'excepto', 'aunque', 'sin embargo', 'todavía']}}], direction='TERMINATE'),
    ConTextRule(literal='puede descartar', category='NEGATED_EXISTENCE', pattern=[{'LOWER': {'IN': ['puede', 'hizo']}}, {'LOWER': 'descartar'}, {'LOWER': {'IN': ['él', 'ella', 'ellos', 'paciente', 'pt']}, 'OP': '?'}, {'LOWER': 'out'}, {'LOWER': {'IN': ['contra', 'a favor']}, 'OP': '?'}], direction='FORWARD'),
    ConTextRule(literal='puede descartar al paciente', category='NEGATED_EXISTENCE', pattern=[{'LOWER': {'IN': ['puede', 'hizo']}}, {'LOWER': 'descartar'}, {'LOWER': 'al'}, {'LOWER': {'IN': ['paciente', 'pt']}}, {'LOWER': 'out'}, {'LOWER': {'IN': ['contra', 'a favor']}, 'OP': '?'}], direction='FORWARD'),
    ConTextRule(literal='causa de', category='NEGATED_EXISTENCE', pattern=[{'LOWER': {'IN': ['causa', 'causas', 'razón', 'razones', 'etiología']}}, {'LOWER': {'IN': ['de', 'para']}}], direction='TERMINATE'),
    ConTextRule(literal='revisado para', category='NEGATED_EXISTENCE', pattern=None, direction='FORWARD'),
    ConTextRule(literal='libre de', category='NEGATED_EXISTENCE', pattern=None, direction='FORWARD'),
    ConTextRule(literal='volver por', category='HYPOTHETICAL', pattern=[{'LOWER': 'volver'}, {'LOWER': 'por', 'OP': '?'}], direction='FORWARD'),
    ConTextRule(literal='preocupado por', category='POSSIBLE_EXISTENCE', pattern=[{'LOWER': {'IN': ['preocupado', 'preocupante']}}, {'LOWER': {'IN': ['por', 'acerca de']}}], direction='FORWARD'),
    ConTextRule(literal='descenso', category='NEGATED_EXISTENCE', pattern=[{'LOWER': {'IN': ['descenso', 'disminución', 'disminuye', 'disminuyendo']}}], direction='FORWARD'),
    ConTextRule(literal='negar', category='NEGATED_EXISTENCE', pattern=[{'LOWER': {'IN': ['negar', 'negado', 'niega', 'negando']}}], direction='FORWARD'),
    ConTextRule(literal='no descartó', category='POSSIBLE_EXISTENCE', pattern=[{'LOWER': {'IN': ['no', 'no lo']}}, {'LOWER': 'descartó'}], direction='BACKWARD'),
    ConTextRule(literal='no parece', category='NEGATED_EXISTENCE', pattern=[{'LOWER': {'IN': ['no', 'no es', 'no fue']}}, {'LOWER': {'IN': ['parecer', 'apreciar', 'demostrar', 'mostrar', 'sentir', 'tuvo', 'tener', 'revelar', 'ver']}, 'OP': '?'}], direction='FORWARD'),
    ConTextRule(literal='duda', category='POSSIBLE_EXISTENCE', pattern=None, direction='FORWARD'),
    ConTextRule(literal='evaluar para', category='NEGATED_EXISTENCE', pattern=None, direction='FORWARD'),
    ConTextRule(literal='no revela', category='NEGATED_EXISTENCE', pattern=None, direction='FORWARD'),
    ConTextRule(literal='libre', category='NEGATED_EXISTENCE', pattern=None, direction='BACKWARD'),
    ConTextRule(literal='libre de', category='NEGATED_EXISTENCE', pattern=None, direction='FORWARD'),
    ConTextRule(literal='ha sido negativo', category='NEGATED_EXISTENCE', pattern=None, direction='BACKWARD'),
    ConTextRule(literal='ha sido descartado', category='NEGATED_EXISTENCE', pattern=[{'LOWER': {'IN': ['ha', 'han']}}, {'LOWER': 'sido'}, {'LOWER': 'descartado'}], direction='BACKWARD'),
    ConTextRule(literal='historia', category='HISTORICAL', pattern=[{'LOWER': {'IN': ['historia', 'hx', 'hist', 'ho']}}], direction='FORWARD'),
    ConTextRule(literal='antecedentes', category='HISTORICAL', pattern=None, direction='FORWARD'),
    ConTextRule(literal='si', category='HYPOTHETICAL', pattern=None, direction='FORWARD'),
    ConTextRule(literal='inconsistente con', category='NEGATED_EXISTENCE', pattern=None, direction='FORWARD'),
    ConTextRule(literal='falta de', category='NEGATED_EXISTENCE', pattern=None, direction='FORWARD'),
    ConTextRule(literal='careció', category='NEGATED_EXISTENCE', pattern=[{'LOWER': {'IN': ['careció', 'careciendo']}}], direction='FORWARD'),
    ConTextRule(literal='puede descartarse', category='POSSIBLE_EXISTENCE', pattern=[{'LOWER': {'IN': ['puede', 'podría', 'debe', 'debería', 'será', 'podría', 'puede']}, 'OP': '?'}, {'LOWER': 'descartarse'}], direction='BACKWARD'),
    ConTextRule(literal='puede descartarse para', category='POSSIBLE_EXISTENCE', pattern=[{'LOWER': {'IN': ['puede', 'podría', 'debe', 'debería', 'será', 'podría', 'puede']}, 'OP': '?'}, {'LOWER': 'descartarse'}, {'LOWER': 'para'}], direction='FORWARD'),
    ConTextRule(literal='podría ser', category='POSSIBLE_EXISTENCE', pattern=[{'LOWER': {'IN': ['puede', 'podría']}}, {'LOWER': 'ser'}], direction='FORWARD'),
    ConTextRule(literal='negativo para', category='NEGATED_EXISTENCE', pattern=[{'LOWER': {'IN': ['negativo', 'neg']}}, {'LOWER': 'para'}], direction='FORWARD'),
    ConTextRule(literal='nunca tuvo', category='NEGATED_EXISTENCE', pattern=[{'LOWER': 'nunca'}, {'LOWER': {'IN': ['desarrollado', 'tuvo']}}], direction='FORWARD'),
    ConTextRule(literal='no', category='NEGATED_EXISTENCE', pattern=[{'LOWER': 'no'}, {'LOWER': {'IN': ['anormal', 'nuevo']}, 'OP': '?'}], direction='FORWARD'),
    ConTextRule(literal='no causa de', category='NEGATED_EXISTENCE', pattern=[{'LOWER': 'no'}, {'LOWER': {'IN': ['causa', 'historia', 'hallazgos', 'quejas', 'signo', 'signos', 'sugerencia']}}, {'LOWER': {'IN': ['de', 'para']}}], direction='FORWARD'),
    ConTextRule(literal='ninguna evidencia de', category='NEGATED_EXISTENCE', pattern=None, direction='FORWARD'),
    ConTextRule(literal='no significativo', category='NEGATED_EXISTENCE', pattern=[{'LOWER': 'no'}, {'LOWER': {'IN': ['significativo', 'sospechoso']}}], direction='FORWARD'),
    ConTextRule(literal='no diagnosticado', category='NEGATED_EXISTENCE', pattern=None, direction='BACKWARD'),
    ConTextRule(literal='no', category='NEGATED_EXISTENCE', pattern=[{'LOWER': {'IN': ['no', 'ni']}}, {'LOWER': {'IN': ['aparecer', 'apreciar', 'demostrar', 'exhibir', 'sentir', 'tener', 'revelar', 'ver']}, 'OP': '?'}], direction='FORWARD'),
    ConTextRule(literal='no asociado con', category='NEGATED_EXISTENCE', pattern=[{'LOWER': {'IN': ['no', 'ni']}}, {'LOWER': 'asociado'}, {'LOWER': 'con'}], direction='FORWARD'),
    ConTextRule(literal='no descartado', category='POSSIBLE_EXISTENCE', pattern=[{'LOWER': {'IN': ['no', 'ni']}}, {'LOWER': 'sido', 'OP': '?'}, {'LOWER': 'descartado'}], direction='BACKWARD'),
    ConTextRule(literal='no se queja de', category='NEGATED_EXISTENCE', pattern=[{'LOWER': {'IN': ['no', 'ni']}}, {'LOWER': {'IN': ['queja', 'sabe']}}, {'LOWER': 'de'}], direction='FORWARD'),
    ConTextRule(literal='no tener evidencia de', category='NEGATED_EXISTENCE', pattern=[{'LOWER': {'IN': ['no', 'ni']}}, {'LOWER': 'tener'}, {'LOWER': 'evidencia'}, {'LOWER': {'IN': ['de', 'para']}}], direction='FORWARD'),
    ConTextRule(literal='no conocido por tener', category='NEGATED_EXISTENCE', pattern=[{'LOWER': {'IN': ['no', 'ni']}}, {'LOWER': 'conocido'}, {'LOWER': 'por'}, {'LOWER': 'tener'}], direction='FORWARD'),
    ConTextRule(literal='no ser', category='NEGATED_EXISTENCE', pattern=[{'LOWER': {'IN': ['no', 'ni']}}, {'LOWER': 'ser'}], direction='FORWARD'),
    ConTextRule(literal='ahora resuelto', category='NEGATED_EXISTENCE', pattern=None, direction='BACKWARD'),
    ConTextRule(literal='origen de', category='NEGATED_EXISTENCE', pattern=[{'LOWER': 'origen'}, {'LOWER': {'IN': ['de', 'para']}}], direction='TERMINATE'),
    ConTextRule(literal='otras posibilidades de', category='NEGATED_EXISTENCE', pattern=None, direction='TERMINATE'),
    ConTextRule(literal='se debe descartar', category='POSSIBLE_EXISTENCE', pattern=[{'LOWER': 'se'}, {'LOWER': 'debe'}, {'LOWER': 'descartar'}], direction='BACKWARD'),
    ConTextRule(literal='historia pasada', category='HISTORICAL', pattern=[{'LOWER': 'historia'}, {'LOWER': 'médica', 'OP': '?'}, {'LOWER': 'pasada'}], direction='FORWARD'),
    ConTextRule(literal='el paciente no fue', category='NEGATED_EXISTENCE', pattern=[{'LOWER': 'el'}, {'LOWER': {'IN': ['paciente', 'pt']}}, {'LOWER': {'IN': ['fue', 'es']}}, {'LOWER': {'IN': ['no', 'ni']}}, {'LOWER': 'fue'}], direction='FORWARD'),
    ConTextRule(literal='posible', category='POSSIBLE_EXISTENCE', pattern=[{'LOWER': {'IN': ['posib', 'posible', 'probablemente', 'probable']}}], direction='FORWARD'),
    ConTextRule(literal='profilaxis', category='NEGATED_EXISTENCE', pattern=None, direction='BACKWARD'),
    ConTextRule(literal='r/o', category='POSSIBLE_EXISTENCE', pattern=None, direction='FORWARD'),
    ConTextRule(literal='en lugar de', category='NEGATED_EXISTENCE', pattern=None, direction='FORWARD'),
    ConTextRule(literal='resuelto', category='NEGATED_EXISTENCE', pattern=None, direction='FORWARD'),
    ConTextRule(literal='regreso', category='HYPOTHETICAL', pattern=None, direction='FORWARD'),
    ConTextRule(literal='rastreo', category='POSSIBLE_EXISTENCE', pattern=None, direction='FORWARD'),
    ConTextRule(literal='en lugar de', category='NEGATED_EXISTENCE', pattern=None, direction='FORWARD'),
    ConTextRule(literal='debe descartarse', category='POSSIBLE_EXISTENCE', pattern=[{'LOWER': 'debe'}, {'LOWER': 'descartarse'}], direction='BACKWARD'),
    ConTextRule(literal='fuente de', category='NEGATED_EXISTENCE', pattern=[{'LOWER': {'IN': ['fuente', 'fuentes']}}, {'LOWER': {'IN': ['de', 'para']}}], direction='TERMINATE'),
    ConTextRule(literal='sugerente de', category='POSSIBLE_EXISTENCE', pattern=None, direction='FORWARD'),
    ConTextRule(literal='sospechoso de', category='POSSIBLE_EXISTENCE', pattern=None, direction='FORWARD'),
    ConTextRule(literal='prueba de', category='NEGATED_EXISTENCE', pattern=None, direction='FORWARD'),
    ConTextRule(literal='para excluir', category='NEGATED_EXISTENCE', pattern=None, direction='FORWARD'),
    ConTextRule(literal='desencadenante de', category='NEGATED_EXISTENCE', pattern=None, direction='TERMINATE'),
    ConTextRule(literal='improbable', category='NEGATED_EXISTENCE', pattern=None, direction='BACKWARD'),
    ConTextRule(literal='sin nada', category='NEGATED_EXISTENCE', pattern=None, direction='FORWARD'),
    ConTextRule(literal='sin', category='NEGATED_EXISTENCE', pattern=None, direction='FORWARD'),
    ConTextRule(literal='sin ninguna evidencia de', category='NEGATED_EXISTENCE', pattern=[{'LOWER': 'sin'}, {'LOWER': 'ninguna', 'OP': '?'}, {'LOWER': {'IN': ['evidencia', 'indicación', 'signo']}}, {'LOWER': {'IN': ['de', 'para']}, 'OP': '?'}], direction='FORWARD'),
    ConTextRule(literal='preocupante por', category='POSSIBLE_EXISTENCE', pattern=None, direction='FORWARD'),
    ConTextRule(literal='preocupado por', category='POSSIBLE_EXISTENCE', pattern=None, direction='FORWARD'),
    ConTextRule(literal=': ninguno', category='NEGATED_EXISTENCE', pattern=None, direction='BACKWARD'),
    ConTextRule(literal=': no', category='NEGATED_EXISTENCE', pattern=None, direction='BACKWARD'),
    ConTextRule(literal='presentación', category='HISTORICAL', pattern=[{'LOWER': {'IN': ['presentación', 'presentando', 'presenta']}}], direction='TERMINATE'),
    ConTextRule(literal='viene con', category='HISTORICAL', pattern=None, direction='TERMINATE')
]

#function to load the nlp pipeline
def load_clinical_NLPpipe(path_destination, name_database, target_rules, simi):
    
    # Load the spacy model
    nlp = spacy.blank("spa")

    # Add the medspacy sentences component to the pipeline
    nlp.add_pipe("medspacy_pyrush")
    
    # Add the medspacy target matcher to the pipeline
    target_matcher = nlp.add_pipe("medspacy_target_matcher")
    target_matcher.add(target_rules)
    
    # Add the QuickUMLS component to the pipeline
    quickumls_file_path = path_destination + name_database    
    nlp.add_pipe('medspacy_quickumls', config={"threshold":simi, "quickumls_fp": quickumls_file_path})
    
    #Add the medspacy context component to the pipeline
    context = nlp.add_pipe("medspacy_context")
    context.add(context_rule)

    print("medspacy nlp-pipeline loaded")    
    return nlp

#function to extract the clinical concepts from the clinical history
"""def extract_clinical_concepts(data, name_col_history_patient, clinical_pipe):
    
    for index, value in data[name_col_history_patient].items():
        doc = clinical_pipe(value)
        list_entities = []
        list_codes = []
        
        for ent in doc.ents:

            if ent._.Diagnostic != "":
                list_entities.append(ent._.Diagnostic)
                list_codes.append(ent.text)
                data.loc[index, "entities"] = " ".join(list_entities)
                data.loc[index, "codes"] = " ".join(list_codes)

                
            elif ent._.Diagnostic == "":
                list_entities.append(ent.text)
                list_codes.append(ent.label_)                
                data.loc[index, "entities"] = " ".join(list_entities)
                data.loc[index, "codes"] = " ".join(list_codes)

    return data"""

def extract_clinical_concepts(patients_list, clinical_pipe):
    patients_seq = []
    dictionary_entities = {}
    for patient in patients_list:
        id_cliente = patient.get("id_cliente")
        label = patient.get("label")
        seq = patient.get("seq")        
        doc = clinical_pipe(seq)
        list_entities = []
        list_codes = []
                        
        for ent in doc.ents:               
            if ent._.description == "":
                #print(ent.text)
                list_entities.append(ent.text)
                #print(ent.label_)
                list_codes.append(ent.label_)
                entity = ent.text.split()
                dictionary_entities[ent.label_] = "_".join(entity)
            
            elif ent._.description != "":
                #print(ent._.Diagnostic)
                list_entities.append(ent._.description)
                #print(ent.text)
                list_codes.append(ent._.cui_code)
                entity = ent._.description.split()
                dictionary_entities[ent._.cui_code] = "_".join(entity)
        
        list_entities_str = " ".join(list_entities)
            #print(list_entities_str)
        list_codes_str = " ".join(list_codes)
            #print(list_codes_str)

        dict_patient = {"id_cliente":id_cliente, "label":label, "entities":list_entities_str, "codes":list_codes_str}
        patients_seq.append(dict_patient)
    print("clinical concepts extracted")
    return patients_seq, dictionary_entities

def list_codes_identified(lista_example_umls_icd):
    lista_NoIdentificados = []
    lista_identificados = []
    for row in lista_example_umls_icd:
        if row["cui_code"] == "None":
            lista_NoIdentificados.append(row)        
        else:
            lista_identificados.append(row)
    print("list of codes identified and not identified created")
    return lista_NoIdentificados, lista_identificados

def buscar_terminos_en_diccionario(lista_terminos):
    """
    Busca una lista de términos en un diccionario y devuelve una lista con los valores correspondientes.

    Args:
        lista_terminos (list): Lista de términos (valores) a buscar.        

    Returns:
        list: Lista con los valores encontrados en el diccionario.
    """
    # Diccionario basado en la tabla proporcionada
    diccionario = {
    "T047": "Disease or Syndrome",
    "T033": "Finding",
    "T191": "Neoplastic Process",
    "T184": "Sign or Symptom",
    "T046": "Pathologic Function",
    "T048": "Mental or Behavioral Dysfunction",
    "T034": "Laboratory or Test Result",
    "T037": "Injury or Poisoning",
    "icd": "icd_10"
    }


    return [{key} for key, value in diccionario.items() if value in lista_terminos]

#python3
#import utils_clinical_concept_extraction
