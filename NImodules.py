# This is a collection of modular VI written by Watson's Lab
#only for absolutly path

from Callout_Lb7 import Call_VI

start_path = "C:\\D\\experiment\\program"
IVretrieve_path = "\\control\\National Instruments Downloads\\VSA (backup)\\IVretriv.llb\\IV Retrieve, simple"
@Call_VI
def sin_py(step):
    pack = dict()
    pack['VIPath'] = start_path + "\\analysis\\sin.vi"
    pack['ParameterNames'] = ["x"]
    pack['Parameters'] = [step]
    pack['Indicators'] = ["y"]
    return pack


@Call_VI
def get_data_py(file):
    pack = dict()
    pack['VIPath'] = "C:\\D\\experiment\\program\\analysis\\test collection\\get_data.vi"
    pack['ParameterNames'] = ["file path"]
    pack['Parameters'] = [file]
    pack['Indicators'] = ["signals"]
    return pack

@Call_VI
def IV_Retrieve_py(file):
    pack = dict()
    pack['VIPath'] = start_path + IVretrieve_path
    pack['ParameterNames'] = ["file","Logs"]
    pack['Parameters'] = [file[0],file[1]]
    pack['Indicators'] = ["output cluster"]
    return pack

@Call_VI
def IV_Retrieve_Titles_log_py(file):
    pack = dict()
    pack['VIPath'] = start_path + IVretrieve_path
    pack['ParameterNames'] = ["file"]
    pack['Parameters'] = [file]
    pack['Indicators'] = ["Titles","# of logs"]
    return pack


@Call_VI
def IV_Retrieve_Current_py(file_C_log):
    pack = dict()
    pack['VIPath'] = start_path + IVretrieve_path
    pack['ParameterNames'] = ["file","Coordinates","Logs"]
    pack['Parameters'] = [file_C_log[0],file_C_log[1],file_C_log[2]]
    pack['Indicators'] = ["C-array"]
    return pack


@Call_VI
def ECG(input):
    pack = dict()
    pack['VIPath'] = start_path + "\\analysis\\MMF_Concept_Demo\\Untitled 1.vi"
    pack['ParameterNames'] = ["Numeric"]
    pack['Parameters'] = [input]
    pack['Indicators'] = ["Y"]
    return pack

@Call_VI
def Data_py(input):
    pack = dict()
    pack['VIPath'] = start_path + "\\control\\Labview 7 2018\\data tools\\analyse tools.llb\\Data MMF.vi"
    pack['ParameterNames'] = ["input data 2"]
    pack['Parameters'] = [input]
    pack['Indicators'] = ["input data 2"]
    return pack



# VSA_control(False) # For Testing
