import numpy as np


LABELS_SOURCE_PATH = './data/processed_labels_32'
LABELS_DUMMY_DESTINATION_PATH = './data/temp_processed_labels'


LABEL_DICT = {'epithelium':1, 'blood_vessels':2, 'stroma':3, 'adipocytes':4}

COLOR_DICT = {'epithelium':np.array([255, 0, 0]), 
              'blood_vessels':np.array([0, 0, 255]), 
              'stroma':np.array([0, 255, 0]) , 
              'adipocytes':np.array([255, 255, 0])}


def get_label_index(path):
    '''
    Given a file name if the class label name is contained 
    in the file return the class index for that image
    '''
    if "epithelium" in path:
        return LABEL_DICT["epithelium"]
    elif "blood_vessels" in path:
        return LABEL_DICT["blood_vessels"]
    elif "stroma" in path:
        return LABEL_DICT["stroma"]
    else:
        return LABEL_DICT["adipocytes"]

### COLOR FUNCTIONS ###

def rgb_to_hex(rgb):
    '''
    Converts RGB to HEX color code
    '''
    return '#{:02x}{:02x}{:02x}'.format(*rgb)

def get_color(path):
    '''
    Given a file name if the class label name is contained 
    in the file return the dict color for that image
    '''
    if "epithelium" in path:
        return COLOR_DICT["epithelium"]
    elif "blood_vessels" in path:
        return COLOR_DICT["blood_vessels"]
    elif "stroma" in path:
        return COLOR_DICT["stroma"]
    else:
        return COLOR_DICT["adipocytes"]