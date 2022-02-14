# -*- coding: utf-8 -*-

import os
import pickle
import base64
import numpy as np
# import urllib.request


def modelPredict(model, data, threshold):
    '''
        Returns the prediction of the model: 0 or 1 depending on the threshold 
        and the exact probability value given by the model.
    '''
    
    pP = model.predict_proba(data)[:,0].item()
    pE = int(np.where(pP<threshold,0,1))
    
    return convToB64(
        dict(
            predProba = pP,
            predExact = pE
            )
        )

def loadModelLightGBM(formatFile='b64'):
    '''
    Unpickle and load the Machine Learning model 'model.pkl'
    Depending on the value of <formatFile>:
        returns the model in its nominal format
        returns the model converted to base-64 format encoded in UTF-8.
    '''
    model = pickle.load(open(os.getcwd()+'/pickle/model.pkl', 'rb'))
    
    if formatFile == 'pkl':
        return model
    elif formatFile == 'b64':
        return convToB64(model)
    else:
        return model.class_weight

def loadColumnsOfModel():
    '''
    Unpickle and returns the columns expected by 
    the Machine Learning model used in this project.
    '''
    return pickle.load(open(os.getcwd()+'/pickle/cols.pkl', 'rb'))    

def convToB64(data):
    '''
    As input: <data> of any kind.
    The function converts the <data> to base-64 then the resulting string is encoded in UTF-8.
    Output: The result obtained.
    '''
    return base64.b64encode(pickle.dumps(data)).decode('utf-8')

def restoreFromB64Str(data_b64_str):
    '''
    Input: Data converted to Base-64 and then encoded to UTF-8. 
          Ideally data from the convToB64 function.
    The function restores the encoded data to its original format.
    Output: The restored data
    '''
    return pickle.loads(base64.b64decode(data_b64_str.encode()))
