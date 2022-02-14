# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

from flask import Flask, request
import utils
import pandas as pd
import sys
import os
import pickle
import shutil
import lightgbm

### Initialisation ###
app = Flask(__name__)
mo = utils.loadModelLightGBM(formatFile='pkl') # Named mo because a function named model already exists
cols = utils.loadColumnsOfModel()
minScore=0
maxScore=1
th = 0.52 # Named th because a function named threshold already exists
MYDIR = os.path.dirname(__file__)
# formatOsSlash = '\\'
formatOsSlash = '/'
tmpDirName = 'tmpSplit'
tmpDir = formatOsSlash+tmpDirName+formatOsSlash
#####################

### app.route - Start ###
@app.route('/lightgbm/',methods=['POST'])
def lightgbm():
    return utils.modelPredict(mo,utils.restoreFromB64Str(request.args.get('data_b64_str')),th)

@app.route('/model/',methods=['POST'])
def model():
    return utils.loadModelLightGBM(formatFile='b64')

@app.route('/ratingSystem/',methods=['POST'])
def ratingSystem():
    '''
    Return the details of the rating system.
    Input: Nothing
    Output: 
    - Minimum score of the scoring system
    - Maximum score of the rating system
    - Threshold of the scoring system
    '''
    print(f'minScore={minScore}', file=sys.stderr)
    print(f'maxScore={maxScore}', file=sys.stderr)
    print(f'th={th}', file=sys.stderr)
    return utils.convToB64((minScore,maxScore,th))

@app.route('/')
def helloworld():
    return '''<h1>Bienvenue sur l' API du Projet implementez un model de scoring'</h1>'''

@app.route('/initSplit/',methods=['POST'])
def initSplit():
    global MYDIR
    global tmpDirName

    print('initSplit', file=sys.stderr)

    # Initialize the destination folder
    # Delete it if it exists, with all that it contains
    shutil.rmtree(tmpDirName, ignore_errors=True)
    # We create the temporary folder
    if not os.path.exists(tmpDirName):
        os.makedirs(tmpDirName)
    return utils.convToB64(True)

@app.route('/merge/',methods=['POST'])
def splitN():
    global MYDIR
    global tmpDir
    pathFile = MYDIR+tmpDir+request.values.get("numSplit")+'.pkl'
    strToSave = request.values.get('txtSplit')
    
    print(f'Merge - numSplit={request.values.get("numSplit")}', file=sys.stderr)

    # We save the received content in a pickle file
    # The pickle file is named after the split number
    pickle.dump(strToSave, open(pathFile, 'wb'))
    return utils.convToB64(True)

@app.route('/endSplit/',methods=['POST'])
def endSplit():
    global MYDIR
    global tmpDir
    global tmpDirName
    txtB64Global = ''

    print('endSplit', file=sys.stderr)
    
    # Restore data
    for i in range(5):
        pathFile = MYDIR+tmpDir+str(i)+'.pkl'
        # We open the pickle file and attach its contents to the global txt
        txtB64Global += pickle.load(open(pathFile, 'rb'))
    

    # On restore les données

    print(f'Len de txtB64Global={len(txtB64Global)}', file=sys.stderr)
    
    # Decode Data
    dataOneCustomer = utils.restoreFromB64Str(txtB64Global)
    
    # Creation of dfOneCustomer
    dfOneCustomer = pd.DataFrame(data=dataOneCustomer, columns=cols)
    
    # For security reasons we delete the temporary folder
    shutil.rmtree(tmpDirName, ignore_errors=True)
    
    # Interrogation of the model and return of the results
    return utils.modelPredict(mo,dfOneCustomer,th)


### app.route - End ###

if __name__ == "__main__":
    app.run()