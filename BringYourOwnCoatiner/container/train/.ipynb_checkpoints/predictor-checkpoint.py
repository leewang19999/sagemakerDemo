# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import pickle
import StringIO
import sys
import signal
import traceback
import flask
import pdb

os.environ['KERAS_BACKEND'] = 'theano'

import pandas as pd
import numpy as np
from sklearn import preprocessing
from tensorflow.keras.models import load_model
#import keras
#from keras import backend

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'ownScriptModel.pth')
output_path = os.path.join(prefix, 'output')

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model = None                # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            cls.model = load_model(os.path.join(model_path, 'rnn-combo-model.h5'))
        return cls.model

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        clf = cls.get_model()
        clf._make_predict_function()
        return clf.predict(input)  	
    
#def predict2(clf,input):
#    return clf.predict(input)

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here
    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None

    # Convert from CSV to pandas
    if flask.request.content_type == 'text/csv':
        data = flask.request.data.decode('utf-8')
        print(data)
        s = StringIO.StringIO(data)
        s_on = data # Need data when using ECR/Flask but s when run locally
        data = pd.read_csv(s_on, index_col = 0) #, header=None) # [Jeremy: here we're using the header]
    else:
        return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')



    ###########################################################
    #                                                         #
    #     Prepare/rescale data based on training dataset      #
    #                                                         #
    ###########################################################


    # Read training data to set scaler parameters based on training data
    print("Model path contains:  ", os.listdir(model_path))
    file = os.path.join(model_path, 'traindata.csv') 
    print("Reading: ", file)
    traindata = pd.read_csv(file, index_col = 0)
    file = os.path.join(model_path, 'sentiment.csv') 
    print("Reading: ", file)
    news = pd.read_csv(file,index_col = 0)

    # Select covariates of interest
    target = 'STOCKA' # Hardcoded for now
    covariates = ['STOCKB'] # Hardcoded for now

    # Function to normalize ML inputs
    #def normalize_data(df, traindata):
    #    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    #    for feat in df.columns[1:]:
    #        print('Rescaling ',feat)
    #        sc_params = scaler.fit(traindata.eval(feat).values.reshape(-1,1)) # Use training data to set params
    #        df[feat] = scaler.fit_transform(df.eval(feat).values.reshape(-1,1),sc_params)
    #    return df

    def normalize_data(df, traindata):
        s = df.shape
        data = pd.concat([traindata,df],axis=0)
        data = data.diff() # This will require to drop symbol before; also applying differencing on merged set prevents NaN for first time step
        data = data.replace(np.nan, 0)
        scaler = preprocessing.StandardScaler() # or: MinMaxScaler(feature_range=(0,1))
        for feat in data.columns: # [1:]:
            print('Rescaling ',feat)
            data[feat] = scaler.fit_transform(data.eval(feat).values.reshape(-1,1))
            norm = data[-s[0]:]
            return norm


    # Function to denormalize ML outputs
    def denormalize(array, traindata):
        #traindata = np.diff(traindata,n=1) # Again reuquires to drop symbol before; also, only difference this set ince predictions made by the rnn are already differenced
        traindata = traindata.diff()
        traindata = traindata.replace(np.nan, 0) # Doesn't work if traindata is not a df 
        #traindata[np.isnan(traindata)] = 0
        scaler = preprocessing.StandardScaler() # or: MinMaxScaler(feature_range=(0,1))
        scaler.fit_transform(traindata['adjclose'].values.reshape(-1,1)) 
        new = scaler.inverse_transform(array.reshape(-1,1)) #df.values.reshape(-1,1))
        return new


    # Main time series
    print('Rescaling ',target)
    test_main = data[data.symbol == target]
    train_main = traindata[traindata.symbol == target]
    test_main["adjclose"] = test_main.close # Moving close to the last column
    test_main.drop(['close','symbol'], 1, inplace=True)
    test_main_ori = test_main
    train_main["adjclose"] = train_main.close # Moving close to the last column
    train_main.drop(['close','symbol'], 1, inplace=True)
    test_main = normalize_data(test_main,train_main)

    # Exogenous time series
    i=0
    for covariate in covariates:
        print('Rescaling ',covariate)
        test_exo1 = data[data.symbol == covariate]
        train_exo = traindata[traindata.symbol == covariate]
        # Plug in market sentiment here
        news = news[news.index >= test_exo1.index[0]]
        news = news[news.index <= test_exo1.index[-1]] 
        test_exo1["s"] = news.s
        test_exo1["adjclose"] = test_exo1.close # Moving close to the last column
        test_exo1.drop(['close','symbol'], 1, inplace=True)
        train_exo["adjclose"] = train_exo.close # Moving close to the last column
        train_exo.drop(['close','symbol'], 1, inplace=True)
        test_exo1 = normalize_data(test_exo1,train_exo)
        if i == 0:
            test_exo = test_exo1
        else:
            test_exo = test_exo.append(test_exo1) 
        i+=1



    ########################################################### 
    #							      #
    #  Select input features and run model to make prediction # 
    #							      #
    ###########################################################

    # TEMPORARY hyperparameter list -- Rearrange hyperparameter calls later
    lag = 22
    horiz = 10

    def load_data(stock, lag, horiz, nfeatures):
        data = stock.values
        lags = []
        horizons = []
        nsample = len(data) - lag - horiz  # Number of time series (Number of sample in 3D)
        for i in range(nsample):
            lags.append(data[i: i + lag , -nfeatures:]) 
            horizons.append(data[i + lag : i + lag + horiz, -1])
        lags = np.array(lags)
        horizons = np.array(horizons)
        x_test = lags
        y_test = horizons 
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], nfeatures))
        return [x_test, y_test, nfeatures]

    Xmain_test, ymain_test, nfeatmain = load_data(test_main, lag, horiz, 2)
    Xexo_test, yexo_test, nfeatexo = load_data(test_exo, lag, horiz, 3)
    # Get the undifferenced data
    ori, dummy, dummy = load_data(test_main_ori, lag, horiz, 2)

    # Do the prediction
    predictions = ScoringService.predict({'main_in': Xmain_test, 'exo_in': Xexo_test})

    # Rescale back to original target range
    # predictions is a 3d tensor where d1 = main vs. combo, d2 = sample instance, d3 = horizon
    pred = denormalize(predictions[0][0],train_main) # here 3d tensor predictions may have to be processed in batches along each dimension as denormalize reshape in 1D I think
    # Add predicted trends to actual values
    pred[0,0] = ori[0,-1,-1] + pred[0,0] # Index 0 assumes only one sample, needs be reviewed 
    for i in range(1,len(pred[:,0])):
    	pred[i,0] = pred[i-1,0] + pred[i,0]

    # The below needs be reviewed
    #results = pd.read_csv(StringIO.StringIO(pred[:,0]), header=None)
    #print('Debug:',results) 
    # Convert from numpy back to CSV
    out = StringIO.StringIO()
    #results.to_csv(out, header=False)
    pd.DataFrame({'results':pred[:,0]}).to_csv(out, header=False)
    result = out.getvalue()
    print('Results: ', result)

    return flask.Response(response=result, status=200, mimetype='text/csv')
