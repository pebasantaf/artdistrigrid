from __future__ import absolute_import, division, print_function, unicode_literals
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import glob

# # DATA MANAGER: Define a function that imports and defines data

#function to import and prepare test data

def importTestData(path, look_back, drop):

    # # PREDICTION DATA

    test_files = glob.glob(path + "/*.csv")  # keeping directories in a list
    n_test = len(test_files)  # number of files
    q = 0
    tests = [None]*n_test

    for csv in test_files:
        pred_data = pd.read_csv(csv, sep=';', encoding='cp1252')

        pred_input = pred_data.drop(columns=drop, index=0)
        pred_input = pred_input.drop(pred_input.index[0:look_back])

        pred_input = pred_input.values
        pred_input = np.array(np.reshape(pred_input, (pred_input.shape[0], 1, pred_input.shape[1])), dtype='float')
        t_step = pred_input.shape[0]

        pred_output = pred_data.drop(columns=drop, index=0)
        pred_output = pred_output.shift(look_back)
        pred_output = pred_output.drop(pred_output.index[0:look_back])
        pred_output = pred_output.values
        pred_output = np.array(np.reshape(pred_output, (pred_output.shape[0], 1, pred_output.shape[1])), dtype='float')

        tests[q] = [pred_input, pred_output]

        q += 1

    return tests, t_step

# function to import and prepare train data

def importTrainData(path, look_back, t_step, drop,inout):
    # # TRAINING DATA

    #Introduce the path and count files

    train_files = glob.glob(path + "/*.csv") #keeping directories in a list
    n_files = len(train_files) #number of files


    #To check encoding of a file just print its path: with open(r'I:\05_Basanta Franco\Python\Data02\Data1574095060.csv') as f:
        #print(f)

    datain = np.zeros([n_files * (t_step+1), inout[0]])
    dataout = np.zeros([n_files * (t_step+1), inout[1]])

    i = t_step+1
    j = 0

    #import all the csv in files and store them in data
    for csv in train_files:
        matrix = pd.read_csv(csv, sep=';', encoding='cp1252')

        matrixin = matrix.drop(columns=drop, index=0)
        matrixout = matrix.drop(columns=drop, index=0)

        matrixin = matrixin.astype(float)
        matrixout = matrixout.astype(float)

        datain[j:i, :] = matrixin.values
        dataout[j:i, :] = matrixout.values

        i += t_step+1
        j += t_step+1


    if look_back ==1:

        #remove last row of input data
        inputs = datain[:-1]
        X = inputs.reshape(inputs.shape[0], 1, inputs.shape[1])

        # shift output data 1 row
        outputs = dataout[1:]
        Y = outputs.reshape(outputs.shape[0], 1, outputs.shape[1])
    else:
        dataX = []
        dataY = []

        for n in range(0, data.shape[0]-look_back):

            # create lists dataX and data Y that will keep values for every time step considering the look back
            seq_in = datain[n:n+look_back]
            seq_out = dataout[n+look_back]
            dataX.append(char for char in seq_in)
            dataY.append(seq_out)
            i += 1

        #reshape the data to output a tensor
        X = np.reshape(dataX, (len(dataX), look_back, dataX[0].shape[1]))
        Y = np.reshape(dataX, (len(dataY), look_back, dataY[0].shape[1]))

    return X, Y


# https://docs.python.org/3/tutorial/modules.html
# https://stackoverflow.com/questions/35727134/module-imports-and-init-py-in-python
# https://www.youtube.com/watch?v=6tNS--WetLI