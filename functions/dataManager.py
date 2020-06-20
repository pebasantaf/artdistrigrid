from __future__ import absolute_import, division, print_function, unicode_literals
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import Callback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from math import sqrt
import glob

class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True

# # DATA MANAGER: Define a function that imports and defines data

#function to import and prepare test data

def importTestData(path, look_back, dropin, dropout):

    # # PREDICTION DATA

    test_files = glob.glob(path + "/*.csv")  # keeping directories in a list
    n_test = len(test_files)  # number of files
    q = 0
    tests = [None]*n_test

    for csv in test_files:
        pred_data = pd.read_csv(csv, sep=';', encoding='cp1252')

        pred_input = pred_data.drop(columns=dropin, index=0)
        pred_input = pred_input.drop(pred_input.index[0:look_back])

        pred_input = pred_input.values
        pred_input = np.array(np.reshape(pred_input, (pred_input.shape[0], 1, pred_input.shape[1])), dtype='float')
        t_step = pred_input.shape[0]

        pred_output = pred_data.drop(columns=dropout, index=0)
        pred_output = pred_output.shift(look_back)
        pred_output = pred_output.drop(pred_output.index[0:look_back])
        pred_output = pred_output.values
        pred_output = np.array(np.reshape(pred_output, (pred_output.shape[0], 1, pred_output.shape[1])), dtype='float')

        tests[q] = [pred_input, pred_output]

        q += 1

    return tests, t_step

# function to import and prepare train data

def importTrainData(path, look_back, t_step, dropin, dropout,inout):
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

        matrixin = matrix.drop(columns=dropin, index=0)
        matrixout = matrix.drop(columns=dropout, index=0)

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


# functions for plotting and testing

def TestPlot(model, tests):
    t = 1
    fig1 = plt.figure()
    fig2 = plt.figure()

    for te in tests:
        NN_pred = model.predict(te[0], batch_size=1)

        # ANALYSIS
        # reshape the prediction for plotting

        NN_pred = np.reshape(NN_pred, (te[1].shape[0], te[1].shape[2]))
        te[0] = np.reshape(te[0], (te[0].shape[0], te[0].shape[2]))
        te[1] = np.reshape(te[1], (te[1].shape[0], te[1].shape[2]))

        # plotspredicted and desired test output

        # plot currents
        graph1 = fig1.add_subplot(len(tests), 1, t)
        graph1.set_title('Test0' + np.str(t))
        graph1.plot(NN_pred[:, -2:])
        graph1.plot(te[1][:, -2:])
        graph1.legend(['I_P_pred', 'I_Q_pred', 'I_P', 'I_Q'])

        # plot voltages
        graph2 = fig2.add_subplot(len(tests), 1, t)
        graph2.set_title('Test0' + np.str(t))
        graph2.plot(NN_pred[:, 0])
        graph2.plot(te[1][:, 0])
        graph2.legend(['V_0_pred', 'V_0'])

        # mean squared error
        rmse = sqrt(mean_squared_error(te[1], NN_pred))
        print('Test RMSE: %.3f' % rmse)

        t = t + 1
    '''
    #  print inputs yes or no
    printin = input('Print inputs as well? [y/n]: ')

    m = True
    while m == True:

        if printin == 'y':

            t = 1
            fig4 = plt.figure()

            for te in tests:
                graph4 = fig4.add_subplot(len(tests), 1, t)
                graph4.set_title('Inputs: V, P, Q')
                graph4.plot(te[0][:, 10])
                t = t + 1
            m = False
        elif printin == 'n':

            m = False
        else:

            printin = input('Answer not valid. Print inputs? [y/n]: ')
        '''
    return fig1, fig2


# plot loss during training

def evaluateTraining(history):
    fig3 = plt.figure()

    graph3 = fig3.add_subplot(211)
    graph3.set_title('Loss')
    graph3.plot(history.history['loss'], label='train')
    graph3.plot(history.history['val_loss'], label='test')
    graph3.legend()
    # plot mse during training
    graph3 = fig3.add_subplot(212)
    graph3.set_title('Mean Squared Error')
    graph3.plot(history.history['mse'], label='train')
    graph3.plot(history.history['val_mse'], label='test')

    return fig3

# https://docs.python.org/3/tutorial/modules.html
# https://stackoverflow.com/questions/35727134/module-imports-and-init-py-in-python
# https://www.youtube.com/watch?v=6tNS--WetLI