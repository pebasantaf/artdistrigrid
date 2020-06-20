from __future__ import absolute_import, division, print_function, unicode_literals

import os
import glob
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import Callback

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
from math import sqrt


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #this removes some warning comments. This warning comments express that this PC has a CPU able to
#compute much faster, and that tensorflow was not designed for it. For the moment, will keep it like this. If necessary, we'll use GPU


keras.backend.reset_uids()
session = tf.compat.v1.Session()


#Early stopping based on validation loss

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

def test_import(path, look_back, dropin, dropout):

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

def train_import(path, look_back, t_step, dropin, dropout,inout):
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

def test_plot(model, tests):
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

def train_eval(history):
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

# if __name__ == __main__ check online. Ill need that to import functions without running whole file

# # PATHS

test_path = r'C:\Users\Usuario\Documents\Universidad\TUM\Subjects\3rd semester\Research Internship\ANN_grids\Test'
train_path = r'C:\Users\Usuario\Documents\Universidad\TUM\Subjects\3rd semester\Research Internship\ANN_grids\Data02'
model_path = r'C:\Users\Usuario\Documents\Universidad\TUM\Subjects\3rd semester\Research Internship\ANN_grids\model\model{}.h5'


# # IMPORT DATA

'''
dropin = ['All calculations', 'DEA_S1_1', 'DEA_S2_1', 'DEA_S2_2', 'DEA_S2_3', 'DEA_S3_1', 'DEA_S3_2', 'DEA_S3_3', 'DEA_S4_1', 'DEA_S4_2', 'DEA_S4_3']
dropout = ['All calculations', 'DEA_S1_1', 'DEA_S2_1', 'DEA_S2_2', 'DEA_S2_3', 'DEA_S3_1', 'DEA_S3_2', 'DEA_S3_3', 'DEA_S4_1', 'DEA_S4_2', 'DEA_S4_3']
'''

inout = [13, 13]
look_back = 1
dropin = ['All calculations']
dropout = ['All calculations']

# import test data
tests, t_step = test_import(test_path, look_back, dropin, dropout) #test is a list with test inputs and outputs.

# import train data

X, Y = train_import(train_path, look_back, t_step, dropin, dropout, inout)


# # NEURAL NETWORK CREATOR

# Creating a model, which is a linear stack of layers
model = Sequential()

'''
LSTM layer of n nodes. Shape of the input is the columns of inputs. activation function is rectifier linear function.
Return sequencies = true basically tells the layer to output a sequence. If we were to have another Recurrent layer, this is necessary. Else not, as it would not understand it
Time distributed is important. That basically relates every input step in the input sequence with its corresponding output. 
Other way we would just be considering the last value of the sequence
'''

l1 = model.add(layers.LSTM(26, input_shape=(look_back, inout[0]), activation='relu', return_sequences=True)) #adding a RNN layer


l2 = model.add(layers.TimeDistributed(layers.Dense(5)))

l3 = model.add(layers.Dropout(0.2))


l4 = model.add(layers.Dense(inout[1])) #fully connected layer. What i would understand as a normal layer


opt = optimizers.RMSprop(lr=1e-03) #how fast the learning rate decays. this helps finding the miminum better

callbacks = [EarlyStoppingByLossVal(monitor='val_loss', value=0.002),
     ModelCheckpoint(filepath=model_path.format(int(time.time())), save_best_only=True)]

#compiling the model. Defining some of sthe features for the fit like the type of loss function, the optimizer and metrics that are interesting for us

model.compile(loss='mean_squared_error',
             metrics=['mse', 'mae'])  # accuracy only valid for classification tasks

history = model.fit(X, Y, epochs=15, validation_split=0.25, callbacks=callbacks, batch_size=200)


# Evaluate the model

scores = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# print a summary of the outputs of every layer

print(model.summary())

# PLOTS

fig1, fig2 = test_plot(model, tests)

fig3 = train_eval(history)

plt.show()
keras.backend.reset_uids()
keras.backend.clear_session()

'''
https://stackoverflow.com/questions/47594861/predicting-a-multiple-forward-time-step-of-a-time-series-using-lstm
https://stackoverflow.com/questions/38714959/understanding-keras-lstms/50235563#50235563
https://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/
'''


