from __future__ import absolute_import, division, print_function, unicode_literals

import os
import glob

from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from math import sqrt


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #this removes some warning comments. This warning comments express that this PC has a CPU able to
#compute much faster, and that tensorflow was not designed for it. For the moment, will keep it like this. If necessary, we'll use GPU

# # IMPORT TEST AND STORE THEM
def test_import(path, look_back, dropin, dropout):

    # # PREDICTION DATA

    test_files = glob.glob(path + "/*.csv")  # keeping directories in a list
    n_test = len(test_files)  # number of files
    q = 0
    tests = [None]*n_test

    for csv in test_files:
        pred_data = pd.read_csv(csv, sep=';', encoding='cp1252')

        pred_input = pred_data.drop(columns=dropin)
        pred_input = pred_input.drop(pred_input.index[0:look_back])
        pred_input = pred_input.drop(pred_input.index[-look_back])
        pred_input = pred_input.values
        pred_input = np.array(np.reshape(pred_input, (pred_input.shape[0], 1, pred_input.shape[1])), dtype='float')

        t_step = pred_input.shape[0]

        pred_output = pred_data.drop(columns=dropout, index=0)
        pred_output = pred_output.shift(-look_back)
        pred_output = pred_output.drop(pred_output.index[-look_back])
        pred_output = pred_output.values
        pred_output = np.array(np.reshape(pred_output, (pred_output.shape[0], 1, pred_output.shape[1])), dtype='float')

        tests[q] = [pred_input, pred_output]

        q += 1

    return tests, t_step
def test_plot(model, tests):
    t = 0
    fig1, axs1 = plt.subplots(3,1, sharex='col')
    fig2, axs2 = plt.subplots(3,1, sharex='col')

    for te in tests:
        NN_pred = model.predict(te[0], batch_size=1)

        # ANALYSIS
        # reshape the prediction for plotting

        NN_pred = np.reshape(NN_pred, (te[1].shape[0], te[1].shape[2]))
        te[0] = np.reshape(te[0], (te[0].shape[0], te[0].shape[2]))
        te[1] = np.reshape(te[1], (te[1].shape[0], te[1].shape[2]))

        # plotspredicted and desired test output

        # plot currents
        #graph1 = fig1.subplot(len(tests), 1, t)
        axs1[t].set_title('Test0' + np.str(t+1))
        axs1[t].plot(NN_pred[:, -2:])
        axs1[t].plot(te[0][:, -2:])
        axs1[t].set_ylabel('I [p.u.]')
        axs1[2].set_xlabel('Time Steps')

        # plot voltages
        #graph2 = fig2.subplot(len(tests), 1, t, sharex=True)
        axs2[t].set_title('Test0' + np.str(t+1))
        axs2[t].plot(NN_pred[:, 0])
        axs2[t].plot(te[0][:, 0])


        axs2[t].axis((0, 74, 0.1, 1.1))
        axs2[t].set_ylabel('V [p.u.]')

        axs2[2].set_xlabel('Time Steps')
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
plt.rcParams.update({'font.size': 16})
test_path = r'C:\Users\Usuario\Documents\Universidad\TUM\Subjects\3rd semester\Research Internship\ANN_grids\Test'
dropin = ['All calculations']
dropout = ['All calculations']

[tests, time_step] = test_import(test_path, 1, dropin,dropout)

test2plot = [tests[0], tests[2], tests[4]]
# select some of the tests in the folder



# Import trained model with non-defined batch size

model = ('model1579631091.h5')

trained_model = load_model(r'C:\Users\Usuario\Documents\Universidad\TUM\Subjects\3rd semester\Research Internship\ANN_grids\model\{}'.format(model))

fig1, fig2 = test_plot(trained_model, test2plot)
fig1.legend(["Active current Pred.", 'Reactive Current pred.', 'Active Current', 'Reactive Current'])
fig2.legend(['Feed-in Voltage pred.', 'Feed-in Voltage'])

plt.show()

# PREDICTIONS WITH THE MODEL








