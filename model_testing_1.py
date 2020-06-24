from __future__ import absolute_import, division, print_function, unicode_literals

import os
import glob

from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from math import sqrt
from Model_def import test_import

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #this removes some warning comments. This warning comments express that this PC has a CPU able to
#compute much faster, and that tensorflow was not designed for it. For the moment, will keep it like this. If necessary, we'll use GPU

# # IMPORT TEST AND STORE THEM

test_path = r'C:\Users\Usuario\Documents\Universidad\TUM\Subjects\3rd semester\Research Internship\ANN_grids\Test'
dropin = ['All calculations']
dropout = ['All calculations']

[tests, time_step] = test_import(test_path, 1, dropin,dropout)

test_files = glob.glob(test_path + "/*.csv")  # keeping directories in a list
n_test = len(test_files)  # number of files
q = 0
inout = (13, 12)
col = ['All calculations', 'MSNS-Trafo', 'MSNS-Trafo.1']
test = [None]*n_test

for csv in test_files:
    pred_data = pd.read_csv(csv, sep=';', encoding='cp1252')
    t_step = pred_data.shape[0]  # as all data has the same shape, we can keep these values for later use


    pred_input = np.array(np.reshape(pred_data.drop(columns=col, index=0).values,
                                     (pred_data.shape[0], 1,  inout[0])), dtype='float') #Remove selected columns and indexes. Reshape data
    pred_output = np.array(pred_data.loc['1':, col[1]: col[2]], dtype='float')


    test[q] = [pred_input, pred_output]

    q = q + 1

# select some of the tests in the folder
test = test[0:5]


# Import trained model with non-defined batch size

model = ('model1576169901.h5')

trained_model = load_model(r'C:\Users\Usuario\Documents\Universidad\TUM\Subjects\3rd semester\Research Internship\ANN_grids\model\{}'.format(model))


# PREDICTIONS WITH THE MODEL

t = 1
fig1 = plt.figure()
fig2 = plt.figure()
for te in test:

    NN_pred = trained_model.predict(te[0], batch_size=1)


    #ANALYSIS
    #reshape the prediction for plotting
    NN_pred = np.reshape(NN_pred, (te[1].shape[0], inout[1]))
    te[0] = np.reshape(te[0], (t_step-1, inout[0]))
    te[1] = np.reshape(te[1], (t_step-1, inout[1]))

    # plotspredicted and desired test output

    # plot currents
    graph1 = fig1.add_subplot(len(test), 1, t)
    graph1.set_title('Test0' + np.str(t))
    graph1.plot(NN_pred[:, -2:])
    graph1.plot(te[1][:, -2:])
    graph1.legend(['I_P_pred', 'I_Q_pred', 'I_P', 'I_Q'])

    # plot voltages
    graph2 = fig2.add_subplot(len(test), 1, t)
    graph2.set_title('Test0' + np.str(t))
    graph2.plot(NN_pred[:, 1])
    graph2.plot(te[1][:, 1])
    graph2.legend(['V_1_pred', 'V_1'])


    # mean squared error
    rmse = sqrt(mean_squared_error(te[1], NN_pred))
    print('Test RMSE: %.3f' % rmse)

    t = t + 1



#  print inputs yes or no
printin = input('Print inputs as well? [y/n]: ')

m = True
while m == True:

    if printin == 'y':

        t = 1
        fig4 = plt.figure()

        for te in test:
            graph4 = fig4.add_subplot(len(test), 1, t)
            graph4.set_title('Inputs: V, P, Q')
            graph4.plot(te[0][:, 10])
            t = t + 1
        m = False
    elif printin == 'n':

        m = False
    else:

        printin = input('Answer not valid. Print inputs? [y/n]: ')

plt.show()




