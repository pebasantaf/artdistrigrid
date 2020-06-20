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

# # IMPORT TESTS AND STORE THEM

test_path = r'C:\Users\Usuario\Documents\Universidad\TUM\Subjects\3rd semester\Research Internship\ANN_grids\Test'

test_files = glob.glob(test_path + "/*.csv")  # keeping directories in a list
n_test = len(test_files)  # number of files
inout = (13, 12)
tests = [None]*n_test

# create column names

index = np.arange(1, 11)
index = np.append(index, 0)
variables = [None]*len(index)
c = 0

# create the column variables
for i in index:
    variables[c] = 'V_{}'.format(i)
    c += 1

variables.extend(['i_P', 'i_Q'])

#
q = 0
for csv in test_files:
    pred_data = pd.read_csv(csv, sep=';', encoding='cp1252')
    pred_data = pred_data.drop(columns= 'All calculations', index=0)
    pred_data.columns = variables
    t_step = pred_data.shape[0]  # as all data has the same shape, we can keep these values for later use

    pred_input = pred_data
    pred_input.V_0 = pred_input.V_0.shift(1) #shift the input 1 value
    pred_input = pred_input.drop(index=1)

    pred_output = pred_data.drop(columns= 'V_0', index=1)


    tests[q] = [pred_input, pred_output]

    q += 1

# Prepare data for simulation

model = ('model1580475213.h5')

trained_model = load_model(
    r'C:\Users\Usuario\Documents\Universidad\TUM\Subjects\3rd semester\Research Internship\ANN_grids\model\{}'.
        format(model))

 #which test

t = 2

tests[t][0] = tests[t][0].astype(np.float32)

results = np.zeros((t_step-1, 13))
results = pd.DataFrame(results, columns=variables)
#results.V_0 = tests[t][0].V_0.values

# select initial row to act as 1st input

results.loc[[0]] = tests[t][0].loc[[2]].values
values = results.values

i = 0
#simulate the rest of the process

for row in values[1:, :]:

    tensor = np.reshape(values[i, :], (1, 1, 13))

    next_step = trained_model.predict(tensor)

    next_step = np.reshape(next_step, (1, 13))

    values[i+1, 0:10] = next_step[0][0:10]
    values[i+1, 11:13] = next_step[0][10:12]


    i += 1


plt.rcParams.update({'font.size': 16})
plt.plot(values[:, 11:13])
plt.plot(tests[t][1][['i_P','i_Q']].astype(float).values)
plt.xlabel('Time Steps')
plt.ylabel('I [p.u.]')
plt.legend(['Active Current pred.', 'Reactive Current pred.', 'Active Current', 'Reactive Current'])
plt.show()

