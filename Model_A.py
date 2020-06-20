from __future__ import absolute_import, division, print_function, unicode_literals

import os
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, optimizers
import numpy as np
import matplotlib.pyplot as plt
import time
from math import sqrt
import keras
from functions.dataManager import importTrainData, importTestData

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #this removes some warning comments. This warning comments express that this PC has a CPU able to
#compute much faster, and that tensorflow was not designed for it. For the moment, will keep it like this. If necessary, we'll use GPU


keras.backend.reset_uids()
inout = [13, 13]
look_back = 1
dropin = ['All calculations']
dropout = ['All calculations']
train_path = r'C:\Users\Usuario\Documents\Universidad\TUM\Subjects\3rd semester\Research Internship\ANN_grids\Data02'
test_path = r'C:\Users\Usuario\Documents\Universidad\TUM\Subjects\3rd semester\Research Internship\ANN_grids\Test'

#Early stopping based on loss
tests, t_step = importTestData(test_path, look_back, dropin, dropout) #test is a list with test inputs and outputs.

X, Y = importTrainData(train_path, look_back, t_step, dropin, dropout, inout)

# # PATHS


test_path = r'I:\05_Basanta Franco\Python\Test'
train_path = r'I:\05_Basanta Franco\Python\Data02'
model_path = r'I:\05_Basanta Franco\Python\model\model01\model{}.h5'

paths = [test_path, train_path, model_path]


# # IMPORT DATA

col = ['All calculations', 'MSNS-Trafo', 'MSNS-Trafo.1']
row_drop = 0
inout = [11, 2]

test, inputs, targets, n_test, t_step = data_manager(paths, col, row_drop, inout) #test is a list with test inputs and outputs.

# # NEURAL NETWORK CREATOR

# Creating a model, which is a linear stack of layers
model = Sequential()

'''
LSTM layer of n nodes. Shape of the input is the columns of inputs. activation function is rectifier linear function.
Return sequencies = true basically tells the layer to output a sequence. If we were to have another Recurrent layer, this is necessary. Else not, as it would not understand it
Time distribute is important. That basically relates every input step in the input sequence with its corresponding output. 
Other way we would just be considering the last value of the sequence
'''


l1 = model.add(layers.GRU(8, batch_input_shape=(292, 1, inout[0]), activation='relu', return_sequences=True)) #adding a RNN layer
model.add(layers.TimeDistributed(layers.Dense(inout[0])))


model.add(layers.Dropout(0.2))

l5 = model.add(layers.Dense(2)) #fully connected layer. What i would understand as a normal layer

opt = optimizers.Adam(lr=1e-03, beta_1=0.95) #how fast the learning rate decays. this helps finding the miminum better

callbacks = [EarlyStoppingByLossVal('val_loss', value=0.00001),
             ModelCheckpoint(filepath=model_path.format(int(time.time())), save_best_only=True)]
#

#compiling the model. Defining some of sthe features for the fit like the type of loss function, the optimizer and metrics that are interesting for us
model.compile(loss='mean_squared_logarithmic_error',
             optimizer=opt,
             metrics=['mse', 'mae'])  # accuracy only valid for clasiffication tasks

history = model.fit(inputs, targets, epochs=150, validation_split=0.2, callbacks=callbacks)


# Evaluate the model

scores = model.evaluate(inputs, targets, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# print a summary of the outputs of every layer

print(model.summary())
weights = model.get_weights()



#SAVING THE MODEL

#
#The model is saved by modelcheckpoint in a folder. Here, we are saving the models arquitecture in a json file
#model_json = model.to_json()
#with open("model/model01/model.json", "w") as json_file:
#    json_file.write(model_json)

# PREDICTIONS WITH THE MODEL

t = 1
fig1 = plt.figure()
for prediction in test:

    NN_pred = model.predict(prediction[0])


    #ANALYSIS
    #reshape the prediction for plotting
    NN_pred = np.reshape(NN_pred, (prediction[1].shape[0], inout[1]))
    prediction[0] = np.reshape(prediction[0], (t_step-1, inout[0]))


    #plots: top, predicted and desired test output. down, test inputs

    plt.subplot(n_test, 1, t)
    plt.title('Test0' + np.str(t))
    plt.plot(NN_pred)
    plt.plot(prediction[1])
    plt.legend(['I_real_pred', 'I_im_pred', 'Ir', 'Ii'])


    # mean squared error
    rmse = sqrt(mean_squared_error(prediction[1], NN_pred))
    print('Test RMSE: %.3f' % rmse)

    t = t + 1

fig2 = plt.figure()

# plot loss during training
plt.subplot(211)
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
# plot mse during training
plt.subplot(212)
plt.title('Mean Squared Error')
plt.plot(history.history['mse'], label='train')
plt.plot(history.history['val_mse'], label='test')


#  print inputs yes or no
printin = input('Print inputs as well? [y/n]: ')

m = True
while m == True:

    if printin == 'y':

        t = 1
        fig3 = plt.figure()

        for prediction in test:
            plt.title('Inputs: V, P, Q')
            plt.subplot(n_test, 1, t)
            plt.plot(prediction[0])
            t = t + 1
        m = False
    elif printin == 'n':

        m = False
    else:

        printin = input('Answer not valid. Print inputs? [y/n]: ')

plt.show()
keras.backend.reset_uids()
keras.backend.clear_session()




