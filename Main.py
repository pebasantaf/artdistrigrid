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
from functions.dataManager import *

inout = [13, 13]
look_back = 1
dropin = ['All calculations']
dropout = ['All calculations']
train_path = r'C:\Users\Usuario\Documents\Universidad\TUM\Subjects\3rd semester\Research Internship\ANN_grids\Data02'
test_path = r'C:\Users\Usuario\Documents\Universidad\TUM\Subjects\3rd semester\Research Internship\ANN_grids\Test'

Tests, t_step = importTestData(test_path, look_back, dropin, dropout)

X, Y = importTrainData(train_path, look_back, t_step, dropin, dropout, inout)


model = Sequential()

model.add(layers.GRU(8, batch_input_shape=(292, 1, inout[0]), activation='relu', return_sequences=True)) #adding a RNN layer
model.add(layers.TimeDistributed(layers.Dense(inout[0])))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(2)) #fully connected layer. What i would understand as a normal layer

opt = optimizers.Adam(lr=1e-03, beta_1=0.95) #how fast the learning rate decays. this helps finding the miminum better

callbacks = [EarlyStoppingByLossVal('val_loss', value=0.00001),
             ModelCheckpoint(filepath=model_path.format(int(time.time())), save_best_only=True)]
#

#compiling the model. Defining some of sthe features for the fit like the type of loss function, the optimizer and metrics that are interesting for us
model.compile(loss='mean_squared_logarithmic_error',
             optimizer=opt,
             metrics=['mse', 'mae'])  # accuracy only valid for clasiffication tasks

history = model.fit(X, Y, epochs=150, validation_split=0.2, callbacks=callbacks)