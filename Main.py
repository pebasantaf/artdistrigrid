from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, optimizers
import time
# own modules and classes
from functions.dataManager import *
from functions.dataPlot import *
from classes.EarlyStoppingByLossVal import EarlyStoppingByLossVal

''' PATHS '''
train_path = r'C:\Users\Usuario\Documents\Universidad\TUM\Subjects\3rd semester\Research Internship\ANN_grids\Data02'
test_path = r'C:\Users\Usuario\Documents\Universidad\TUM\Subjects\3rd semester\Research Internship\ANN_grids\Test'
model_path = r'C:\Users\Usuario\Documents\Universidad\TUM\Subjects\3rd semester\Research Internship\ANN_grids\model'

''' USER INPUT '''
inout = [13, 13]
look_back = 1
drop = ['All calculations']
loss_functions = 'mean_squared_error'
metrics = ['mse', 'mae']
epochs = 15
validation_split = 0.25

''' DATA IMPORT '''
Tests, t_step = importTestData(test_path, look_back, drop)
X, Y = importTrainData(train_path, look_back, t_step, drop, inout)

''' MODEL DEFINITION '''
model = Sequential()

l1 = model.add(layers.LSTM(26, input_shape=(look_back, inout[0]), activation='relu', return_sequences=True)) #adding a RNN layer
l2 = model.add(layers.TimeDistributed(layers.Dense(5)))
l3 = model.add(layers.Dropout(0.2))
l4 = model.add(layers.Dense(inout[1])) #fully connected layer. What i would understand as a normal layer

opt = optimizers.RMSprop(lr=1e-03) #how fast the learning rate decays. this helps finding the miminum better

callbacks = [EarlyStoppingByLossVal(monitor='val_loss', value=0.002),
     ModelCheckpoint(filepath=model_path.format(int(time.time())), save_best_only=True)]

#compiling the model. Defining some of sthe features for the fit like the type of loss function, the optimizer and metrics that are interesting for us

model.compile(loss=loss_functions,
             metrics=metrics)  # accuracy only valid for classification tasks

history = model.fit(X, Y, epochs=epochs, validation_split=validation_split, callbacks=callbacks, batch_size=200)