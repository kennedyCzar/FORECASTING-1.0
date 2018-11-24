# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 13:58:17 2018

@author: kennedy
"""
import numpy as np
import os
from os import chdir
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

#import keras libraries
#initialize recurrent neural network
from keras.models import Sequential 
#creates output layer of our RNN
from keras.layers import Dense, Activation
from keras.layers import Flatten, Convolution1D, Dropout
#Long Short Memory RNN
from keras.layers import LSTM
#import the keras GRU
from keras.layers import GRU
#Dense, Activation and Dropout
#from keras.layers.core import Dense, Activation, Dropout
#Convolutional 1D
from keras.layers.convolutional import Conv1D
#MaxPooling1D
from keras.layers.convolutional import MaxPooling1D
#imprt optimizer
from keras.optimizers import SGD

#set the directory path
chdir('D:\\GIT PROJECT\\ERIC_PROJECT101\\FREELANCE_KENNETH')
#def predictors(path, dataset):
'''Define data path and source'''
os.path.exists('DATASET/')
#set parh for the dataset
path = 'DATASET/'
test_data_path = 'TEST_DATA/'
#stock name
stock_name = 'AC.MX.csv'
#return dataset
dataset = pd.read_csv(path + stock_name, index_col='Date', parse_dates=True)
test_data = pd.read_csv(test_data_path + stock_name, index_col='Date', parse_dates=True)
'''High_Low Volatility change'''
#-------------------------------------TRAIN DATA-----------------------------------
dataset['HL_PCT'] = (dataset['High'] - dataset['Low'])/(dataset['Low']*100)
dataset['PCT_CHNG'] = (dataset['Close'] - dataset['Open'])/(dataset['Open']*100)

#-------------------------------------TEST DATA-----------------------------------
test_data['HL_PCT'] = (test_data['High'] - test_data['Low'])/(test_data['Low']*100)
test_data['PCT_CHNG'] = (test_data['Close'] - test_data['Open'])/(test_data['Open']*100)


#get the training data
training_set = dataset.iloc[:, [3, 6, 7]]
training_set.plot()

#get the test data
test_set = test_data.iloc[:, [3, 6, 7]]

#Normalize the dataset
MinMax_SC = MinMaxScaler()
#Normalize test set
MinMax_SC_test = MinMaxScaler()
'''In the end we will inverse transform this
training set to get back our original dataset

NOTE: that prices will now be in 0 and 1'''
#fit transform data
transformed_df = MinMax_SC.fit_transform(training_set)

#fit transform test_data
transformed_test_df = MinMax_SC_test.fit_transform(test_set)

#getting the input and output
#gets the training data at time t
X_train = transformed_df[0:len(transformed_df)-1]
#get transformed test_data
X_test = transformed_test_df[0:len(transformed_test_df)]
#y_train would be the stock price of our
#training set shifteed by 1
Y_train = transformed_df[1:len(transformed_df)]

#training set shifteed by 1
Y_test = transformed_test_df[1:len(transformed_test_df)]

#reshape input functions
'''Arguments for reshape
np.reshape(train_data, (m-dimension of the data, time_step, features))
'''
X_train_resh = np.reshape(X_train, (len(X_train), 1, 3))

#Reshape X_test
X_test_resh = np.reshape(X_test, (len(X_test), 1, 3))
#%% FIRST REGRESSOR LSTM

'''RESULT OF FIRST RNN REGRESSION
with Adam optimer:
loss: 0.0071 - acc: 0.9777
----------------------------------
with SGD optimizer:
loss: 0.0073 - acc: 0.9777
-----------------------------------
Adding the 'Close' give us the result
loss: 0.0051 - acc: 0.9219
----------------plus validation we have
loss: 0.0050 - acc: 0.8871 - val_loss: 0.0063 - val_acc: 0.9985
----------------------------------------------------------------
============Over 500iterations
loss: 0.0049 - acc: 0.8884 - val_loss: 0.0074 - val_acc: 0.9985
'''
#initiliaze sequential class object to predict continous outcome
#This is a regression model of RNN
regressor = Sequential() #sequence of layers
#Add input layer and LSTM layer
'''Arguments of the LSTM regressor:
  Units: Units of Neurons used in the LSTM
  Activation: function used to activate/transmit the neuron : e.g 'sigmoid', 'relu', 'tanh'
  input shape: None: to indicate that the model can accept any timestep: 2 is the number of feature
'''
regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 3)))

#Add output layer..a Dense layer
'''The output layer consist of High_Low Volatility change
  :HL_PCT:
  :PCT_CHNG:
'''
#output layers
regressor.add(Dense(units = 3, kernel_initializer='uniform', activation='sigmoid'))

#SGD optimzer
sgd = SGD(lr=0.01, nesterov=True, decay=1e-6, momentum=0.9)
#compile the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])

#fit regressor to training set
#regressor.fit(X_train_resh, Y_train, batch_size = 32, epochs = 200)
regressor.fit(X_train_resh, Y_train, batch_size = 32, epochs = 500, validation_split=0.3)
#throw the predicted stock price
predicted_stock = regressor.predict(X_test_resh)
#return the inverse of the predicted values..
predicted_inversed = MinMax_SC.inverse_transform(predicted_stock)
plt.plot(np.arange(0, len(predicted_inversed), 1), predicted_inversed[:, 0])

#%% PLOT THE CLOSE

plot_it = pd.DataFrame(predicted_inversed, columns = ['predicted_Close', 'Hl', 'PL_CHG'])
plot_it = plot_it.drop(['Hl', 'PL_CHG'], axis = 1)
init_close = pd.DataFrame(training_set['Close'].values, columns = ['Close'])
new = pd.concat([plot_it, init_close], axis = 1)
new.plot()
#%% GRU

'''RESULT OF FIRST RNN REGRESSION
with Adam optimer:
loss: 0.0071 - acc: 0.9777
----------------------------------
with SGD optimizer:
loss: 0.0073 - acc: 0.9777
-----------------------------------
Adding the 'Close' give us the result
loss: 0.0051 - acc: 0.9219
----------------plus validation we have
loss: 0.0050 - acc: 0.8858 - val_loss: 0.0076 - val_acc: 0.9985
----------------------------------------------------------------
==============Over 500iterations
loss: 0.0048 - acc: 0.8877 - val_loss: 0.0053 - val_acc: 0.9985
==============Over 1000iterations
loss: 0.0048 - acc: 0.8877 - val_loss: 0.0055 - val_acc: 0.9985

'''
#initiliaze sequential class object to predict continous outcome
#This is a regression model of RNN
regressor_GRU = Sequential() #sequence of layers
#Add input layer and LSTM layer
'''Arguments of the LSTM regressor:
  Units: Units of Neurons used in the LSTM
  Activation: function used to activate/transmit the neuron : e.g 'sigmoid', 'relu', 'tanh'
  input shape: None: to indicate that the model can accept any timestep: 2 is the number of feature
'''
regressor_GRU.add(GRU(units = 4, activation = 'tanh', input_shape = (None, 3)))

#Add output layer..a Dense layer
'''The output layer consist of High_Low Volatility change
  :HL_PCT:
  :PCT_CHNG:
'''
#output layers
regressor_GRU.add(Dense(units = 3, kernel_initializer='uniform', activation='sigmoid'))

#SGD optimzer
sgd = SGD(lr=0.01, nesterov=True, decay=1e-6, momentum=0.9)
#compile the RNN
regressor_GRU.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])

#fit regressor to training set
#regressor.fit(X_train_resh, Y_train, batch_size = 32, epochs = 200)
regressor_GRU.fit(X_train_resh, Y_train, batch_size = 32, epochs = 500, validation_split=0.3)
#throw the predicted stock price
predicted_stock = regressor_GRU.predict(X_test_resh)
#return the inverse of the predicted values..
predicted_inversed = MinMax_SC.inverse_transform(predicted_stock)
#%% RESULT OF SECOND RNN RESULT

'''
loss: 0.0072 - acc: 0.9779 - val_loss: 0.0068 - val_acc: 0.9773
---------------------------------------------------
yOU WOULD NOTICE THE LOSS IS PRETTY HIGH EXACTLY .0002 LESS
BETTER THAN THE FIRST RNN
--------------------------------------------------
=========With 'close' feature vector================
loss: 0.0052 - acc: 0.8741 - val_loss: 0.0200 - val_acc: 0.9985

'''
regressor_model2 = Sequential() #sequence of layers
#Add input layer and LSTM layer
'''Arguments of the LSTM regressor:
  Units: Units of Neurons used in the LSTM
  Activation: function used to activate/transmit the neuron : e.g 'sigmoid', 'relu', 'tanh'
  input shape: None: to indicate that the model can accept any timestep: 2 is the number of feature
'''
regressor_model2.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 3)))

#Add output layer..a Dense layer
'''The output layer consist of High_Low Volatility change
  :HL_PCT:
  :PCT_CHNG:
'''
regressor_model2.add(Dense(units = 8, input_dim=3, kernel_initializer='uniform', activation='sigmoid'))
#Add two inner layes
regressor_model2.add(Dense(units = 12, input_dim=3, kernel_initializer='uniform', activation='sigmoid'))
regressor_model2.add(Dense(units = 8, input_dim=3, kernel_initializer='uniform', activation='sigmoid'))

#output layers
regressor_model2.add(Dense(units = 3, kernel_initializer='uniform', activation='sigmoid'))

#compi;e the RNN
regressor_model2.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])



#fit regressor to training set
regressor_model2.fit(X_train_resh, Y_train, batch_size = 32, epochs = 200, validation_split=0.3)


final_refessor = (regressor + regressor_model2)/2
#save the trained regressor

#%% RESULT OF RNN AND CNN

'''
loss: 0.0072 - acc: 0.9779 - val_loss: 0.0068 - val_acc: 0.9773
--------------------------------------------------
We observed the addition of a CNN  into this network improved the network 
performance by reducing the loss by 0.002 as compared to just using RNN
'''
#INSTANTIANTIATE SEQUENTIAL CLASS
regressor_model3 = Sequential()
#Add input LSTM  layer
regressor_model3.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 2), return_sequences=True))
#Add dropout
regressor_model3.add(Dropout(0.2))
#Add CNN layer
regressor_model3.add(Conv1D(32, padding = 'same', activation='sigmoid'))
#Add max pooling
regressor_model3.add(MaxPooling1D(pool_size=2))
#Add another LSTM layer
regressor_model3.add(LSTM(units = 4, return_sequences=False))
#Add a dropout
regressor_model3.add(Dropout(0.2))

regressor_model3.add(Dense(units = 2))  # Linear dense layer to aggregate into 1 val

regressor_model3.add(Activation('linear'))
timer_start = time.time()
regressor_model3.compile(loss='mean_squared_error', optimizer='adam')
print('Model built in: ', time.time()-timer_start)
# Training model
regressor_model3.fit(X_train_resh, Y_train, batch_size = 32, epochs = 200, validation_split=0.3)

#%% CONVOLUTIONAL NEURAL NETWORK

'''
loss: 0.0564 - acc: 0.9779 - val_loss: 0.0539 - val_acc: 0.9773
------------------------------------------------------------
We observe although CNN has an equally good accuracy as the RNN network, it has a higher 
loss function than RNN and this will turn out not so for our prediction.
--------------------------------------------------
=========With 'close' feature vector================

loss: 0.0175 - acc: 0.5587 - val_loss: 0.0697 - val_acc: 0.0015
'''
# Keras model with one Convolution1D layer
# unfortunately more number of covnolutional layers, filters and filters lenght 
# don't give better accuracy
regressor_model4 = Sequential()
regressor_model4.add(Convolution1D(nb_filter=512, filter_length=1, input_shape=(1, 3)))
regressor_model4.add(Activation('sigmoid'))
regressor_model4.add(Flatten())
regressor_model4.add(Dropout(0.4))
regressor_model4.add(Dense(2048, activation='sigmoid'))
regressor_model4.add(Dense(1024, activation='sigmoid'))
regressor_model4.add(Dense(3))
regressor_model4.add(Activation('softmax'))


sgd = SGD(lr=0.01, nesterov=True, decay=1e-6, momentum=0.9)
regressor_model4.compile(loss='mean_squared_error',optimizer=sgd,metrics=['accuracy'])

regressor_model4.fit(X_train_resh, Y_train, epochs = 200, batch_size=16, validation_split=0.3)










