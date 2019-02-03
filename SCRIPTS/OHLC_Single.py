# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 03:40:13 2019

@author: kennedy
"""
#imort default
from __future__ import division, print_function

seed = 1333
from numpy.random import seed
seed(19)
from tensorflow import set_random_seed
set_random_seed(19)

import os
from STOCK import stock, loc
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None
import lightgbm as lgb
from datetime import datetime
import matplotlib.pyplot as plt
from Preprocess import process_time
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import (AdaBoostRegressor, #Adaboost regressor
                              RandomForestRegressor, #Random forest regressor
                              GradientBoostingRegressor, #Gradient boosting
                              BaggingRegressor, #Bagging regressor
                              ExtraTreesRegressor) #Extratrees regressor


#get ojects in the dataset folder and 
#strip extension
def ls_STOK():
  '''
  :Return:
    List of stock in dataset
  '''
  DIR_OBJ = os.listdir()
  STOK_list_ = []
  for x in range(len(DIR_OBJ)):
    STOK_list_.append(DIR_OBJ[x].strip('.csv'))
    
  return STOK_list_


def window(MIN_LAG, MAX_LAG, 
           STEP, STOCK_name, price, next_day):
  '''
  :Arguments:
      MIN_LAG :
        
      MAX_LAG :
        
      STEP :
        
      STOCK_index_ :
        
      price :
        
  :Return:
    
  '''

  
  
  #get the data we need
  data = loc.read_csv(STOCK_name + str('.csv'))
  #the stock class
  stock_data = stock(data)
  #get OHLC, HL_PCT, PCT_CHNG
  df_OHLC = pd.concat([stock_data.OHLC(), stock_data.HL_PCT()], axis = 1)
  #listing...
  
  process_time(df_OHLC)
  #extract specific time features

  #forecastiing parameter
  Day_of_week = [x for x in df_OHLC.DayOfTheWeek.unique()]  
  #forecast dates ahead minus holidays
  def filter_Mexico_holidays(df, nextday):
    
    '''
    :Arguments:
      df: dataframe
    :Nextday:
      datetime format
      
    :Return:
      List of filtered dates using US calender
    '''
    import holidays
    #holidays in Mexico
    us_holidays = holidays.Mexico()
    hol_dates = []
    dat_frac = list((pd.bdate_range(pd.to_datetime(df_OHLC.index[-1]), next_day)).date)
    #iterate using date index
    for ii in range(len(dat_frac)):
      print(dat_frac[ii])
      if isinstance(us_holidays.get(dat_frac[ii]), str):
        hol_dates.append(dat_frac[ii])
    
    if hol_dates == []:
      print('No holidays')
    else:
      for ii in hol_dates:
        print('Holiday present on {}'.format(ii))
        
    dat_frac = sorted([x for x in set(dat_frac).difference(set(hol_dates))])[1:]
          
        
    return (dat_frac, hol_dates)
  
  print('*'*30)
  print('Fininshed extracting holidays')
  
  dt_range, hol_dates = filter_Mexico_holidays(df_OHLC, next_day)
  trad_days = len(dt_range)
  
  
  #all series
  al_dt = list(df_OHLC.index) + list(dt_range)
  #lagged_time series
  forecast_window = pd.DataFrame(al_dt, columns = ['timestamp'])
  forecast_window.set_index('timestamp', inplace = True)
  
  print('Forecast ahead dates created')
  '''
  Prrice shift
  '''
  
  print('Create laggs..')
  
  def lagg(param, df, t_days):
    '''
    :Arguments:
      :param:
        feature
      :df:
        dataframe
      :t_days:
        forecast window/days ahead
        
    :Return:
      values at time t and t+x where x = 1,...,n
      
    '''
    df_param_t = list(df[param])
    df_param_t_1 = list(df[param].shift(t_days))
    df_param_t_plus = list(df.ix[-t_days:, param])
    
    return df_param_t_1, df_param_t_plus
  
  
  '''
  HL_PCT shift
  '''
  #create laggs for every feature in stock i.e OHLC
  df_ = {}
  params = [price, 'HL_PCT', 'PCT_CHNG']
  
  '''create a loop here for forecasting each feature'''
  for ii in params:
    df_['df_{}_t_1'.format(ii)], df_['df_{}_t_plus'.format(ii)] = lagg(ii, df_OHLC, trad_days)
  
  
  #lagged
  for ii in df_['df_{}_t_plus'.format(params[0])]:
    df_['df_{}_t_1'.format(params[0])].append(ii)
  
  for ii in df_['df_{}_t_plus'.format(params[1])]:
    df_['df_{}_t_1'.format(params[1])].append(ii)
    
  for ii in df_['df_{}_t_plus'.format(params[2])]:
    df_['df_{}_t_1'.format(params[2])].append(ii)
  
  #create the forecast laggs
  for ii, val in df_.items():
    if len(val) > trad_days:
#      if len(val) == len
      forecast_window['lagged_'+ str('{}'.format(ii))] = val
#    else:
#      raise('Incorrect data setting.\nDate should be shifted forward..')
  forecast_window = forecast_window.dropna()
  
  #convert to stock class
  #Exponential laggs for each feature
  EWM_m = {}
  for w in forecast_window.columns:
    for ij in range(MIN_LAG, MAX_LAG, STEP):
      EWM_m['{}_{}'.format(w, ij)] = forecast_window[w].ewm(ij).mean()
    
  for p, q in EWM_m.items():
    forecast_window['{}'.format(p)] = q
  
  
  #delta time
  time_dt = pd.DataFrame({'timestamp':forecast_window.index})
  process_time(time_dt).set_index('timestamp', inplace = True)
  
  #filter weekends from data
  time_dt = time_dt.loc[time_dt.DayOfTheWeek.isin(Day_of_week)]
  #keep feature columns
  time_dt = time_dt.loc[:, [x for x in OHLC_features_]]
  forecast_window = pd.concat([forecast_window, time_dt], axis = 1)
  
  
  return (forecast_window, trad_days, dt_range)


def Scale_train_test(window, tr_days):
  '''
  :Arguments:
    :window:
      forecast window dataset
    :tr_days:
      allowed trading days
  :Return:
    :X_train:
      transformed X 70%
    :X_test:
      transformed X 30%
    :Y_train:
      untransformed Y 70%
    :Y_test:
      untransformed Y 30%
  '''
  if np.where(window.values >= np.finfo(np.float64).max)[1] == [] is True:
    X_transform = pd.DataFrame(StandardScaler().fit_transform(window),
                               columns = [x for x in window.columns])
  else:
    X_transform = pd.DataFrame(StandardScaler().fit_transform(np.where(window.values\
                               >= np.finfo(np.float64).max,0, window)),
                               columns = [x for x in window.columns])
    
  X_train = X_transform.iloc[:-tr_days, 1:]
  Y_train = window.iloc[:-tr_days, 0].values
  X_test = X_transform.iloc[-tr_days:, 1:]
  Y_test = window.iloc[-tr_days:, 0].values
  
  return X_train, X_test, Y_train, Y_test


#%% Modelling

def RNN(data, epochs):
  #import required libaries
  from sklearn.preprocessing import MinMaxScaler
  from keras.models import Sequential 
  from keras.layers import Dense
  from keras.layers import LSTM
  from keras.optimizers import SGD
  from keras.callbacks import (ReduceLROnPlateau, 
                               ModelCheckpoint)

  MinMax_SC = MinMaxScaler()
  
  if np.where(data.values >= np.finfo(np.float64).max)[1] == [] is True:
    transformed_df = MinMax_SC.fit_transform(data)
  else:
    transformed_df = MinMax_SC.fit_transform(np.where(data.values \
                                                      >= np.finfo(np.float64).max,0, data))

  X_train = transformed_df[:-trad_days, 1:]
  Y_train = MinMax_SC.fit_transform(pd.DataFrame(data.iloc[:-trad_days, 0].values))
  X_test = transformed_df[-trad_days:, 1:]
  Y_test = MinMax_SC.fit_transform(pd.DataFrame(data.iloc[-trad_days:, 0].values))
  #reshape and regress
  X_train_resh = np.reshape(X_train, (len(X_train), 1, X_train.shape[1]))
  X_test_resh = np.reshape(X_test, (len(X_test), 1, X_train.shape[1]))
  
  regressor = Sequential() #sequence of layers
  regressor.add(LSTM(units = 4, activation = 'relu', input_shape = (None, X_train.shape[1])))
  regressor.add(Dense(units = 1, kernel_initializer='uniform', activation='sigmoid'))
  sgd = SGD(lr=0.1, nesterov=True, decay=1e-6, momentum=0.9)
  reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=5, min_lr=0.000001, verbose=1)
  checkpointer = ModelCheckpoint(filepath="../MODEL/test.h5", verbose=1, save_best_only=True)
  regressor.compile(optimizer = sgd, loss = ['mean_squared_error'],
                    metrics=['accuracy', 'categorical_accuracy'])
  
  regressor.fit(X_train_resh, Y_train, batch_size = len(X_train), 
                epochs = epochs, validation_split=0.3, 
                callbacks=[reduce_lr, checkpointer])
  
  
  predicted_stock = regressor.predict(X_test_resh)
  predicted_inversed = MinMax_SC.inverse_transform(predicted_stock)
  
  return predicted_inversed

def Modeller(X_train, X_test, Y_train, Y_test, dt_, params, epochs):
#  
  #required by LBGM
  train_data=lgb.Dataset(X_train,Y_train)
  valid_data=lgb.Dataset(X_test,Y_test)
  
  if X_train.shape[0] < params['min_child_samples']//2 or X_train.shape[0] > params['min_child_samples']//3:
    params['min_child_samples'] //=100
    params['n_estimators'] //=1
  elif X_train.shape[0] < params['min_child_samples']//4 or X_train.shape[0] > params['min_child_samples']//5:
    params['min_child_samples'] //=400
    params['n_estimators'] //= 4
  elif X_train.shape[0] < params['min_child_samples']//5 or X_train.shape[0] > params['min_child_samples']//6:
    params['min_child_samples'] //=400
    params['n_estimators'] //=5
  elif X_train.shape[0] < params['min_child_samples']//7 or X_train.shape[0] > params['min_child_samples']//8:
    params['min_child_samples'] //=400
    params['n_estimators'] //=5
  elif X_train.shape[0] < params['min_child_samples']//8 or X_train.shape[0] > params['min_child_samples']//9:
    params['min_child_samples'] //=400
    params['n_estimators'] //=6
  elif X_train.shape[0] < params['min_child_samples']//10 or X_train.shape[0] > params['min_child_samples']//11:
    params['min_child_samples'] //=400
    params['n_estimators'] //=6
  elif X_train.shape[0] < params['min_child_samples']//12 or X_train.shape[0] > params['min_child_samples']//13:
    params['min_child_samples'] //=400
    params['n_estimators'] //=6
  elif X_train.shape[0] < params['min_child_samples']//14 or X_train.shape[0] > params['min_child_samples']//14:
    params['min_child_samples'] //=400
    params['n_estimators'] //=6
  elif X_train.shape[0] < params['min_child_samples']//15 or X_train.shape[0] > params['min_child_samples']//16:
    params['min_child_samples'] //=400
    params['n_estimators'] //=6
  elif X_train.shape[0] < params['min_child_samples']//17 :
    params['min_child_samples'] //=400
    params['n_estimators'] //=6
  else:
    params['min_child_samples']
    params['n_estimators']
  
  Regress1 = RandomForestRegressor(max_depth = params['max_depth'], 
                                   random_state = params['random_state'], 
                                   n_estimators = params['n_estimators'])
  
  Regress2 = GradientBoostingRegressor(learning_rate = params['learning_rate'], 
                                       loss = params['loss'],
                                       n_estimators = params['n_estimators'])
  
  Regress3 = ExtraTreesRegressor(max_depth = params['max_depth'], 
                                 random_state = params['random_state'], 
                                 n_estimators = params['n_estimators'])
  
    
  Regress4 = XGBRegressor(max_depth = params['max_depth'], 
                          n_estimators = params['n_estimators'],
                          min_child_weight = params['min_child_weight'], 
                          colsample_bytree = params['colsample_bytree'],
                          subsample = params['subsample'],
                          eta = params['eta'], 
                          seed = params['seed'])
  
  
  
  Regress1.fit(X_train, Y_train)
  Regress2.fit(X_train, Y_train)
  Regress3.fit(X_train, Y_train)
  Regress4.fit(X_train, Y_train, eval_metric="rmse")
  

  print('Parameter value: {}\nN_estimators:{}'.format(params['min_child_samples'], params['n_estimators']))

    
  Regress5 = lgb.train(params, train_data, 
                       valid_sets = [train_data, valid_data],
                       num_boost_round = 2500)
  
  Predic_ = Regress1.predict(X_test)
  Predic_2 = Regress2.predict(X_test)
  Predic_3 = Regress3.predict(X_test)
  Predic_4 = Regress4.predict(X_test)
  Predic_5 = Regress5.predict(X_test)
  Predic_6 = [x[0] for x in RNN(forecast_window, epochs)]
  
  forcast_date = pd.DataFrame({'timestamp': dt_, 
                               'RandForest_{}_Projection'.format(price): Predic_,
                               'GradBoost_{}_Projection'.format(price): Predic_2,
                               'ExtraTrees_{}_Projection'.format(price): Predic_3,
                               'XGB_{}_Projection'.format(price): Predic_4,
                               'LGB_{}_Projection'.format(price): Predic_5,
                               'RNN_{}_Projection'.format(price): Predic_6})
      
  forcast_date['Average_{}_Projection'.format(price)] = forcast_date.mean(axis = 1)
  forcast_date.set_index('timestamp', inplace = True)
  return forcast_date
#  return (dt_, Predic_close, Predic_close2, Predic_close3,
#          Predic_close4, Predic_close5, Predic_close6)

  


    
if __name__ == '__main__':
  #set global parameters
  MIN_LAG = 5
  MAX_LAG = 25
  STEP = 5
  STOCK_name ='ALSEA.MX'
  price = 'High'
  #Select Hyper-Parameters
  EPOCHS = 100
  params = {'metric' : 'auc',
            'max_depth': 10,
            'learning_rate': 0.1,
            'boosting_type' : 'gbdt',
            'colsample_bytree' : 0.8,
            'num_leaves' : 20,
            'eta': 0.3,
            'seed': 19,
            'n_estimators' : 100,
            'min_child_samples': 400, 
            'min_child_weight': 0.1,
            'reg_alpha': 2,
            'reg_lambda': 5,
            'subsample': 0.8,
            'verbose' : -1,
            'num_threads' : 4,
            'random_state': 0,
            'loss': 'ls'
            }
  
  next_day = datetime(2019, 2, 5)
  OHLC_features_ = ['years', #trading year
                  'days', #trading days
                  'months', #months
                  'DayOfTheWeek', #days of week
                  'time_epoch', #time epoch
                  'wday_sin', #sine of trading day
                  'wday_cos', #cosine of trading day
                  'mday_sin', #sine of days of the month
                  'mday_cos', #cosine of days of the month
                  'yday_sin', #sine of day of year
                  'yday_cos', #cosine of day of year
                  'month_sin', #sine of month
                  'month_cos'] #cosine of month
  
  #set working directory
  loc.set_path('D:\\BITBUCKET_PROJECTS\\Forecasting 1.0\\DATASET')
  #stock list
  STOCK_list_ = ls_STOK()
  #window
  forecast = {}
  #//Extract Forecast window
  forecast_window, trad_days, dt_range= window(MIN_LAG, MAX_LAG, STEP, STOCK_name, price, next_day)
  #train test
  X_train, X_test, Y_train, Y_test = Scale_train_test(forecast_window, trad_days)
  #yhat for all models
  Forecast = Modeller(X_train, X_test, Y_train, Y_test, dt_range, params, EPOCHS)
  
  #save to csv
  loc.set_path('D:\\BITBUCKET_PROJECTS\\Forecasting 1.0\\PREDICTED')
  Forecast.to_csv('{}_Single_forecasting.csv'.format(STOCK_name.strip('.MX')))




