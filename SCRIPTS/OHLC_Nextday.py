# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 19:02:46 2018

@author: kennedy
"""

from STOCK import stock, loc
import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
from Preprocess import process_time
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

'''
Create date holder
start date: indicates the last date in the series
End date: indicates the numbers of days we want to project into
'''


#set directory to fetch data
loc.set_path('D:\\GIT PROJECT\\ERIC_PROJECT101\\FREELANCE_KENNETH\\DATASET')



def predict_OHLC(NXT_DAY):
  '''
  :Arguments:
    STOCKLIST: List of downloaded stock in the dataset folder
    NXTDAY: nextday to predict
    stock_data: stock class
    
  :Return:
    Next day Open, High, Low, Close for all stock
    
  '''
  #get ojects in the dataset folder and 
  #strip extension
  
  DIR_OBJ = os.listdir()
  STOCK_list_ = []
  
  for x in range(len(DIR_OBJ)):
    STOCK_list_.append(DIR_OBJ[x].strip('.csv'))
  
  MIN_LAG = 5
  MAX_LAG = 25
  STEP = 5
  
#  process_time(df_OHLC)
  
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
  
 
  '''
  recall later
  dt_range = []
  for ii in range(1, window):
    if list(pd.to_datetime(df_OHLC.index[-1]) + pd.to_timedelta(np.arange(window+1), 'D'))[ii].dayofweek in Day_of_week:
      dt_range.append(list(pd.to_datetime(df_OHLC.index[-1]) + pd.to_timedelta(np.arange(window+1), 'D'))[ii])
    else:
      pass
  '''
  
  
  NXT_open_ = []
  NXT_high_ = []
  NXT_low_ = []
  NXT_close_ = []
  
  for _i_ in range(len(STOCK_list_)):
    #get the data we need
    data = loc.read_csv(STOCK_list_[_i_] + str('.csv'))
    #the stock class
    stock_data = stock(data)
    #get OHLC
    df_OHLC = stock_data.OHLC()
  
    '''listing...'''
    
    process_time(df_OHLC)
    #forecastiing parameter
    Day_of_week = [x for x in df_OHLC.DayOfTheWeek.unique()]
    
    #NXT_DAY = datetime(2018, 12, 21)
    dt_range = pd.bdate_range(pd.to_datetime(df_OHLC.index[-1]), NXT_DAY)[1:]
    
    trad_days = len(dt_range)
    
    #all series
#    df_dt = list(df_OHLC.index)
#    for ii in list(dt_range):
#      df_dt.append(list(ii))
    al_dt = list(df_OHLC.index) + list(dt_range)
    #lagged_time series
    forecast_window = pd.DataFrame({'timestamp': al_dt})
    forecast_window.set_index('timestamp', inplace = True)
  
    '''
    Get Open High Low Close
    '''
    
    for _ix in df_OHLC.columns:
      '''extract columns to forecast'''
      
      df_t = list(df_OHLC[_ix])
      df_t_1 = list(df_OHLC[_ix].shift(trad_days))
      df_t_plus = list(df_OHLC[_ix][-trad_days:])
    
      #lagged
      for w in df_t_plus:
        df_t_1.append(w)
      
      forecast_window['lagged_'+str('{}days'.format(str('t')))] = df_t_1
      forecast_window = forecast_window.dropna()
      
      #convert to stock class
      for ij in range(MIN_LAG, MAX_LAG, STEP):
        forecast_window['lagged_t_'+str('{}'.format(ij))] = forecast_window.lagged_tdays.ewm(ij).mean()
      
      #delta time
      time_dt = pd.DataFrame({'timestamp': forecast_window.index})
      process_time(time_dt).set_index('timestamp', inplace = True)
      
      #filter weekends from data
      time_dt = time_dt.loc[time_dt.DayOfTheWeek.isin(Day_of_week)]
      #keep feature columns
      time_dt = time_dt.loc[:, [x for x in OHLC_features_]]
      forecast_window = pd.concat([forecast_window, time_dt], axis = 1)
      
      
      #standardize
      X_transform = pd.DataFrame(StandardScaler().fit_transform(forecast_window),
                                 columns = [x for x in forecast_window.columns])
      
      #train test splits
      X_train = X_transform.iloc[:-trad_days, 1:]
      Y_train = forecast_window.lagged_tdays[:-trad_days].values
      X_test = X_transform.iloc[-trad_days:, 1:]
      Y_test = forecast_window.lagged_tdays[-trad_days:].values
      
      #model
      Regress = RandomForestRegressor(max_depth = 20, random_state = 0,
                                      n_estimators = 100)
      #fit model
      Regress.fit(X_train, Y_train)
      #predict feature
      Predic_ = Regress.predict(X_test)
      
      if _ix == 'Open':
        NXT_open_.append(Predic_)
      elif _ix == 'High':
        NXT_high_.append(Predic_)
      elif _ix == 'Low':
        NXT_low_.append(Predic_)
      elif _ix == 'Close':
        NXT_close_.append(Predic_)
        
  return pd.DataFrame({'Stocks': STOCK_list_, 'Next_day_Open': NXT_open_,
                            'Next_day_High': NXT_high_, 'Next_day_Low': NXT_low_,
                            'Next_day_Close': NXT_close_})
  

        
      
predict_OHLC(datetime(2018, 12, 21))   
    
    
      
  






















  