# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 15:13:19 2019

@author: kennedy
"""

__author__ = "kennedy Czar"
__email__ = "kennedyczar@gmail.com"
__version__ = '1.0'

seed = 1333
from numpy.random import seed
seed(19)
from tensorflow import set_random_seed
set_random_seed(19)

import os
from STOCK import stock, loc
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
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

#%% SIGNAL GENERATOR --> MACD, BOLLINGER BAND, RSI


##RSI signal
def RSI_signal(STK_data, period, lw_bound, up_bound):
  '''
  :Arguments:
    df:
  :Return type:
    signal
  '''
  stock_data = stock(STK_data)
  OHLC = stock_data.OHLC()
  df = stock_data.CutlerRSI(OHLC, period)
  try:
    assert isinstance(df, pd.Series) or isinstance(df, pd.DataFrame)
    #convert to dataframe
    if isinstance(df, pd.Series):
      df = df.to_frame()
    else:
      pass
    #get signal
    #1--> indicates buy position
    #0 --> indicates sell posotion
    df['signal'] = np.zeros(df.shape[0])
    pos = 0
    for ij in df.loc[:, ['RSI_Cutler_'+str(period)]].values:
      print(df.loc[:, ['RSI_Cutler_'+str(period)]].values[pos])
      if df.loc[:, ['RSI_Cutler_'+str(period)]].values[pos] >= up_bound:
        df['signal'][pos:] = 1 #uptrend
      elif df.loc[:, ['RSI_Cutler_'+str(period)]].values[pos] <= lw_bound:
        df['signal'][pos:] = 0 #downtrend
      pos +=1
  except:
    pass
  finally:
    print('*'*40)
    print('RSI Signal Generation completed')
    print('*'*40)
  return df
  

def macd_crossOver(STK_data, fast, slow, signal):
  '''
  :Argument:
    MACD dataframe
  :Return type:
    MACD with Crossover signal
  '''
  stock_data = stock(STK_data)
  df = stock_data.MACD(fast, slow, signal)
  try:
    assert isinstance(df, pd.DataFrame) or isinstance(df, pd.Series)
    #dataframe
    if isinstance(df, pd.Series):
      df = df.to_frame()
    else:
      pass
    #1--> indicates buy position
    #0 --> indicates sell posotion
    df['result'] = np.nan
    df['signal'] = np.where(df.MACD > df.MACD_SIGNAL, 1, 0)
    df['result'] = np.where((df['signal'] == 1) & (df['MACD_HIST'] >= 0), 1, 0)
  except IOError as e:
    raise('Dataframe required {}' .format(e))
  finally:
    print('*'*40)
    print('MACD signal generated')
    print('*'*40)
  return df

def SuperTrend_signal(STK_data, multiplier, period):
  '''
  :Argument:
    MACD dataframe
  :Return type:
    MACD with Crossover signal
  '''
  stock_data = stock(STK_data)
  df = stock_data.SuperTrend(STK_data, multiplier, period)
  try:
    assert isinstance(df, pd.DataFrame) or isinstance(df, pd.Series)
    #dataframe
    if isinstance(df, pd.Series):
      df = df.to_frame()
    else:
      pass
    #1--> indicates buy position
    #0 --> indicates sell posotion
    df = df.fillna(0)
    df['signal'] = np.nan
    df['signal'] = np.where(stock_data.Close >= df.SuperTrend, 1, 0)
  except IOError as e:
    raise('Dataframe required {}' .format(e))
  finally:
    print('*'*40)
    print('SuperTrend Signal generated')
    print('*'*40)
  return df


def bollinger_band_signal(STK_data, period, deviation, strategy = ''):
  '''
  :Argument:
    df:
  :Return type:
    :bollinger band signal
  '''
  stock_data = stock(STK_data)
  Close = stock_data.Close
  df = stock_data.Bolinger_Band(period, deviation)
  df = df.fillna(value = 0)
  assert isinstance(df, pd.DataFrame) or isinstance(df, pd.Series)
  #dataframe
  if isinstance(df, pd.Series):
    df = df.to_frame()
    
  #get signal
  #1--> indicates buy position
  #0 --> indicates sell posotion
  df['signal'] = np.zeros(df.shape[0])
  pos = 0
  try:
    if strategy == ''  or strategy == '0' or strategy == '2':
      for ii in Close:
        print(Close[pos])
        if Close[pos] >= df.Upper_band.values[pos]:
          df['signal'][pos:] = 0
        elif Close[pos] <= df.Lower_band.values[pos]:
          df['signal'][pos:] = 1
        pos += 1
    elif strategy == '1' or strategy == '3':
      for ii in Close:
        print(Close[pos])
        if Close[pos] >= df.Upper_band.values[pos]:
          df['signal'][pos:] = 1
        elif Close[pos] <= df.Lower_band.values[pos]:
          df['signal'][pos:] = 0
        pos += 1
    else:
      raise('You have entered an incorrect strategy value')
  except:
    pass
  finally:
    print('*'*40)
    print('Bollinger Signal Generation completed')
    print('*'*40)
  
  return df


def trading_signal(RSI, MACD, Bollinger_Band, SuperTrend = None, strategy = ''):
  '''
  :Arguments:
    :MACD:
      dataframe containing MACD signal
    :Bollinger_Band:
      dataframe containing Bollinger band signal
    :RSI:
      dataframe containing RSI signal
  :Return Type:
    Buy Sell or Hold signal
  '''
  MACD_signal = MACD.signal.values
  RSI_signal = RSI.signal.values
  BB_signal = Bollinger_Band.signal.values
  if strategy == '' or strategy == '0' or strategy == '1':
    df_prediction = pd.DataFrame({'MACD_signal': MACD_signal,
                                'RSI_signal': RSI_signal,
                                'BB_signal': BB_signal})
  else:
    SuperTrend_Signal = SuperTrend.signal.values
    df_prediction = pd.DataFrame({'MACD_signal': MACD_signal,
                                  'RSI_signal': RSI_signal,
                                  'BB_signal': BB_signal,
                                  'SuperTrend_signal': SuperTrend_Signal})
  df_prediction['POSITION'] = ''
  try:
    if strategy == '' or strategy == '0':
      print('Calling default strategy')
      for ij in range(data.shape[0]):
        print(ij)
        if MACD_signal[ij] == 1 and\
            RSI_signal[ij] == 1 and\
            BB_signal[ij] == 1:
          df_prediction.POSITION[ij] = 'BUY'
        elif  MACD_signal[ij] == 0 and\
              RSI_signal[ij] == 0 and\
              BB_signal[ij] == 0:
          df_prediction.POSITION[ij] = 'SELL'
        else:
          df_prediction.POSITION[ij] = 'HOLD'
    elif strategy == '1':
      print('Calling strategy %s'%strategy)
      for ij in range(data.shape[0]):
        print(ij)
        if MACD_signal[ij] == 1 and\
            RSI_signal[ij] == 1 and\
            BB_signal[ij] == 1:
          df_prediction.POSITION[ij] = 'BUY'
        elif  MACD_signal[ij] == 0 and\
              RSI_signal[ij] == 0 and\
              BB_signal[ij] == 0:
          df_prediction.POSITION[ij] = 'SELL'
        else:
          df_prediction.POSITION[ij] = 'HOLD'
    elif strategy == '2':
      print('Calling strategy %s'%strategy)
      for ij in range(data.shape[0]):
        print(ij)
        if MACD_signal[ij] == 1 and\
            RSI_signal[ij] == 1 and\
            BB_signal[ij] == 1 and\
            SuperTrend_Signal[ij] == 1:
          df_prediction.POSITION[ij] = 'BUY'
        elif MACD_signal[ij] == 0 and\
              RSI_signal[ij] == 0 and\
              BB_signal[ij] == 0 and\
              SuperTrend_Signal[ij] == 0:
          df_prediction.POSITION[ij] = 'SELL'
        else:
          df_prediction.POSITION[ij] = 'HOLD'
    elif strategy == '3':
      print('Calling strategy %s'%strategy)
      for ij in range(data.shape[0]):
        print(ij)
        if MACD_signal[ij] == 1 and\
            RSI_signal[ij] == 1 and\
            BB_signal[ij] == 1 and\
            SuperTrend_Signal[ij] == 1:
          df_prediction.POSITION[ij] = 'BUY'
        elif MACD_signal[ij] == 0 and\
              RSI_signal[ij] == 0 and\
              BB_signal[ij] == 0 and\
              SuperTrend_Signal[ij] == 0:
          df_prediction.POSITION[ij] = 'SELL'
        else:
          df_prediction.POSITION[ij] = 'HOLD'
  except:
    pass
  finally:
    #-----------------------------------------------------------
    #reset column and save to throw to csv     
    if strategy == '' or strategy == '0' or strategy == '1':
      enlist = ['BB_signal', 'MACD_signal' , 'RSI_signal','POSITION']
      df_prediction = df_prediction.reindex(columns=enlist)
    else:
      enlist = ['BB_signal', 'MACD_signal' , 'RSI_signal', 'SuperTrend_signal','POSITION']
      df_prediction = df_prediction.reindex(columns=enlist)
      
    print('*'*40) 
    print('Signal generation completed...')
    print('*'*40)
  
  return df_prediction
    

if __name__ == '__main__':
  '''
  ----------------------------------
  # Trading strategy
  ------------------------------------
  [X][STRATEGY 0 or ''] -->  USES DEFAULT BOLLINGER BAND:: BUY WHEN CLOSE IS BELOW LOWER BOLLINGER 
           SELL WHEN CLOSE IS ABOVE UPPER BOLLINGER BAND
  [X][STRATEGY 1] -->  SETS BOLLINGER TO:: BUY WHEN CLOSE IS ABOVE UPPER BOLLINGER BAND
           AND SELL WHEN CLOSE IS BELOW LOWER BOLLINGER BAND.
  [X][STRATEGY 2] --> USES STRATEGY 0 WITH SUPER TREND INDICATOR
  [X][STRATEGY 3] -->  USES STRATEGY 1 WITH SUPER TREND INDICATOR
  
  '''
  #---------GLOBAL SETTINGS-------------------
  path = 'D:\\BITBUCKET_PROJECTS\\Forecasting 1.0\\'
  STRATEGY = '3'
  DEVIATION = MULTIPLIER = 2
  PERIOD = 20
  DATA_LIMIT = 400
  #--------RSI_SETTINGS------------------------
  LOWER_BOUND = 30
  UPPER_BOUND = 70
  #--------MACD SETTINGS-----------------------
  FAST = 12
  SLOW = 26
  SIGNAL = 9
  loc.set_path(path+'DATASET')
  #-------get the data we need------------------
  STOK_list = ls_STOK()
  
  Signal_Gen = {}
  for ii in range(DATA_LIMIT):
    print('{}'.format(STOK_list[ii]))
    data = loc.read_csv('{}'.format(STOK_list[ii]) + str('.csv'))
    data.index = pd.to_datetime(data.index)
    #-----convert to the stock class--------------
    stock_data = stock(data)
    Fibo_SUP_RES_ = stock_data.fibonacci_pivot_point()
    df_RSI = RSI_signal(data, PERIOD, lw_bound = LOWER_BOUND, up_bound = UPPER_BOUND)
    df_MACD = macd_crossOver(data, FAST, SLOW, SIGNAL)
    df_BB = bollinger_band_signal(data, PERIOD, deviation = DEVIATION, strategy = STRATEGY)
    #-----select strategy for saving-------------------
    if STRATEGY == '2' or STRATEGY == '3':
      df_STrend = SuperTrend_signal(data, MULTIPLIER, PERIOD)
      prediction = trading_signal(df_RSI, df_MACD, df_BB, df_STrend, STRATEGY)
      prediction.set_index(data.index, inplace = True)
      prediction = pd.concat([Fibo_SUP_RES_, prediction], axis = 1)
      Signal_Gen['{}'.format(STOK_list[ii])] = prediction.iloc[-1]
    else:
      prediction = trading_signal(df_RSI, df_MACD, df_BB, STRATEGY)
      prediction.set_index(data.index, inplace = True)
      prediction = pd.concat([Fibo_SUP_RES_, prediction], axis = 1)
      Signal_Gen['{}'.format(STOK_list[ii])] = prediction.iloc[-1]
      
  #-------------------Styling--------
  color_col = {'Support 1': 'lightgreen', 'Support 2': 'palegreen', 'Support 3': 'limegreen',
            'Resistance 1': 'salmon', 'Resistance 2': 'tomato', 'Resistance 3': 'orangered'}
  
  def col_map(m, color_col):
      if m.name in color_col.keys():
          return ['background-color: {}'.format(color_col[m.name])] * len(m)
      elif m.name == 'POSITION':
          return ['background-color: skyblue']* len(m)
      else:
          return [''] * len(m)
  #---------------------------------------
  Signal_Gen = pd.DataFrame.from_dict(Signal_Gen).T
  Signal_Gen.to_csv('SIGNAL_GEN_ALL.csv', mode = 'w')
  Signal_Gen.style.apply(col_map, color_col = color_col)
  loc.set_path(path+ 'PREDICTED')
  Signal_Gen.to_excel('SIGNAL_GEN_ALL.xlsx')
  
  
    













