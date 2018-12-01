# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 14:57:09 2018

@author: kennedy
"""
#import class numpy
import numpy as np
#import pandas class
import pandas as pd

class loc:
  '''
  Class Location:
    Contains elementary functions for 
    getting path and data
    
  '''
  #Call set path to fetch data
  def set_path(path = ''):
      '''
      :Arguments:
        path: set path to workig directory
        
      :Return:
        path to anydirectory of choice
        
        ex. of path input
        
          D:\\GIT PROJECT\\ERIC_PROJECT101\\
      '''
      import os
      if os.path.exists(path):
        if not os.chdir(path):
          os.chdir(path)
        else:
          os.chdir(path)
      else:
        raise OSError('path not existing::'+
                      'Ensure path is properly refrenced')

  def read_csv(csv):
    '''
    :Arguments:
      path: path to csv stock dataset
      
    :Return:
      returns the csv data
    '''
    
    data = pd.read_csv(csv)
    data = data.set_index('Date')
    return data
  
##-------------------------------------------------------
#%%
      
class stock(object):
  '''
  Stock properties
  
  :Return:
    
  '''

  
  #init constructor
  def __init__(self, data = None):
    '''
    :Argument:
      data input
      
      :Return:
        
    '''
    self.data = data
  
  
  '''
  :Class properties:
    These are properties of the stock class.
    they can be called using self.property_name
    
    ex.
    volume = self.Volume
    instead of self.data.Volume
  
  '''
  @property
  def Close(self):
    '''
    :Return:
      Closing price of stock
    '''
    return self.data.Close
  
  
  @property
  def Open(self):
    '''
    :Return:
      Opening price of stock
    '''
    return self.data.Open
  
  
  @property
  def Volume(self):
    '''
    :Return:
      Volume of stock
    '''
    return self.data.Volume
  
  
  @property
  def High(self):
    '''
    :Return:
      High price of stock
    '''
    return self.data.High
  
  
  @property
  def Low(self):
    '''
    :Return:
      Low price of stock
    '''
    return self.data.Low
  
  
  @property
  def c(self):
    '''
    :Return:
      Closing price of stock
    '''
    return self.data.Close
  
  @property
  def l(self):
    '''
    :Return:
      Low price of stock
    '''
    return self.data.Low
  
  
  @property
  def h(self):
    '''
    :Return:
      High price of stock
    '''
    return self.data.High
  
  
  @property
  def Adj_close(self):
    '''
    :Return:
      Adjusted closing price of stock
    '''
    return self.data['Adj Close']
  
  
  @property
  def adj(self):
    '''
    :Return:
      Adjusted closing price of stock
    '''
    return self.data['Adj Close']
  
  
  @property
  def v(self):
    '''
    :Return:
      Volume of stock
    '''
    return self.data.Volume
  
  
  @property
  def vol(self):
    '''
    :Return:
      Closing price of stock
    '''
    return self.data.Volume
  
  @property
  def o(self):
    '''
    :Return:
      Opening stock price
    '''
    return self.data.Open
  
  '''
  Working functions
  '''
  
  
  def hl_spread(self):
    '''
    :Return:
      Spread of the stock
    '''
    
    return self.High - self.Low
  
  
  def average_price(self):
    '''
    :Return:
      Average price of stpck
    '''
    
    return (self.Close + self.High + self.Low)/3
  
  def true_range(self):
    '''
    Returns:
      the true range
    '''
    return self.High - self.Low.shift(1)
  
  '''
  :Price function:: Utils
  '''
  def sma(self, df, n):
    '''
    Arguments:
      df: dataframe or column vector
      n: interval
    :Return:
      simple moving average
    '''
    self.df = df
    self.n = n
    
    return self.df.rolling(self.n).mean()
  
  def ema(self, df, n):
    '''
    Arguments:
      df: dataframe or column vector
      n: interval
    :Return:
      simple moving average
    '''
    self.df = df
    self.n = n
    return self.df.ewm(self.n).mean()
  
  def returns(self, df):
    '''
    :Arguments:
      df: x or dataframe vector
      
    :Return:
      Stock returns
    '''
    self.df = df
    return (self.df/ self.df.shift(1) - 1)
  
  def log_returns(self, df):
    '''
    :Arguments:
      df: input feature vector
      
    :Returns:
      log returns
    '''
    self.df = df
    
    return np.log(self.df / self.df.shift(1))
    
  def cm_annual_growth(self, df):
    '''
    :Argument:
      df: dataframe
    
    ::Return:
      Compound annual growth
    '''
    self.df = df
    self.DAYS_IN_YEAR = 365.35
    start = df.index[0]
    end = df.index[-1]
    
    return np.power((df.ix[-1] / df.ix[0]), 1.0 / ((end - start).days / self.DAYS_IN_YEAR)) - 1.0
  
  def quadrant(self):
    '''
    :Return:
      Quandrants: [0], [1], [2], [3], [4]
    '''
    
    #divide the price by 4
    quater_price = self.hl_spread()/4
    #get the lowest price for the day
    bottom_line = self.Low
    #get the first line
    first_line = bottom_line + quater_price
    #the middle line
    middle_line = quater_price + first_line
    #the third quadrant
    third_line = quater_price + middle_line
    #the fourth line or price high
    #It can also be third_line + quadrant_price
    top_line = self.High
    
    return pd.DataFrame({'Low': bottom_line, 'first_quad': first_line, 
                         'middle_quad': middle_line, 'third_quad': third_line, 
                         'High': top_line})
    
    
  def fibonacci_pivot_point(self):
    '''
    :Returns:
      Fibonaccci Pivot point
      
      S1, S2, S3: Support from 1--> 3
      R1, R2, R3: Resistance from 1-->3
      
      0.382, 0.618, 1 :--> Fibonacci Retracement Numbers
    '''
    
    #average price
    avg_price = self.average_price()
    #high low spread or high low price difference
    high_low_spread = self.hl_spread()
    #support 1
    S1 = avg_price - (0.382 * high_low_spread)
    #support 2
    S2 = avg_price - (0.618 * high_low_spread)
    #support 3
    S3 = avg_price - (1 * high_low_spread)
    #Resistance 1
    R1 = avg_price + (0.382 * high_low_spread)
    #Resistance 2
    R2 = avg_price + (0.618 * high_low_spread)
    #Resistance 3
    R3 = avg_price + (1 * high_low_spread)
    
    return pd.DataFrame({'Support 1': S1, 'Support 2': S2, 'Support 3': S3,
                         'Resistance 1': R1, 'Resistance 2': R2, 'Resistance 3': R3})
    
    
  def money_flow(self):
    '''
    :Return:
      Money Flow
    '''
    
    return ((self.Close - self.Low) - (self.High - self.Close)) / (self.High - self.Low)
  
  
  def money_flow_volume(self):
    '''
    :Return:
      Money flow Volume
    '''
    return self.money_flow() * self.Volume

    
  def Money_flow_Index(self, n = None):
    '''
    :Argument:
      N: period
    :Return:
      Money flow index
    '''
    self.n = n
    if self.n == None:
      raise OSError('missing n value:: Add a period value n')
    else:
      #get the average/typical price
      typical_price = self.average_price()
      #Raw money flow
      raw_money_flow = typical_price * self.Volume
      #money flow ratio
      Money_flow_ratio = (raw_money_flow.shift(self.n))/(raw_money_flow.shift(-self.n))
      #Money flow index
      Money_flow_index = 100 - 100/(1 + Money_flow_ratio)
      
    return pd.DataFrame({'Money flow index': Money_flow_index})
  
  
  def OHLC(self):
    '''
    :Returns :
      OHLC --> Open, High, Low, Close
      
    '''
    return pd.DataFrame({'Open': self.Open, 'High': self.High,
                         'Low': self.Low, 'Close': self.Close})
  
  def Bolinger_Band(self, price, dev):
    '''
    :Argument:
      Price: average price to calculate bolinger band
      Dev: deviation factor from the moving average
    
    :Return:
      Upper, MAe and Lower price band.
      MA: Moving Average
      Upper: MA + std(Closing_price)
      Lower: MA - std(Closing_price)
      
    '''
    self.price = price
    self.dev = dev
    MA = self.Close.rolling(self.price).mean()
    Upper_band = MA + 2 * self.Close.rolling(price).std()
    Lower_band = MA - 2 * self.Close.rolling(price).std()
    
    return pd.DataFrame({'Moving Average': MA,
                         'Upper_band': Upper_band,
                         'Lower_band': Lower_band})
  

  
  
  
  
  
  
  
  
  
  
  