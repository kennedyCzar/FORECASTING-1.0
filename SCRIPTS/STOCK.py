# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 14:57:09 2018

@author: kennedy
"""
#import class numpy
import numpy as np
#import pandas class
import pandas as pd
pd.options.mode.chained_assignment = None

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
        raise OSError('path not existing::'+\
                      'Ensure path is properly referenced')

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
  
  def average_true_range(self, df, n):
        """
        
        :param df: pandas.DataFrame
        :param n: data window
        :return: pandas.DataFrame
        """
        i = 0
        TR_l = [0]
        while i < df.index[-1]:
          TR = max(df.loc[i + 1, 'High'], df.loc[i, 'Close']) - min(df.loc[i + 1, 'Low'], df.loc[i, 'Close'])
          TR_l.append(TR)
          i = i + 1
        TR_s = pd.Series(TR_l)
        ATR = pd.Series(TR_s.ewm(span=n, min_periods=n).mean(), name='ATR_' + str(n))
        return ATR
  
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
  
  def std(self, df, n):
    '''
    Arguments:
      df: dataframe or column vector
      n: interval
    :Return:
      standard deviation of a price
    '''
    self.df = df
    self.n = n
    
    return self.df.rolling(self.n).std()
  
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
  
  def HL_PCT(self):
    '''
    :Return:
      HL_PCT
      PCT_CHNG
    '''
    return pd.DataFrame({'HL_PCT':(self.High - self.Low)/(self.Low*100),
                         'PCT_CHNG': (self.Close - self.Open)/(self.Open*100)})
  
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
      
    ex.
    stock_class.Bollinger_Band(20,2)
    '''
    self.price = price
    self.dev = dev
    #sma
    MA = self.sma(self.Close, self.price)
    #standard deviation
    SDEV = self.std(self.Close, self.price)
    SDEV = self.Close.rolling(self.price).std()
    Upper_band = MA + (SDEV * self.dev)
    Lower_band = MA - (SDEV * self.dev)
    
    return pd.DataFrame({'bollinger_band': MA,
                         'Upper_band': Upper_band,
                         'Lower_band': Lower_band})
  
  
  def MACD(self, n_fast, n_slow, signal):
    '''
    :Arguments:
      :n_fast: <integer> representing fast exponential
              moving average
              
      :n_slow: <integer> representing slow exponential
              moving average
              
      :signal: Signal line
      
    :Return:
      MACD: fast, slow and signal.
    '''
    
    self.n_fast = n_fast
    self.n_slow = n_slow
    self.signal = signal
    #defin MACD
    macd = self.ema(self.Close, n_fast) - self.ema(self.Close, n_slow)
    #MACD signal
    macd_signal = self.ema(macd, self.signal)
    #MACD histo
    macd_histo_ = macd - macd_signal
    return pd.DataFrame({'MACD': macd, 'MACD_HIST': macd_histo_,
                         'MACD_SIGNAL': macd_signal})
  
  def WilderRSI(self, df, n):
    """
    Calculate Relative Strength Index(RSI) for given data.
      :param df: pandas.DataFrame
      :param n: data period
    :Return: pandas.DataFrame
    """
    i = 0
    UpI = [0]
    DoI = [0]
    while i + 1 <= len(df.index) - 1:
      UpMove = df.ix[i + 1, 'High'] - df.ix[i, 'High']
      DoMove = df.ix[i, 'Low'] - df.ix[i + 1, 'Low']
      if UpMove > DoMove and UpMove > 0:
        UpD = UpMove
      else:
        UpD = 0
      UpI.append(UpD)
      if DoMove > UpMove and DoMove > 0:
        DoD = DoMove
      else:
        DoD = 0
      DoI.append(DoD)
      i = i + 1
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(self.ema(UpI, n))
    NegDI = pd.Series(self.ema(DoI, n))
    RSI = pd.Series(PosDI / (PosDI + NegDI), name='WilderRSI_' + str(n))
    return RSI*100
  
  def CutlerRSI(self, df, n):
    """
    Calculate Relative Strength Index(RSI) for given data.
      :param df: pandas.DataFrame
      :param n: data period
    :Return: pandas.DataFrame
    """
    i = 0
    UpI = [0]
    DoI = [0]
    while i + 1 <= len(df.index) - 1:
      UpMove = df.ix[i + 1, 'High'] - df.ix[i, 'High']
      DoMove = df.ix[i, 'Low'] - df.ix[i + 1, 'Low']
      if UpMove > DoMove and UpMove > 0:
        UpD = UpMove
      else:
        UpD = 0
      UpI.append(UpD)
      if DoMove > UpMove and DoMove > 0:
        DoD = DoMove
      else:
        DoD = 0
      DoI.append(DoD)
      i = i + 1
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(self.sma(UpI, n))
    NegDI = pd.Series(self.sma(DoI, n))
    RSI = pd.Series(PosDI / (PosDI + NegDI), name='RSI_Cutler_{}'.format(n))
    return RSI*100
  
  def ATR(self, df, n):
    '''
    :Argument:
      df:
        dataframe
      n: period
      
    :Return:
      Average True Range
    '''
    df = df.copy(deep = True)
    df['High_Low'] = abs(self.High - self.Low)
    df['High_PrevClose'] = abs(self.High - self.Close.shift(1))
    df['Low_PrevClose'] = abs(self.Low - self.Close.shift(1))
    df['True_Range'] = df[['High_Low', 'High_PrevClose', 'Low_PrevClose']].max(axis = 1)
    df = df.fillna(0)
    df['ATR']=np.nan
    df['ATR']= self.ema(df['True_Range'], n)
    return df['ATR']
  
  def SuperTrend(self, df, multiplier, n):
    '''
    :Arguments:
      df:
        dataframe
      :ATR:
        Average True range
      :multiplier:
        factor to multiply with ATR for upper and lower band
      :n:
        period
    
    :Return type:
      Supertrend
    '''
    df = df.copy(deep = True)
    ATR = self.ATR(df, n)
    df['Upper_band_start'] = (self.High + self.Low)/2 + (multiplier * ATR)
    df['Lower_band_start'] = (self.High + self.Low)/2 - (multiplier * ATR)
    df = df.fillna(0)
    df['SuperTrend'] = np.nan
    #Upper_band
    df['Upper_band']=df['Upper_band_start']
    df['Lower_band']=df['Lower_band_start']
    #Upper_band
    for ii in range(n,df.shape[0]):
        if df['Close'][ii-1]<=df['Upper_band'][ii-1]:
            df['Upper_band'][ii]=min(df['Upper_band_start'][ii], df['Upper_band'][ii-1])
        else:
            df['Upper_band'][ii]=df['Upper_band_start'][ii] 
            
    #Lower_band
    for ij in range(n,df.shape[0]):
      if df['Close'][ij-1] >= df['Lower_band'][ij-1]:
        df['Lower_band'][ij]=max(df['Lower_band_start'][ij], df['Lower_band'][ij-1])
      else:
        df['Lower_band'][ij]=df['Lower_band_start'][ij] 
        
    #SuperTrend 
    for ik in range(1, len(df['SuperTrend'])):
      if df['Close'][n - 1] <= df['Upper_band'][n - 1]:
        df['SuperTrend'][n - 1] = df['Upper_band'][n - 1]
      elif df['Close'][n - 1] > df['Upper_band'][ik]:
        df = df.fillna(0)
        df['SuperTrend'][n - 1] = df['Lower_band'][n - 1]
    for sp in range(n,df.shape[0]):
      if df['SuperTrend'][sp - 1] == df['Upper_band'][sp - 1] and\
          df['Close'][sp]<=df['Upper_band'][sp]:
        df['SuperTrend'][sp]=df['Upper_band'][sp]
      elif  df['SuperTrend'][sp - 1] == df['Upper_band'][sp - 1] and\
            df['Close'][sp]>=df['Upper_band'][sp]:
        df['SuperTrend'][sp]=df['Lower_band'][sp]
      elif df['SuperTrend'][sp - 1] == df['Lower_band'][sp - 1] and\
            df['Close'][sp]>=df['Lower_band'][sp]:
        df['SuperTrend'][sp]=df['Lower_band'][sp]
      elif df['SuperTrend'][sp - 1] == df['Lower_band'][sp - 1] and\
            df['Close'][sp] <= df['Lower_band'][sp]:
        df['SuperTrend'][sp] = df['Upper_band'][sp]
    #return supertrend only    
    return df['SuperTrend']

  def Keltner_channel(self, df, period, atr_period, multiplier):
    '''
    :Arguments:
      :period:
      :atr_period:

    :Return type:
      :keltner channel
    '''
    ATR = self.ATR(df, atr_period)
    Mid_band = self.ema(self.Close, period)
    Lower_band = Mid_band + multiplier * ATR.values
    Upper_band = Mid_band - multiplier * ATR.values
    return pd.DataFrame({'ul': Upper_band, 'ml': Mid_band, 'll': Lower_band})
      
      
    
        
        
    
    

  
  
  
  
  
  
  
  
  