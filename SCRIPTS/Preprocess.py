# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 18:49:14 2018

@author: kennedy
"""
import pandas as pd
import numpy as np

def process_time(df):
  if 'timestamp' not in df.columns:
    df.index = pd.to_datetime(df.index)
    df['seconds'] = df.index.second
    df['minutes'] = df.index.minute 
    df['hours'] = df.index.hour
    df['min'] = df['hours'].map(str)+":"+df['minutes'].map(str)
    df['years'] = df.index.year
    df['days'] = df.index.day
    df['months'] = df.index.month
    df['DayOfTheWeek'] = df.index.dayofweek
    df['WeekDay'] = (df['DayOfTheWeek'] < 5).astype(int)
    df['daylight'] = ((df['hours'] >= 8) & (df['hours'] <= 20)).astype(int)
    df['mnth'] = df['months'].map(str)+":"+df['years'].map(str)
    df['time_epoch'] = (df.index.astype(np.int64)/100000000000).astype(np.int64)
    
    # Cyclical variable transformations
    # wday has period of 5
    df['wday_sin'] = np.sin(2 * np.pi * df['DayOfTheWeek'] / len(df.DayOfTheWeek.unique()))
    df['wday_cos'] = np.cos(2 * np.pi * df['DayOfTheWeek'] / len(df.DayOfTheWeek.unique()))
    
    #days has period 31 or 30 or 28
    df['mday_sin'] = np.sin(2 * np.pi * df['days'] / df.days.max())
    df['mday_cos'] = np.cos(2 * np.pi * df['days'] / df.days.max())
      
    # yday has period of 365
    df['yday_sin'] = np.sin(2 * np.pi * df['DayOfTheWeek'] / 365)
    df['yday_cos'] = np.cos(2 * np.pi * df['DayOfTheWeek'] / 365)
    
    # month has period of 12
    df['month_sin'] = np.sin(2 * np.pi * df['months'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['months'] / 12)
    
    # time has period of 24
    df['time_sin'] = np.sin(2 * np.pi * df['hours'] / 24)
    df['time_cos'] = np.cos(2 * np.pi * df['hours'] / 24)
  else:
    df['seconds'] = df['timestamp'].dt.second
    df['minutes'] = df['timestamp'].dt.minute
    df['hours'] = df['timestamp'].dt.hour
    df['min'] = df['hours'].map(str)+":"+df['minutes'].map(str)
    df['years'] = df['timestamp'].dt.year
    df['days'] = df['timestamp'].dt.day
    df['months'] = df['timestamp'].dt.month
    df['DayOfTheWeek'] = df['timestamp'].dt.dayofweek
    df['WeekDay'] = (df['DayOfTheWeek'] < 5).astype(int)
    df['daylight'] = ((df['hours'] >= 8) & (df['hours'] <= 20)).astype(int)
    df['mnth'] = df['months'].map(str)+":"+df['years'].map(str)
    df['time_epoch'] = (df['timestamp'].astype(np.int64)/100000000000).astype(np.int64)
    
    # Cyclical variable transformations
    # wday has period of 5
    df['wday_sin'] = np.sin(2 * np.pi * df['DayOfTheWeek'] / len(df.DayOfTheWeek.unique()))
    df['wday_cos'] = np.cos(2 * np.pi * df['DayOfTheWeek'] / len(df.DayOfTheWeek.unique()))
    
    #days has period 31 or 30 or 28
    df['mday_sin'] = np.sin(2 * np.pi * df['days'] / df.days.max())
    df['mday_cos'] = np.cos(2 * np.pi * df['days'] / df.days.max())
        
    # yday has period of 365
    df['yday_sin'] = np.sin(2 * np.pi * df['DayOfTheWeek'] / 365)
    df['yday_cos'] = np.cos(2 * np.pi * df['DayOfTheWeek'] / 365)
    
    # month has period of 12
    df['month_sin'] = np.sin(2 * np.pi * df['months'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['months'] / 12)
    
    # time has period of 24
    df['time_sin'] = np.sin(2 * np.pi * df['hours'] / 24)
    df['time_cos'] = np.cos(2 * np.pi * df['hours'] / 24)
  
  return df
  
  
