# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 11:50:05 2018

@author: kennedy
"""

__author__ = "kennedy Czar"
__email__ = "kennedyczar@gmail.com"
__version__ = '1.0'

import os
import time
from datetime import datetime
from os import chdir
from selenium import webdriver
from TICKERS import TICKER

class Data_collector(object):
  def __init__(self, path):
    '''
    :Argument:
        :path:
            Enter the woring directory os state
            the location you would love to create the dataset
            
        :example:
            create_folder('D:\\YourDirectory')
            
            Creates the DATASET FOLDER @D:\\MyDirectory\\DATASET
            
    :Complexity for file Creation:
        The Function runs in:
            Time Complexity: O(N*logN)
            Space Complexity: O(1)
    '''
    self.path = path
    try:
      if os.path.exists(self.path):
        try:
          self.FOLDERS = ['\\DATASET', 
                     '\\TICKERS', 
                     '\\PREDICTED', 
                     '\\MODEL']
          FOLDER_COUNT = 0
          for folders in self.FOLDERS:
            '''If folder is not created or created but deleted..Recreate/Create the folder.
            Check for all folders in the FOLDERS list'''
            if not os.path.exists(self.path + self.FOLDERS[FOLDER_COUNT]):
              os.makedirs(path + self.FOLDERS[FOLDER_COUNT])
              print('====== 100% Completed ==== : {}'.format(self.path + self.FOLDERS[FOLDER_COUNT]))
              FOLDER_COUNT += 1
            elif os.path.exists(self.path + self.FOLDERS[FOLDER_COUNT]):
              '''OR check if the file is already existing using a boolean..if true return'''
              print('File Already Existing : {}'.format(self.path + self.FOLDERS[FOLDER_COUNT]))
              FOLDER_COUNT += 1
        except OSError as e:
            '''raise OSError('File Already Existing {}'.format(e))'''
            print('File Already existing: {}'.format(e))
      elif not os.path.exists(self.path):
          raise OSError('File path: {} does not exist\n\t\tPlease check the path again'.format(self.path))
      else:
          print('File Already Existing')
    except Exception as e:
        raise(e)
    finally:
        print('Process completed...Exiting')
         
  def STOCK_EXTRACTOR(self, url, start, end):
    '''
    :Functionality:
        Collects stock data using the yahoo API
        Collects all excel data and stores in DATASET FOLDER
        append .csv to all files downloaded
        self.END = datetime.today().date()
    '''
    import fix_yahoo_finance as yahoo
    import pandas as pd
    from datetime import datetime
    
    self.url = url
    '''Set the start date'''
    self.START = start 
    self.END = end
    
#        start_date = pd.Timestamp(2010, 12, 29)
#        end_date = 
    '''Create a list of stocks to download'''
    self.TICKERS = []
    self.SYMBOLS = TICKER(self.url).parse()
    
    for ii in self.SYMBOLS['IB_Symbol'].values:
      self.TICKERS.append(ii + '{}'.format('.MX'))
      
    '''write the stock data to specific format by 
    appending the right extension'''
    STOCK_TICKER_ = pd.DataFrame(self.TICKERS)
    self.FORMAT = ['.csv', '.xlsx', '.json']
    for extension in self.FORMAT:
        STOCK_TICKER_.to_csv('../TICKERS/STOCK_TICKER{}'.format(extension))
    print('======= Begin downloading stock dataset ======')
    try:
      self.unavailable_ticks = []
      for self.TICK_SYMBOLS in self.TICKERS:
          '''just in case your connection breaks, 
          we'd like to save our progress! by appending
          downloaded dataset to DATASET FOLDER'''
          if not os.path.exists('../DATASET/{}.csv'.format(self.TICK_SYMBOLS)):
            try:
              df = yahoo.download(self.TICK_SYMBOLS, start = self.START, end = self.END)
              df.reset_index(inplace = True)
              df.set_index("Date", inplace = True)
              #check size of file before saving
              import sys
              if sys.getsizeof(df) <= 1024:
                pass
              else:
                df.to_csv('../DATASET/{}.csv'.format(self.TICK_SYMBOLS))
            except ValueError:
              print('{} is unavailable'.format(self.TICK_SYMBOLS))
              self.unavailable_ticks.append(self.TICK_SYMBOLS)
              pass
          else:
            #this section redownloads the file even though it
            #is already existing..
            try:
              df = yahoo.download(self.TICK_SYMBOLS, start = self.START, end = datetime.now())
              df.reset_index(inplace = True)
              df.set_index("Date", inplace = True)
              #check size of file before saving
              import sys
              if sys.getsizeof(df) <= 1024:
                pass
              else:
                df.to_csv('../DATASET/{}.csv'.format(self.TICK_SYMBOLS))
            except ValueError:
              print('{} is unavaible'.format(self.TICK_SYMBOLS))
              self.unavailable_ticks.append(self.TICK_SYMBOLS)
              pass
#                    print('File Already existing: {}'.format(self.TICK_SYMBOLS))
    except OSError as e:
        raise OSError('Something wrong with destination path: {}'.format(e))
    finally:
        print('API Download Completed..!')
        print('*'*40)
        print('External Download Begin..!')
        print('Unavailable tickers are \n{}\n'.format(self.unavailable_ticks))
        print('*'*60)
        print('Downloading Unavailable data from YAHOO..')
        print('*'*40)
        print('External Download Completed..!')
        print('*'*40)
        print('Process Completed..!')
        print('A total of {} unavailable tickers'.format(len(self.unavailable_ticks)))
  

def YAHOO_(path, start, end, stock_):
  import pandas as pd
  import numpy as np
  date = [start, end]
  date_epoch = pd.DatetimeIndex(date)
  date_epoch = date_epoch.astype(np.int64) // 10**9
  
  chrom_options_ = webdriver.ChromeOptions()
  
  prefer_ = {'download.default_directory': path,
             'profile.default_content_settings.popups': 0,
             'directory_upgrade': True}
  
  chrom_options_.add_experimental_option('prefs',prefer_)
  
  
  for ii in stock_:
    try:
      yahoo_page_ = 'https://finance.yahoo.com/quote/{}/history?period1={}&period2={}&interval=1d&filter=history&frequency=1d'.format(ii, date_epoch[0], date_epoch[1])
      driver = webdriver.Chrome("C:/chromedriver.exe", chrome_options = chrom_options_)
      driver.minimize_window()
      driver.get(yahoo_page_)
      time.sleep(2)
      driver.find_element_by_css_selector('.btn.primary').click()
      time.sleep(2)
      driver.find_element_by_css_selector('.Fl\(end\).Mt\(3px\).Cur\(p\)').click()
      time.sleep(10)
      driver.close()
    except:
      pass
  
if __name__ == '__main__':
    #specify the start and end dates for stocks
    import pandas as pd
    start = pd.Timestamp(2010, 1, 1)
    end = datetime.today().date()
    '''Define a path on your drive where this project folder is located'''
    path = 'D:\\BITBUCKET_PROJECTS\\Forecasting 1.0'
    base_url = "https://www.interactivebrokers.com/en/index.php?f=2222&exch=mexi&showcategories=STK&p=&cc=&limit=100"
    Data_collector(path).STOCK_EXTRACTOR(base_url, start, end)
    











