# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 11:50:05 2018

@author: kennedy
"""

__author__ = "kennedy Czar"
__email__ = "kennedyczar@gmail.com"
__version__ = '1.0'

import os
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
                
                Creates the DATASSET FOLDER @D:\\MyDirectory\\DATASET
                
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
                               '\\TICKERS' , 
                               '\\PREDICTED', 
                               '\\SAMPLE1', 
                               '\\SAMLE2', 
                               '\\SAMPLE2']
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
           
    def STOCK_EXTRACTOR(self, url):
        '''
        :Functionality:
            Collects stock data using the yahoo API
            Collects all excel data and stores in DATASET FOLDER
            append .csv to all files downloaded
        '''
        import fix_yahoo_finance as yahoo
        import pandas as pd
        from datetime import datetime
        
        self.url = url
        '''Set the start date'''
        self.START = '2010-01-01'
#        self.END = '2018-10-01'
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
            STOCK_TICKER_.to_csv('TICKERS/STOCK_TICKER{}'.format(extension))
        print('======= Begin downloading stock dataset ======')
        try:
            for self.TICK_SYMBOLS in self.TICKERS:
                '''just in case your connection breaks, 
                we'd like to save our progress! by appending
                downloaded dataset to DATASET FOLDER'''
                if not os.path.exists('DATASET/{}.csv'.format(self.TICK_SYMBOLS)):
                  try:
                    df = yahoo.download(self.TICK_SYMBOLS, start = self.START, end = datetime.now())
                    df.reset_index(inplace = True)
                    df.set_index("Date", inplace = True)
                    df.to_csv('DATASET/{}.csv'.format(self.TICK_SYMBOLS))
                  except ValueError:
                    pass
                else:
                    print('File Already existing: {}'.format(self.TICK_SYMBOLS))
        except OSError as e:
            raise OSError('Something wrong with destination path: {}'.format(e))
        finally:
            print('Download Completed..Exiting!')
        
      
if __name__ == '__main__':
    '''Define a path on your drive where this project folder is located'''
    path = 'D:\\GIT PROJECT\\ERIC_PROJECT101\\FREELANCE_KENNETH'
    base_url = "https://www.interactivebrokers.com/en/index.php?f=2222&exch=mexi&showcategories=STK&p=&cc=&limit=100"
    Data_collector(path).STOCK_EXTRACTOR(base_url)
    
    


















