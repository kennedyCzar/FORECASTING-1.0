# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 22:12:27 2018

@author: kennedy
"""

__author__ = "kennedy Czar"
__email__ = "kennedyczar@gmail.com"
__version__ = '1.0'

import pandas as pd
from bs4 import BeautifulSoup
import requests 



    
class TICKER(object):
  '''
  :Arguments:
    url: IB url containing symbol names and
        description
        
  :Return:
    dataframe of table as seen on IB website
    
  '''
  def __init__(self, url):
    self.url = url
    
  def parse(self):
    self.url_list = []
    self.IB_Symbol = []
    self.Prod_Descr = []
    self.Symbol = []
    self.Currency = []
    
    #catch the page list
    for ii in range(1, 36):
      self.url = (self.url + "&page=%d" % ii)
      self.url_list.append(self.url)
    
    for self.url in self.url_list:
      html_string = requests.get(self.url)
      soup = BeautifulSoup(html_string.text, 'lxml')
      table = soup.find('div',{'class':'table-responsive no-margin'})
      for row in table.findAll('tr')[1:]:
        print(row)
        #grabs columns
        '''append to empty list for each column
        This would have been good in a for loop
        but that would only increase the time 
        complexity of our script, making it slow.
        '''
        ticker = row.findAll('td')[0].text
        Pdescr = row.findAll('td')[1].text
        Sym = row.findAll('td')[2].text
        Curr = row.findAll('td')[3].text
        
        self.IB_Symbol.append(ticker)
        self.Prod_Descr.append(Pdescr)
        self.Symbol.append(Sym)
        self.Currency.append(Curr)
      
    #append the resultant list in a pandas dataframe
    df = pd.DataFrame({'IB_Symbol':self.IB_Symbol, 
                       'Prod_Descr': self.Prod_Descr,
                       'Symbol': self.Symbol,
                       'Currency': self.Currency})
        
    return df 


