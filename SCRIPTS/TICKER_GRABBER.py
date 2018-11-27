# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 20:32:38 2018

@author: kennedy
"""

__author__ = "kennedy Czar"
__email__ = "kennedyczar@outlook.com"
__version__ = '1.0'


import pandas as pd
from bs4 import BeautifulSoup
import requests 

base_url = "https://www.interactivebrokers.com/en/index.php?f=2222&exch=mexi&showcategories=STK&p=&cc=&limit=100"
n = 1

url_list = []

while n <= 35:
	url = (base_url + "&page=%d" % n)
	url_list.append(url)
	n = n+1
    
#%% CODE- E
    
def parse_websites(url_list):
  df = pd.DataFrame(columns = range(4), index = [0])
  for url in url_list:
    html_string = requests.get(url)
    soup = BeautifulSoup(html_string.text, 'lxml') # Parse the HTML as a string
    table = soup.find('div',{'class':'table-responsive no-margin'}) # Grab the first table

    for row_marker, row in enumerate(table.findAll('tr')):
      columns = row.findAll('td')
      try:
        df.loc[row_marker-1] = [column.get_text() for column in columns]
      except ValueError:
        continue
    
    print('========================={}====================='.format(row_marker))
    print(df)
    df.to_csv('D:\\GIT PROJECT\\ERIC_PROJECT101\\FREELANCE_KENNETH\\SCRAPPING\\test1.csv', mode='a', header = False)


parse_websites(url_list)

#%%CODE K

def parse(url_list):
  IB_Symbol = []
  Prod_Descr = []
  Symbol = []
  Currency = []
  for url in url_list:
    html_string = requests.get(url)
    soup = BeautifulSoup(html_string.text, 'lxml')
    table = soup.find('div',{'class':'table-responsive no-margin'})
    for row in table.findAll('tr')[1:]:
      print(row)
      #grabs columns
      ticker = row.findAll('td')[0].text
      Pdescr = row.findAll('td')[1].text
      Sym = row.findAll('td')[2].text
      Curr = row.findAll('td')[3].text
      
      '''append to empty list for each column
      This would have been good in a for loop
      but that would only increase the time 
      complexity of our script, making it slow.
      '''
      IB_Symbol.append(ticker)
      Prod_Descr.append(Pdescr)
      Symbol.append(Sym)
      Currency.append(Curr)
    
  #append the resultant list in a pandas dataframe
  df = pd.DataFrame({'IB_Symbol':IB_Symbol, 
                     'Prod_Descr': Prod_Descr,
                     'Symbol': Symbol,
                     'Currency': Currency})
      
  return df 


data = parse(url_list)

