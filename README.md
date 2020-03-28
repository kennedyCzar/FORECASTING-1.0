# FORECASTING 1.0 [![HitCount](http://hits.dwyl.io/kennedyCzar/https://github.com/kennedyCzar/FORECASTING-1.0.svg)](http://hits.dwyl.io/kennedyCzar/https://github.com/kennedyCzar/FORECASTING-1.0) ![](https://img.shields.io/badge/python-v3.6-orange.svg)
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

Forecasting 1.0 is an intelligent model for time-series prediction and forecasting. Implemented using ensemble models, Boosting Model and Recurrent Neural Network.

The General idea is to be able to predict important features available from stock data.

The algorithm is also capable of forecasting future market prices by filtering out bank holidays and weekend before digesting the data.

This project is similar to the Facebook prophet except we havent compared the result that much to see which is better for forecasting. Our algo is infact capable of forecasting accurately the weekly prices. beyond that, the accuracy of forecasting reduces. Note that it also serves as a signal generator.


## HOW TO USE


```bash
 git clone https://github.com/kennedyCzar/FORECASTING-1.0
 ```
 [Data collector](https://github.com/kennedyCzar/FORECASTING-1.0/blob/master/SCRIPTS/Data_Collector.py)
 <ul>
  <li>Open the SCRIPT FOLDER and run  | Use the command <small>python Data_Collector.py</small></li>
  <li>Note that this has been tailored to first download ticks from xm before downloading EOD data using yahoo API</small></li>
</ul>

[STOCK class](https://github.com/kennedyCzar/FORECASTING-1.0/blob/master/SCRIPTS/STOCK.py)
 <ul>
  <li>This class contains the technical indicators used required for generating signals and forecasting</li>
</ul>

[Forecast OHLC of Nextdays](https://github.com/kennedyCzar/FORECASTING-1.0/blob/master/SCRIPTS/OHLC_OHLC.py)
 <ul>
  <li>Still in the SCRIPT folder you will find OHLC_OHLC.py. Specify the days forward you will like to forecast on line 422  | Use the command <small>python OHLC_OHLC.py</small></li>
</ul>

[Forecast OHLC of Nextdays](https://github.com/kennedyCzar/FORECASTING-1.0/blob/master/SCRIPTS/OHLC_Specific.py)
 <ul>
  <li>This script responsible for forecasting OHLC for specific stocks. Specidy the days forward you will like to forecast on line 420 | Use the command <small>python OHLC_Specific.py</small></li>
  <li> Update the name of the stocks you want to forecast on line 468</li>
</ul>

[Forecast Open, High, Low or Close of Nextdays](https://github.com/kennedyCzar/FORECASTING-1.0/blob/master/SCRIPTS/OHLC_Single.py)
 <ul>
  <li> OHLC_Single.py is Specifically for this purpose. the days forward you will like to forecast on line 425  | Use the command <small>python OHLC_Single.py</small></li>
  <li>This script is only used to predict either of the four features(OHLC) at a time</li>
</ul>

[Forecast OHLC of all STOCKS in the dataset](https://github.com/kennedyCzar/FORECASTING-1.0/blob/master/SCRIPTS/OHLC_Forecaster_All.py)
 <ul>
  <li>This script is responsible for forecasting OHLC for all stocks in the DATASET FOLDER  | </li>
  <li>Use the command <small>python OHLC_Forecaster_All.py</small></li>
  <li>Note that this a not the mult-threaded version, so it would take some considerable abount of time.</li>
</ul>

[Signal Generator](https://github.com/kennedyCzar/FORECASTING-1.0/blob/master/SCRIPTS/Signal_generator.py)
 <ul>
  <li>This script to generate signals for specific stock  | </li>
  <li>Use the command <small>python Signal_generator.py</small></li>
  <li>Edit name of stock on line 326</li>
  <li>The generated signals is saved in the PREDICTED FOLDER</li>
</ul>

[Signal Generator Forecaster](https://github.com/kennedyCzar/FORECASTING-1.0/blob/master/SCRIPTS/Signal_Gen_Forecaster.py)
 <ul>
  <li>Script to generate signals and forecast future signals | </li>
  <li>Use the command <small>python Signal_Gen_Forecaster.py</small></li>
  <li>Edit name of stock on line 735</li>
  <li>The generated signals is saved in the PREDICTED FOLDER</li>
</ul>


## OUTPUT IMAGES

![Image of AC](https://github.com/kennedyCzar/FORECASTING-1.0/blob/master/IMAGES/AC.png)
![Image of Axtel](https://github.com/kennedyCzar/FORECASTING-1.0/blob/master/IMAGES/AXTEL1.png)
![Image of img](https://github.com/kennedyCzar/FORECASTING-1.0/blob/master/IMAGES/IMG.png)
![Image of img2](https://github.com/kennedyCzar/FORECASTING-1.0/blob/master/IMAGES/IMG2.png)
![Image of Axtel](https://github.com/kennedyCzar/FORECASTING-1.0/blob/master/IMAGES/CEM1.png)
