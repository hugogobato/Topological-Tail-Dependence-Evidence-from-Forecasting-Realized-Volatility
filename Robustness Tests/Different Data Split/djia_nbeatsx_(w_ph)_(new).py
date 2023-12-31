# -*- coding: utf-8 -*-
"""DJIA_NBEATSx_(w_PH) (new).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1OO4iqCJ4rhpZawTOIcrlc4gQ2ay8rPhd
"""

!pip install neuralforecast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATSx
from neuralforecast.losses.pytorch import MQLoss, DistributionLoss, MSE, MAE
from neuralforecast.tsdataset import TimeSeriesDataset, TimeSeriesLoader
from numpy.random import seed
from random import randrange
from neuralforecast.losses.numpy import rmse, mape
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from neuralforecast.utils import AirPassengers, AirPassengersPanel, AirPassengersStatic

df = pd.read_excel("DJIA RV.xlsx",index_col='Date',parse_dates=True)

df.reset_index(inplace=True)
df.rename(columns={'Date':'ds'}, inplace=True)
df.rename(columns={'RV':'y'}, inplace=True)
df.drop(columns=["VIX"], inplace=True)
df["unique_id"]="airplane1"
train = df.iloc[:int(len(df['y'])*0.8)]
test = df.iloc[int(len(df['y'])*0.8):]
validation_length=int(len(df['y'])*0.8)-int(len(df['y'])*0.6)

df.head()

n_inputs = [5,10,21,63,84,126,252]
mlp_units = [[[712, 712], [712, 712]],[[512, 512], [512, 512]],[[250, 250], [250, 250]],[[100, 100], [100, 100]]]
epochs=[25,50,100,150,250,350,450,550,750]
learning_rate=[0.0005,0.0001,0.00005,0.00001]
num_lr_decays=[5,3,2,1]
dropouts=[0,0.2,0.3,0.4,0.5]
scaler_type=["robust","standard",'minmax']
stack_types=[['identity','identity'],['identity','identity'],['identity','identity'],['identity','identity']]
n_harmonics=[0,0,1,1]
n_blocks=[[1, 1],[2, 2],[3, 3],[5, 5]]
n_polynomials=[0,1,0,1]
losses=[MSE(),MAE(),MQLoss(level=[90]),MQLoss(level=[80, 90]),MQLoss(level=[95]), MQLoss(level=[75])]
RMSE = []
Accuracy = []

"""# H=1 Forecasts"""

RMSE=[]
Quasilikelihood=[]

test_length=len(df["y"])-int(len(df['y'])*0.6)

model = NBEATSx(h=1, input_size=5,
                loss=losses[1],
                scaler_type='standard',
                learning_rate= 0.00001,
                stack_types=['identity','identity'],
                n_blocks=[5,5],
                mlp_units= [[712, 712], [712, 712]],
                windows_batch_size=60,
                num_lr_decays=2,
                val_check_steps=500,
                n_harmonics=0, n_polynomials=0,
                hist_exog_list=['wasserstein_dists_2D'],
                max_steps=150,
                early_stop_patience_steps=1,
                random_seed=62404816
                )
fcst = NeuralForecast(models=[model],freq=CustomBusinessDay(calendar=USFederalHolidayCalendar()))
forecasts = fcst.cross_validation(df=df,val_size=5,static_df=AirPassengersStatic,n_windows=None, test_size=test_length-test_length%1,step_size=1)
forecasts = forecasts.dropna()
if "NBEATSx-median" not in list(forecasts.columns.values):
  Y_hat1=forecasts["NBEATSx"].values
else:
  Y_hat1=forecasts["NBEATSx-median"].values
Y_true=forecasts["y"].values
RMSE.append(np.sqrt(np.sum(((Y_true-Y_hat1)**2))/len(Y_true)))
Quasilikelihood.append(np.sum(Y_true/Y_hat1-np.log(Y_true/Y_hat1)-1)/len(Y_true))

print(f"RMSE: {round(np.mean(RMSE)*100,4)}%")
print(f"QLIKE: {round(np.mean(Quasilikelihood)*100,2)}%")

print(f"MAE: {round(np.sum(abs(Y_true-Y_hat1))/len(Y_true)*100,4)}%")

plt.plot(forecasts["ds"], Y_true)
plt.plot(forecasts["ds"], Y_hat1)
plt.ylabel("DJIA Realized Volatility")
plt.xticks(rotation=50)
plt.show()

Data = {'Date': forecasts["ds"],
        'Actuals': Y_true,
        'Forecast with PH': Y_hat1
        }
df1=pd.DataFrame(data=Data)
df1.to_excel("Forecast with PH (NBEATSx).xlsx")

