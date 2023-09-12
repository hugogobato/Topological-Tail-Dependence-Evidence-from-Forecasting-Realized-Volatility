# -*- coding: utf-8 -*-
"""RUT_HARST_(PH).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1u2UCUm5DIG0GW1QL1MoxZbTCPPoN34BD
"""

import pandas as pd
import numpy as np
import math
from scipy.optimize import fmin, minimize

df = pd.read_excel("RUT RV.xlsx",index_col='Date',parse_dates=True)
df.drop(columns=["VIX"], inplace=True)
train = df.iloc[:int(len(df['RV'])*0.8)]
test= df.iloc[int(len(df['RV'])*0.8)-22:]

train.head()

train=np.array(train)
RVs=train[:,1]
Zt=train[:,0]

test=np.array(test)
RVs_test=test[:,1]
Zt_test=test[:,0]

np.percentile(Zt,75)

data_test=np.append(RVs_test, Zt_test)

data=np.append(RVs, Zt)

def HARST_estimation(data, phi):
  # RVs = 1xT vector with Realized Volatility Values
  # Zt = 1xT vector with chosen transition variable
  # phi = 1x12 vector with model parameters
  RVs=data[:int(len(data)/2)]
  Zt=data[int(len(data)/2):]

  RV_d = []
  for i in range(len(RVs)-1):
    RV_d.append(RVs[i])

  Zt_d = []
  for i in range(len(RVs)-1):
    Zt_d.append(Zt[i])

  RV_w = []
  for i in range(len(RVs)-5):
    RV_w.append(np.mean(RVs[i:i+5]))

  RV_m = []
  for i in range(len(RVs)-22):
    RV_m.append(np.mean(RVs[i:i+22]))

  RV_d=RV_d[len(RV_d)-len(RV_m):]
  Zt_d=Zt_d[len(Zt_d)-len(RV_m):]
  RV_w=RV_w[len(RV_w)-len(RV_m):]

  RV_actuals=RVs[22:]

  Data ={'Actual': RV_actuals[:],
  'D': RV_d[:],
  'W': RV_w[:],
  'M': RV_m[:],
         'Zt': Zt_d[:]}
  dataframe = pd.DataFrame(data=Data)
  rv=dataframe
  Threshold=0.0424632562116633


  RV_hat=np.zeros(len(rv["Actual"]))
  for i in range(len(RV_hat)):
    if rv["Zt"].iloc[i]>=Threshold:
      RV_hat[i]=(phi[0]+phi[1]*rv["D"].iloc[i]+phi[2]*rv["W"].iloc[i]+phi[3]*rv["M"].iloc[i]+
              phi[4]*rv["D"].iloc[i]*(1/(1+np.exp(-phi[7]*(rv["Zt"].iloc[i]-phi[8]))))+phi[5]*rv["W"].iloc[i]*(1/(1+np.exp(-phi[7]*(rv["Zt"].iloc[i]-phi[8]))))
              +phi[6]*rv["M"].iloc[i]*(1/(1+np.exp(-phi[7]*(rv["Zt"].iloc[i]-phi[8])))))
    else:
      RV_hat[i]=(phi[0]+phi[1]*rv["D"].iloc[i]+phi[2]*rv["W"].iloc[i]+phi[3]*rv["M"].iloc[i]+
              phi[9]*rv["D"].iloc[i]*(1/(1+np.exp(-phi[11]*(rv["Zt"].iloc[i]-phi[12]))))+phi[10]*rv["W"].iloc[i]*(1/(1+np.exp(-phi[11]*(rv["Zt"].iloc[i]-phi[12]))))
              +phi[11]*rv["M"].iloc[i]*(1/(1+np.exp(-phi[11]*(rv["Zt"].iloc[i]-phi[12])))))

  return RV_hat

def QMLE(phi, data):
  RV_hat=HARST_estimation(data, phi)
  RVs=data[:int(len(data)/2)]
  RV_actuals=RVs[22:]
  QMLE=np.sum((RV_actuals-RV_hat)**2)/(len(RV_actuals))

  return QMLE

phi=(np.ones(13)*0.3)

phi

opt = {'maxiter':2000, 'maxfev':20e2}
opt_out = minimize(QMLE, phi, args = data,
                   method='Nelder-Mead',options=opt)

opt_out

opt = {'maxiter':500, 'maxfev':5e2}
opt_out = minimize(QMLE, phi, args = data,
                   method='Nelder-Mead',options=opt)

opt_out.x

Y_hat=HARST_estimation(data_test, opt_out.x)

Y_hat.shape

RVs_test=RVs_test[22:]

print(f"RMSE: {round(np.sqrt(np.sum((Y_hat-RVs_test)**2)/len(Y_hat))*100,3)} %")
print(f"MAE: {round((np.sum(abs(Y_hat-RVs_test))/len(Y_hat))*100,3)} %")
print(f"QLIKE: {round((np.sum((RVs_test/Y_hat)-np.log(RVs_test/Y_hat)-1)/len(Y_hat))*100,2)} %")

test= df.iloc[int(len(df['RV'])*0.8):]

test["RV"]=Y_hat

test

test.to_excel("RUT - HARST - PH.xlsx")

