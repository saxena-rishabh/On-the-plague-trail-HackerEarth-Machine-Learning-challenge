# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 14:25:44 2019

@author: Rishabh
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
pd.set_option('display.max_columns', None) # to display the total number columns present in the dataset

data = pd.read_csv('train.csv')

pd.set_option('float_format', '{:f}'.format)

import datetime as dt
data['DateTime'] = pd.to_datetime(data['DateTime'])
data['DateTime']=data['DateTime'].map(dt.datetime.toordinal)

data.drop(['ID', 'WindDir', 'HiDir'], axis=1, inplace=True)

X = data.iloc[:, :-7].values
y1 = data.iloc[:, [-1] ].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size = 0.25, random_state = 0)


################## XGBOOST ####################################

import xgboost as xgb


xg_reg2 = xgb.XGBRegressor(booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.8, gamma=0.5, learning_rate=0.1,
       max_delta_step=0, max_depth=5, min_child_weight=5, missing=None,
       n_estimators=250, n_jobs=1, nthread=-1, objective='reg:linear',
       random_state=0, reg_alpha=10, reg_lambda=1, scale_pos_weight=1,
       seed=None, silent=True, subsample=0.8)

xg_reg2.fit(X_train,y_train)

y_pred = xg_reg2.predict(X_test)


from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_test, y_pred))

############### TEST DATASET #############################################

data2 = pd.read_csv('test.csv')

data2['DateTime'] = pd.to_datetime(data2['DateTime'])
data2['DateTime']=data2['DateTime'].map(dt.datetime.toordinal)

idlist = data2["ID"].tolist()

data2.drop(['ID', 'WindDir', 'HiDir'], axis=1, inplace=True)


X2 = data2.iloc[:, :].values

# PREDICTING

y_pred_testset_PA = xg_reg2.predict(X2)

y_pred_testset_PB = xg_reg2.predict(X2)

y_pred_testset_PC = xg_reg2.predict(X2)

y_pred_testset_PD = xg_reg2.predict(X2)

y_pred_testset_PE = xg_reg2.predict(X2)

y_pred_testset_PF = xg_reg2.predict(X2)

y_pred_testset_PG = xg_reg2.predict(X2)

#converting to list
y_pred_testset_PA_list = list(y_pred_testset_PA)

y_pred_testset_PB_list = list(y_pred_testset_PB)

y_pred_testset_PC_list = list(y_pred_testset_PC)

y_pred_testset_PD_list = list(y_pred_testset_PD)

y_pred_testset_PE_list = list(y_pred_testset_PE)

y_pred_testset_PF_list = list(y_pred_testset_PF)

y_pred_testset_PG_list = list(y_pred_testset_PG)

#creating CSV file
df = pd.DataFrame(data={"ID":idlist , "PA": y_pred_testset_PA_list,
                        "PB":y_pred_testset_PB_list, "PC":y_pred_testset_PC_list,
                        "PD":y_pred_testset_PD_list, "PE":y_pred_testset_PE_list,
                        "PF":y_pred_testset_PF_list, "PG":y_pred_testset_PG_list})
df.to_csv("submission1.csv", sep=',',index=False)


