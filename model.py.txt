import pandas as pd
import numpy as np
import matplotlib as plt
import datetime as dt
from sklearn import * 
% matplotlib inline

y_data = pd.read_csv(r'C:\Users\anhem44\Desktop\Capstone 2\train_2016_v2.csv',index_col='parcelid')
train_data = pd.read_csv(r'C:\Users\anhem44\Desktop\Capstone 2\zillow_final.csv',index_col='parcelid')
final_df = pd.read_csv(r'C:\Users\anhem44\Desktop\Capstone 2\zillow_train.csv',index_col='parcelid')
	
train_data['transactiondate'] = pd.to_datetime(train_data['transactiondate']) #make transdate a datetime
train_data['year'] = train_data['transactiondate'].dt.year #extract year
train_data['month']  = train_data['transactiondate'].dt.month #extract month

train_data['is_spring']  = 0
train_data['is_spring'][(train_data['transactiondate'].dt.month >= 3) & (train_data['transactiondate'].dt.month <= 5)] = 1
    
train_data['is_fall']  = 0
train_data['is_fall'][(train_data['transactiondate'].dt.month >= 9) & (train_data['transactiondate'].dt.month <= 11)] = 1
                            
train_data['is_winter']  = 0
train_data['is_winter'][(train_data['transactiondate'].dt.month >= 12) | (train_data['transactiondate'].dt.month <= 2)] = 1
                          
train_data['is_summer']  = 0
train_data['is_summer'][(train_data['transactiondate'].dt.month >= 6) & (train_data['transactiondate'].dt.month <= 8)] = 1
                          
train_data['is_spring_fall']  = 0
train_data['is_spring_fall'][(train_data['is_spring'] == 1) | (train_data['is_fall'] == 1)] = 1

train_data['parcelid'] = train_data.index #make a coulumn out of the index
train_data.set_index(['parcelid','transactiondate'], inplace=True) #reset index

from sklearn import *

target = train_data['logerror'] #pull y from training data
final_train = train_data.drop(train_data[['logerror']],axis=1) 

final_df['year'] = 2016

#6 data sets for six predictions
final_201610 = final_df.copy()
final_201611 = final_df.copy()
final_201612 = final_df.copy()

final_201710 = final_df.copy()
final_201711 = final_df.copy()
final_201712 = final_df.copy()

#month and year for every period
final_201610['month'] =10
final_201610['is_spring_fall'] = 1
final_201610['is_spring'] = 0
final_201610['is_fall'] = 1
final_201610['is_summer'] = 0
final_201610['is_winter'] = 0

final_201611['month'] =11
final_201611['is_spring_fall'] = 1
final_201611['is_spring'] = 0
final_201611['is_fall'] = 1
final_201611['is_summer'] = 0
final_201611['is_winter'] = 0

final_201612['month'] =12
final_201612['is_spring_fall'] = 0
final_201612['is_spring'] = 0
final_201612['is_fall'] = 0
final_201612['is_summer'] = 0
final_201612['is_winter'] = 1


final_201710['month'] =10
final_201710['is_spring_fall'] = 1
final_201710['is_spring'] = 0
final_201710['is_fall'] = 1
final_201710['is_summer'] = 0
final_201710['is_winter'] = 0

final_201711['month'] =11
final_201711['is_spring_fall'] = 1
final_201711['is_spring'] = 0
final_201711['is_fall'] = 1
final_201711['is_summer'] = 0
final_201711['is_winter'] = 0

final_201712['month'] =12
final_201712['is_spring_fall'] = 0
final_201712['is_spring'] = 0
final_201712['is_fall'] = 0
final_201712['is_summer'] = 0
final_201712['is_winter'] = 1

final_201710['year'] = 2017
final_201711['year'] = 2017
final_201712['year'] = 2017

#best model

RF = ensemble.RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features=2, max_leaf_nodes=None, min_impurity_split=1e-07,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=150, n_jobs=1,
           oob_score=False, random_state=None, verbose=0, warm_start=False)

RF.fit(final_train,target)

#six predictions
pred_201610 = RF.predict(final_201610)
pred_201611 = RF.predict(final_201611)
pred_201612 = RF.predict(final_201612)

pred_201710 = RF.predict(final_201710)
pred_201711 = RF.predict(final_201711)
pred_201712 = RF.predict(final_201712)

#load sample submission
sub = pd.read_csv(r'C:\Users\anhem44\Desktop\Capstone 2\sample_submission.csv',index_col='ParcelId')

#populate sample with predictions
sub['201610'] = pred_201610
sub['201611'] = pred_201611
sub['201612'] = pred_201612
sub['201710'] = pred_201710
sub['201711'] = pred_201711
sub['201712'] = pred_201712
sub = sub.round(4)
sub.to_csv(r'C:\Users\anhem44\Desktop\Capstone 2\zillow_submission.csv') 
