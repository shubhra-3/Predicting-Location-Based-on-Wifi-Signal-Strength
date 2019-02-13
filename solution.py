# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 00:02:23 2018

@author: saumya
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("train.csv")
testset = pd.read_csv('test.csv')#, names=colnames, header=None)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

X = dataset.iloc[:, :-4].values
Y = dataset.iloc[:, 14:18].values
test = testset.iloc[:, 1:].values

X[:, 0:10] = np.log2(X[:, 0:10]) 
Y[:, 0:10] = np.log2(Y[:, 0:5]) 
test[:, 0:10] = np.log2(test[:, 0:10]) 

from sklearn.cross_validation import train_test_split
X_train ,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0, random_state=0)

"""import statsmodels.formula.api as sm
X = np.append(arr = np.ones((200000 ,1)).astype(int), values = X , axis = 1)


#for first result
X_opt0 = X[:, [0 ,1 ,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
regressor_OLS = sm.OLS( endog = Y[:,0] ,exog = X_opt0).fit()
regressor_OLS.summary()

X_opt0 = X[:, [0 ,1 ,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14]]
regressor_OLS = sm.OLS( endog = Y[:,0] ,exog = X_opt0).fit()
regressor_OLS.summary()

#for second result
X_opt1 = X[:, [0 ,1 ,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
regressor_OLS = sm.OLS( endog = Y[:,1] ,exog = X_opt1).fit()
regressor_OLS.summary()

X_opt1 = X[:, [0 ,1 ,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14]]
regressor_OLS = sm.OLS( endog = Y[:,1] ,exog = X_opt1).fit()
regressor_OLS.summary()

#for third result
X_opt2 = X[:, [0 ,1 ,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
regressor_OLS = sm.OLS( endog = Y[:,2] ,exog = X_opt2).fit()
regressor_OLS.summary()

X_opt2 = X[:, [0 ,1 ,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14]]
regressor_OLS = sm.OLS( endog = Y[:,2] ,exog = X_opt2).fit()
regressor_OLS.summary()

#for fourth result
X_opt3 = X[:, [0 ,1 ,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
regressor_OLS = sm.OLS( endog = Y[:,3] ,exog = X_opt3).fit()
regressor_OLS.summary()

X_opt3 = X[:, [0 ,1 ,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14]]
regressor_OLS = sm.OLS( endog = Y[:,3] ,exog = X_opt3).fit()
regressor_OLS.summary()

X_train = np.append(arr = np.ones((133333 ,1)).astype(int), values = X_train , axis = 1)
X_test = np.append(arr = np.ones((66667 ,1)).astype(int), values = X_test , axis = 1)

Y_train[:, 0:5] = np.log2(Y_train[:, 0:5]) 
X_trainopt = X_train[:, [ 0 ,1 ,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14]]
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_trainopt , Y_train[:,0])

X_testopt = X_test[:, [0 ,1 ,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14]]
Y_pred = regressor.predict(X_testopt)
Y_pred=np.exp2(Y_pred[:,])
"""

"""from sklearn.ensemble import RandomForestClassifier
regressorRF = RandomForestClassifier(n_estimators = 100,random_state = 0)
regressorRF.fit(X_train , Y_train[:,0])

Y_predDT = regressorRF.predict(X_test)
rmsRF = sqrt(mean_squared_error(Y_test, Y_predRF))


from sklearn.metrics import mean_squared_error
from math import sqrt
Y_pred[:, 0:1] = np.exp(Y_pred[:, 0:1]) 
rms = sqrt(mean_squared_error(Y_test[:,0], Y_pred))
"""
test = np.append(arr = np.ones((41601 ,1)).astype(int), values = test , axis = 1)
testopt = test[:, [ 0 ,1 ,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14]]

from sklearn.neighbors import KNeighborsRegressor
for i in range(10) :
    i=i+1
    regressorKNN = KNeighborsRegressor(n_neighbors = i, metric = 'minkowski', p = 2)
    regressorKNN.fit(X_trainopt, Y_train[:,0])

    Y_predKNN = regressorKNN.predict(X_testopt)
    print  ('rmse_score ', sqrt(mean_squared_error(Y_test[:,0], Y_predKNN)))



pred = regressor.predict(X_testopt)


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, Y_train)
pred = regressor.predict(X_test)
#pred = np.exp2(pred)
rms = sqrt(mean_squared_error(Y_test, pred))

pred = regressor.predict(test)
solution = pd.DataFrame({'Run1 (ms)':pred[:,0], 'Run2 (ms)' : pred[:,1],'Run3 (ms)':pred[:,2],'Run4 (ms)':pred[:,3], 'Id':testset['Id']})
solution.to_csv('sampleSubmission.csv',index = False, sep=',',  header=True, columns=["Id","Run1 (ms)","Run2 (ms)","Run3 (ms)","Run4 (ms)"])


from sklearn.ensemble import RandomForestRegressor
regressorRF = RandomForestRegressor(n_estimators = 100,random_state = 0)
regressorRF.fit(X_train , Y_train)

Y_predRF = regressorRF.predict(X_test)
rmsRF = sqrt(mean_squared_error(Y_test, Y_predRF))

pred = regressorRF.predict(test)
solution = pd.DataFrame({'Run1 (ms)':pred[:,0], 'Run2 (ms)' : pred[:,1],'Run3 (ms)':pred[:,2],'Run4 (ms)':pred[:,3], 'Id':testset['Id']})
solution.to_csv('sampleSubmission.csv',index = False, sep=',',  header=True, columns=["Id","Run1 (ms)","Run2 (ms)","Run3 (ms)","Run4 (ms)"])

"""