# -*- coding: utf-8 -*-
"""
Created on Thu Nov 09 12:15:57 2017

@author: Sonal
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 14:29:27 2017

@author: Sonal
"""

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from hmmlearn import HMM_Model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
import scipy.io as sio
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

import os
import math


path = 'C:\Users\Sonal\Documents\Third Sem\AGGM\Proj 1\Proj1 - Training Data\Trainning Data'

def read_file(path):
	columns = ['Body_temp', 'Heart_rate', 'Env_temp', 'Humidity', 'X_Mean', 'Y_Mean', 'Z_Mean', 'VarX', 'VarY', 'VarZ', 'CoVarX', 'CoVarY', 'CoVarZ', 'Skew_X', 'Skew_Y', 'Skew_Z', 'Kurt_X', 'Kurt_Y', 'Kurt_Z', 'PSM_X', 'PSM_Y', 'PSM_Z', 'Freq_X', 'Freq_Y', 'Freq_Z', 'A_X_Mean', 'A_Y_Mean', 'A_Z_Mean', 'A_VarX', 'A_VarY', 'A_VarZ', 'A_CoVarX', 'A_CoVarY', 'A_CoVarZ', 'A_Skew_X', 'A_Skew_Y', 'A_Skew_Z', 'A_Kurt_X', 'A_Kurt_Y', 'A_Kurt_Z', 'A_PSM_X', 'A_PSM_Y', 'A_PSM_Z', 'A_Freq_X', 'A_Freq_Y', 'A_Freq_Z', 'Mean_val', 'LinearTrend', 'Variance', 'Skewness', 'Kurtosis', 'Ran_min_max']
	data = pd.DataFrame()
	target = pd.DataFrame()
	for root, dirs, files in os.walk(path):
		for file in files:
			with open(os.path.join(root, file), "r") as auto:
				mat = sio.loadmat(root + '\\'+file)
				matx = pd.DataFrame(mat['x'])
				maty = pd.DataFrame(mat['y'])
				data = pd.concat([data,pd.DataFrame(mat['x'])], axis = 0)
				target = pd.concat([target, pd.DataFrame(mat['y'])], axis = 0)

	data.columns = columns
	data.fillna(0, inplace = True)

	target.columns = ['Target']

	return data, target

def preproc_basic(X, Y):
    X = X.reset_index()
    Y = Y.reset_index()
    
    min_max_scaler = preprocessing.MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(X)
    #Y_scaled = min_max_scaler.fit_transform(Y) 
    X_normalized = pd.DataFrame(X_scaled)
    return X_normalized, Y


def featsel_correlation(data):
    corr =  data.corr(method = 'pearson')
    plt.matshow(corr)
    plt.show()
    l = list(corr)
    for i in l:
        for j in l:
            if corr[i][j] > 0.9 and i != j and j in data:
                data = data.drop(j,1)

	#print data.shape # Shape is (113901, 41)
    return data

def featsel_pca(data):
	num_features = len(list(data))
	pca = PCA()
	data_pca = pca.fit_transform(data)
	#print "Data PCA - 41"
	#print data_pca
	x = np.arange(num_features) + 1
	y = np.std(data_pca, axis=0)**2
	plt.plot(x, y, "o-")
	plt.xticks(x, [str(i) for i in x], rotation=60)
	plt.ylabel("Variance")
	plt.show()

	data_pca_4 = PCA(n_components = 12).fit_transform(data)
	return data_pca_4

def featsel_randomforests(data, target):
	regr = RandomForestRegressor()
	regr.fit(data, target)
	importances = regr.feature_importances_
	#print importances
	regr_array = [linear_model.LinearRegression(), linear_model.Lasso(), linear_model.Ridge()]


	for regr in regr_array:
		preproc_rf_data = errorplots_randomforests(data, target, regr, importances)

	return importances

def errorplots_randomforests(X,Y,regr,importances):
	columns = list(X)

	sorted_columns = [x for _,x in sorted(zip(importances,columns), reverse=True)]
	print 'Sorted Columns', sorted_columns
	error_array = []
	for i in range(1,len(sorted_columns)):
		err = regr_model(X[sorted_columns[:i]], Y['Target'], regr)
		error_array.append(err)
	print error_array

	x = range(len(list(X))-1)
	y = error_array
	plt.plot(x, y, "o-")
	plt.ylabel('Error')
	plt.xlabel("First 'n' features")
	plt.show()


def regr_model_cv(X, Y, regr):
	loo = LeaveOneOut()
	error = []

	for train_index, test_index in loo.split(X):
		model = regr.fit(X.loc[train_index],Y.loc[train_index])
		predictions = model.predict(X.loc[test_index])
		actuals = Y.loc[test_index]
		error.append(mean_squared_error(actuals, predictions))

	print regr
	return sum(error) / float(len(error))
	#return error

def regr_model(X, Y, regr):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    model = regr.fit(X_train,Y_train)
    predictions = model.predict(X_test)
    actuals = Y_test
    error = mean_squared_error(actuals, predictions)
    r2_error = r2_score(actuals, predictions)
	#print regr
    return error, r2_error
	#return error


def model_linearreg(X, Y):
    print list(Y)
    regr = linear_model.LinearRegression()
    #error = regr_model_cv(X, Y, regr)
    error, r2_error = regr_model(X, Y['Target'], regr)
    print "Linear Regression --> ", error, r2_error

    #return error

def model_lassoreg(X, Y):
	regr = linear_model.Lasso(alpha=0.5)
	#error = regr_model_cv(X, Y, regr)
	error, r2_error = regr_model(X, Y['Target'], regr)
	print "Lasso Regression --> ", error, r2_error

	#return error

def model_ridgereg(X, Y):
	regr = linear_model.Ridge(alpha=0.1)
	#error = regr_model_cv(X, Y, regr)
	error, r2_error = regr_model(X, Y['Target'], regr)
	print "Ridge Regression --> ", error, r2_error

'''
def model_xgboostreg(X, Y):
    regr = xgb.XGBRegressor()
    #error = regr_model_cv(X, Y, regr)
    #print Y
    error, r2_error = regr_model(X, Y['Target'], regr)
    print "XGBoost Regression --> ", math.sqrt(error), r2_error

	#return error
'''
def model_xgboostreg(X, Y):
    print 'XGBoost'
    min_error = float('inf')
    '''
    for lr in np.arange(0,0.1,0.01):
        for md in range(3,10):
            for mcw in range(1,6):
                regr = xgb.XGBRegressor(learning_rate = lr, max_depth = md, min_child_weight = mcw)
                error, r2_error = regr_model(X, Y['Target'], regr)
                print 'Learning rate = ', lr, 'Max Depth = ', md, 'Min Child Weight = ', mcw, ' Error = ', math.sqrt(error), 'R2 = ', r2_error
                if error < min_error:
                    learning_rate = lr
                    max_depth = md
                    min_child_weight = mcw
                    min_error = error
	'''
    learning_rate = 0.03
    max_depth = 9
    min_child_weight = 1
    
    regr = xgb.XGBRegressor(learning_rate = learning_rate, max_depth = max_depth, min_child_weight = min_child_weight)
    #error = regr_model_cv(X, Y['Target'], regr)
    #print Y
    error, r2_error = regr_model(X, Y['Target'], regr)
    print "XGBoost Regression --> ", error, r2_error

	#return error



def model_nnreg(X, Y):
    regr = MLPRegressor(learning_rate='adaptive',solver='adam',activation='tanh')
    error, r2_error = regr_model(X, Y['Target'], regr)
    print "NN Regression --> ", error, r2_error


data, target = read_file(path)
data = data.reset_index()
target = target.reset_index()

data, target = preproc_basic(data, target)

preproc_correlation_data = featsel_correlation(data)
#preproc_rf_data = featsel_randomforests(data, target)
preproc_pca_data = featsel_pca(preproc_correlation_data)

print 'Data', data.shape
print 'Preproc - corr data', preproc_correlation_data.shape
print 'Preproc - pca data', preproc_pca_data.shape

print "Correlation Data"

#model_linearreg(preproc_correlation_data, target)
#model_lassoreg(preproc_correlation_data, target)
#model_ridgereg(preproc_correlation_data, target)

model_xgboostreg(preproc_correlation_data, target)
#model_nnreg(preproc_correlation_data, target)

print "PCA Data"

#model_linearreg(preproc_pca_data, target)
#model_lassoreg(preproc_pca_data, target)
#model_ridgereg(preproc_pca_data, target)

model_xgboostreg(preproc_pca_data, target)
#model_nnreg(preproc_pca_data, target)

print "Random Forests - Data"
#preproc_importance_rf_data 
#featsel_randomforests(data, target)

print "Random Forests - Corr Data"
#preproc_importance_rf_corrdata
#featsel_randomforests(preproc_correlation_data, target)
