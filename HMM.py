# -*- coding: utf-8 -*-
"""
Created on Thurs Nov 9 14:29:27 2017

@author: Sonal
"""

from sklearn.decomposition import PCA
from sklearn import linear_model
from pomegranate import *
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, train_test_split
import pandas as pd
import scipy.io as sio
from hmmlearn import hmm
from autohmm import ar
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import os
import math
from pylab import hist

path = 'Trainning Data'

def read_file(path):
	columns = ['Body_temp', 'Heart_rate', 'Env_temp', 'Humidity', 'X_Mean', 'Y_Mean', 'Z_Mean', 'VarX', 'VarY', 'VarZ', 'CoVarX', 'CoVarY', 'CoVarZ', 'Skew_X', 'Skew_Y', 'Skew_Z', 'Kurt_X', 'Kurt_Y', 'Kurt_Z', 'PSM_X', 'PSM_Y', 'PSM_Z', 'Freq_X', 'Freq_Y', 'Freq_Z', 'A_X_Mean', 'A_Y_Mean', 'A_Z_Mean', 'A_VarX', 'A_VarY', 'A_VarZ', 'A_CoVarX', 'A_CoVarY', 'A_CoVarZ', 'A_Skew_X', 'A_Skew_Y', 'A_Skew_Z', 'A_Kurt_X', 'A_Kurt_Y', 'A_Kurt_Z', 'A_PSM_X', 'A_PSM_Y', 'A_PSM_Z', 'A_Freq_X', 'A_Freq_Y', 'A_Freq_Z', 'Mean_val', 'LinearTrend', 'Variance', 'Skewness', 'Kurtosis', 'Ran_min_max']
	data = pd.DataFrame()
	target = pd.DataFrame()
	for root, dirs, files in os.walk(path):
		for file in files:
			with open(os.path.join(root, file), "r") as auto:
				mat = sio.loadmat(root + '/'+file)
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
    X_normalized.to_csv('output.csv')
    return X_normalized, Y


def featsel_correlation(data):
    corr =  data.corr(method = 'pearson')
    plt.matshow(corr)
    #plt.show()
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
	#plt.show()

	data_pca_4 = PCA(n_components = 10).fit_transform(data)
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



def hmmlearn(data, target):
	X_train, X_test, Y_train, Y_test = train_test_split(data, target['Target'], test_size=0.33, random_state=42)
	model = hmm.GaussianHMM(n_components=100, covariance_type="full", n_iter=100)
	model.fit(X_train)
	pred = model.predict(X_test)
	print min(pred), max(pred), len(pred)



def pomegranate(data, target):
	X_train, X_test, Y_train, Y_test = train_test_split(data, target['Target'], test_size=0.33, random_state=42)
	#model = HiddenMarkovModel.from_samples(NormalDistribution, n_components=5, X=X_train.as_matrix())
	'''
	s1 = State(Distribution(NormalDistribution(5, 1)))
	s2 = State(Distribution(NormalDistribution(1, 7)))
	s3 = State(Distribution(NormalDistribution(8, 2)))
	s4 = State(Distribution(NormalDistribution(3, 3)))
	s5 = State(Distribution(NormalDistribution(2, 4)))
	s6 = State(Distribution(NormalDistribution(4, 5)))
	s7 = State(Distribution(NormalDistribution(6, 6)))
	s8 = State(Distribution(NormalDistribution(7, 7)))
	model = HiddenMarkovModel()
	model.add_states(s1, s2, s3, s4, s5, s6, s7, s8)
	model.add_transition(model.start, s1, 1.0)
	model.add_transition(s1, s1, 0.7)
	model.add_transition(s1, s2, 0.3)
	model.add_transition(s2, s2, 0.8)
	model.add_transition(s2, s3, 0.2)
	model.add_transition(s3, s3, 0.9)
	model.add_transition(s3, model.end, 0.1)
	model.bake()
	'''
	model = HiddenMarkovModel("Global Alignment")
	'''
	# Create the insert states
	i0 = State(i_d, name="I0")
	i1 = State(i_d, name="I1")
	i2 = State(i_d, name="I2")
	i3 = State(i_d, name="I3")

	# Create the match states
	m1 = State(d1, name="M1")
	m2 = State(d2, name="M2")
	m3 = State(d3, name="M3")

	# Create the delete states
	d1 = State(None, name="D1")
	d2 = State(None, name="D2")
	d3 = State(None, name="D3")
	'''
	i1 = State(Distribution(NormalDistribution(5, 1)))
	i2 = State(Distribution(NormalDistribution(1, 7)))
	i3 = State(Distribution(NormalDistribution(8, 2)))
	i0 = State(Distribution(NormalDistribution(3, 3)))
	m1 = State(Distribution(NormalDistribution(2, 4)))
	m2 = State(Distribution(NormalDistribution(4, 5)))
	m3 = State(Distribution(NormalDistribution(6, 6)))
	d1 = State(Distribution(NormalDistribution(7, 7)))
	d2 = State(Distribution(NormalDistribution(3, 8)))
	d3 = State(Distribution(NormalDistribution(1, 4)))

	# Add all the states to the model
	model.add_states(i0, i1, i2, i3, m1, m2, m3, d1, d2, d3)

	# Create transitions from match states
	model.add_transition(model.start, m1, 0.9)
	model.add_transition(model.start, i0, 0.1)
	model.add_transition(m1, m2, 0.9)
	model.add_transition(m1, i1, 0.05)
	model.add_transition(m1, d2, 0.05)
	model.add_transition(m2, m3, 0.9)
	model.add_transition(m2, i2, 0.05)
	model.add_transition(m2, d3, 0.05)
	model.add_transition(m3, model.end, 0.9)
	model.add_transition(m3, i3, 0.1)

	# Create transitions from insert states
	model.add_transition(i0, i0, 0.70)
	model.add_transition(i0, d1, 0.15)
	model.add_transition(i0, m1, 0.15)

	model.add_transition(i1, i1, 0.70)
	model.add_transition(i1, d2, 0.15)
	model.add_transition(i1, m2, 0.15)

	model.add_transition(i2, i2, 0.70)
	model.add_transition(i2, d3, 0.15)
	model.add_transition(i2, m3, 0.15)

	model.add_transition(i3, i3, 0.85)
	model.add_transition(i3, model.end, 0.15)

	# Create transitions from delete states
	model.add_transition(d1, d2, 0.15)
	model.add_transition(d1, i1, 0.15)
	model.add_transition(d1, m2, 0.70) 

	model.add_transition(d2, d3, 0.15)
	model.add_transition(d2, i2, 0.15)
	model.add_transition(d2, m3, 0.70)

	model.add_transition(d3, i3, 0.30)
	model.add_transition(d3, model.end, 0.70)

	# Call bake to finalize the structure of the model.
	model.bake()
	#states = model.predict(Y_test, algorithm = 'viterbi')
	model.fit(X_train.as_matrix())
	seq = model.predict(X_test)
	emissions = model.predict_proba(seq)
	print emissions.shape
	vals = []
	'''
	
	pred = model.sample(length=len(Y_test))
	print min(pred), max(pred), len(pred)
	vals = []
	for i in pred:
		vals.append(model.forward_backward(i))
	print len(vals), vals[0:5]
	
	mse = mean_squared_error(vals, Y_test)
	'''
	return vals

def autohmm(data, target):
	'''
	vals = np.diff(target['Target'])
	data = data[1:]
	data = np.column_stack([data, vals])
	'''
	vals = []
	X_train, X_test, Y_train, Y_test = train_test_split(data, target['Target'], test_size=0.33, random_state=42)
	states = clustering(target)
	model = ar.ARTHMM(n_unique = len(states))
	print len(Y_train), X_train.shape
	model.fit(X_train) #X_train.values for numpy array as lists of lists
	log_proba, state_seq = model.decode(X_test)
	print "state seq", len(state_seq), set(state_seq)
	path = state_seq
	prediction_state = np.argmax(model.transmat_[state_seq[-1],:])
	prediction = np.mean([state_seq[prediction_state], state_seq[prediction_state-1]])


	#prediction_state = np.argmax(a[path[-1],:])
	#prediction = np.argmax(b[prediction_state,:])

	vals.append(prediction)
	return state_seq, vals
	

def clustering(target):
	#model = GaussianMixture()
	#model.fit(target)
	events, edges, patches = hist(target['Target'])
	print "Edges", edges, len(edges)
	return edges


data, target = read_file(path)
data = data.reset_index()
target = target.reset_index()

data, target = preproc_basic(data, target)
preproc_correlation_data = featsel_correlation(data)
preproc_pca_data = featsel_pca(preproc_correlation_data)

#hmmlearn(preproc_pca_data, target)
#state_seq, vals = autohmm(preproc_pca_data, target)
pomegranate(preproc_pca_data, target)

