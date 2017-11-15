from collections import Counter
from math import log
import numpy as np 
from random import *
from custom_errors import FileExists 

def createLinUCBDict(starter):
	return_dict = {}
	return_dict['dimension'] = starter['dimension'] if starter.has_key('dimension') else 16
	return_dict['alpha'] = starter['alpha'] if starter.has_key('alpha') else 0.3
	return_dict['lambda_'] = starter['lambda_'] if starter.has_key('lambda_') else 0.1
	return_dict['n_users'] = starter['n_users'] if starter.has_key('n_users') else 10
	return return_dict

def createCoLinDict(starter, W):
	return_dict = {}
	# 	algorithms['CoLin'] = AsyCoLinUCBAlgorithm(dimension=context_dimension, alpha = alpha, lambda_ = lambda_, n = n_users, W = UM.getW())
	return_dict['dimension'] = starter['dimension'] if starter.has_key('dimension') else 16
	return_dict['alpha'] = starter['alpha'] if starter.has_key('alpha') else 0.3
	return_dict['lambda_'] = starter['lambda_'] if starter.has_key('lambda_') else 0.1
	return_dict['n_users'] = starter['n_users'] if starter.has_key('n_users') else 10
	return_dict['W'] = W
	return return_dict

def createHLinUCBDict(starter):
	return_dict = {}
	#HLinUCBAlgorithm(context_dimension = context_dimension, latent_dimension = latent_dimension, alpha = 0.1, alpha2 = 0.1, lambda_ = lambda_, n = n_users, itemNum=n_articles, init='zero', window_size = -1)
	return_dict['context_dimension'] = starter['context_dimension'] if starter.has_key('context_dimension') else 16
	return_dict['latent_dimension'] = starter['latent_dimension'] if starter.has_key('latent_dimension') else 0
	return_dict['alpha'] = starter['alpha'] if starter.has_key('alpha') else 0.1
	return_dict['alpha2'] = starter['alpha2'] if starter.has_key('alpha2') else 0.1
	return_dict['lambda_'] = starter['lambda_'] if starter.has_key('lambda_') else 0.1
	return_dict['n_users'] = starter['n_users'] if starter.has_key('n_users') else 10
	return_dict['n_articles'] = starter['n_articles'] if starter.has_key('n_articles') else 1000
	return return_dict

def gaussianFeature(dimension, argv):
	mean = argv['mean'] if 'mean' in argv else 0
	std = argv['std'] if 'std' in argv else 1

	mean_vector = np.ones(dimension)*mean
	stdev = np.identity(dimension)*std
	vector = np.random.multivariate_normal(np.zeros(dimension), stdev)

	l2_norm = np.linalg.norm(vector, ord = 2)
	if 'l2_limit' in argv and l2_norm > argv['l2_limit']:
		"This makes it uniform over the circular range"
		vector = (vector / l2_norm)
		vector = vector * (random())
		vector = vector * argv['l2_limit']

	if mean is not 0:
		vector = vector + mean_vector

	vectorNormalized = []
	for i in range(len(vector)):
		vectorNormalized.append(vector[i]/sum(vector))
	return vectorNormalized
	#return vector

def featureUniform(dimension, argv = None):
	vector = np.array([random() for _ in range(dimension)])

	l2_norm = np.linalg.norm(vector, ord =2)
	
	vector = vector/l2_norm
	return vector

def getBatchStats(arr):
	return np.concatenate((np.array([arr[0]]), np.diff(arr)))

def checkFileExists(filename):
	try:
		with open(filename, 'r'):
			return 1
	except IOError:
		return 0 

def fileOverWriteWarning(filename, force):
	if checkFileExists(filename):
		if force == True:
			print "Warning : fileOverWriteWarning %s"%(filename)
		else:
			raise FileExists(filename)


def vectorize(M):
	# temp = []
	# for i in range(M.shape[0]*M.shape[1]):
	# 	temp.append(M.T.item(i))
	# V = np.asarray(temp)
	# return V
	return np.reshape(M.T, M.shape[0]*M.shape[1])

def matrixize(V, C_dimension):
	# temp = np.zeros(shape = (C_dimension, len(V)/C_dimension))
	# for i in range(len(V)/C_dimension):
	# 	temp.T[i] = V[i*C_dimension : (i+1)*C_dimension]
	# W = temp
	# return W
	#To-do: use numpy built-in function reshape.
	return np.transpose(np.reshape(V, ( int(len(V)/C_dimension), C_dimension)))
