from collections import Counter
from math import log
import numpy as np 
from random import *
from custom_errors import FileExists 

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
	return np.reshape(M, M.shape[0]*M.shape[1])

def matrixize(V, C_dimension):
	# temp = np.zeros(shape = (C_dimension, len(V)/C_dimension))
	# for i in range(len(V)/C_dimension):
	# 	temp.T[i] = V[i*C_dimension : (i+1)*C_dimension]
	# W = temp
	# return W
	#To-do: use numpy built-in function reshape.
	return np.reshape(V, (C_dimension, int(len(V)/C_dimension)))
