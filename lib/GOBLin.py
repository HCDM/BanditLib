import numpy as np
from scipy.linalg import sqrtm
import math
import time
import datetime

from util_functions import vectorize, matrixize
#from CoLin import CoLinUCBAlgorithm, CoLinUCB_SelectUserAlgorithm
from CoLin import CoLinUCBAlgorithm




class GOBLinSharedStruct:
	def __init__(self, featureDimension, lambda_, userNum, W):
		self.W = W
		self.userNum = userNum
		self.A = lambda_*np.identity(n = featureDimension*userNum)
		self.b = np.zeros(featureDimension*userNum)
		self.AInv = np.linalg.inv(self.A)

		self.theta = np.dot(self.AInv , self.b)
		print np.kron(W, np.identity(n=featureDimension))
		self.STBigWInv = sqrtm( np.linalg.inv(np.kron(W, np.identity(n=featureDimension))) )
		self.STBigW = sqrtm(np.kron(W, np.identity(n=featureDimension)))
	def updateParameters(self, articlePicked, click, userID, update):
		featureVectorM = np.zeros(shape =(len(articlePicked.contextFeatureVector), self.userNum))
		featureVectorM.T[userID] = articlePicked.contextFeatureVector
		featureVectorV = vectorize(featureVectorM)

		CoFeaV = np.dot(self.STBigWInv, featureVectorV)
		result = np.outer(CoFeaV, CoFeaV)
		self.A = self.A + result
		self.b = self.b + click * CoFeaV

		self.AInv = np.linalg.inv(self.A)

		self.theta = np.dot(self.AInv, self.b)
	def getProb(self,alpha , article, userID):
		
		featureVectorM = np.zeros(shape =(len(article.contextFeatureVector), self.userNum))
		featureVectorM.T[userID] = article.contextFeatureVector
		featureVectorV = vectorize(featureVectorM)

		CoFeaV = np.dot(self.STBigWInv, featureVectorV)
		
		mean = np.dot(np.transpose(self.theta), CoFeaV)		
		a = np.dot(CoFeaV, self.AInv)
		var = np.sqrt( np.dot( np.dot(CoFeaV, self.AInv) , CoFeaV))
		
		pta = mean + alpha * var
		
		return pta
# inherite from CoLinUCBAlgorithm
class GOBLinAlgorithm(CoLinUCBAlgorithm):
	def __init__(self, arg_dict):
		CoLinUCBAlgorithm.__init__(self, arg_dict)
		self.USERS = GOBLinSharedStruct(self.dimension, self.lambda_, self.n_users, self.W)
		#self.estimates['CanEstimateCoUserPreference'] = False
	def getLearntParameters(self, userID):
		thetaMatrix =  matrixize(self.USERS.theta, self.dimension) 
		return thetaMatrix.T[userID]

	def decide(self, pool_articles, userID, k = 1):
		# MEAN
		art_features = np.empty([len(pool_articles), len(pool_articles[0].contextFeatureVector)*self.n_users])
		for i in range(len(pool_articles)):
			TempFeatureM = np.zeros(shape =(len(pool_articles[0].contextFeatureVector), self.n_users))
			TempFeatureM.T[userID] = pool_articles[i].contextFeatureVector
			art_features[i, :] = vectorize(TempFeatureM)
		CoFeaV = np.dot(art_features, self.USERS.STBigWInv)
		mean_matrix = np.dot(CoFeaV, self.USERS.theta)
		var_matrix = np.sqrt(np.dot(np.dot(CoFeaV, self.USERS.AInv), CoFeaV.T).clip(0))
		pta_matrix = mean_matrix + self.alpha*np.diag(var_matrix)


		pool_positions = np.argsort(pta_matrix)[(k*-1):]
		articles = []
		for i in range(k):
			articles.append(pool_articles[pool_positions[i]])
		return articles
		# pool_position = np.argmax(pta_matrix)
		# return pool_articles[pool_position]

#inherite from CoLinUCB_SelectUserAlgorithm
# class GOBLin_SelectUserAlgorithm(CoLinUCB_SelectUserAlgorithm):
# 	def __init__(self, dimension, alpha, lambda_, n, W):
# 		CoLinUCB_SelectUserAlgorithm.__init__(self, dimension, alpha, lambda_, n, W)
# 		self.USERS = GOBLinSharedStruct(dimension, lambda_, n, W)
# 	def getLearntParameters(self, userID):
# 		thetaMatrix =  matrixize(self.USERS.theta, self.dimension) 
# 		return thetaMatrix.T[userID]