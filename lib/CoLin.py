import numpy as np
from scipy.linalg import sqrtm
import math
import time
import datetime
from util_functions import vectorize, matrixize
from Recommendation import Recommendation
from BaseAlg import BaseAlg
import warnings

class CoLinUCBUserSharedStruct(object):
	def __init__(self, featureDimension, lambda_, userNum, W):
		self.currentW = np.identity(n = userNum)

		self.W = W
		print "W: ", self.W
		self.userNum = userNum
		self.A = lambda_*np.identity(n = featureDimension*userNum)
		self.b = np.zeros(featureDimension*userNum)
		self.AInv =  np.linalg.inv(self.A)
		
		self.UserTheta = np.zeros(shape = (featureDimension, userNum))
		self.CoTheta = np.zeros(shape = (featureDimension, userNum))

		self.BigW = np.kron(np.transpose(W), np.identity(n=featureDimension))
		print "Big W: ", self.BigW
		self.CCA = np.dot(np.dot(self.BigW , self.AInv), np.transpose(self.BigW))
		self.alpha_t = 0.0
		self.sigma = 1.e-200   #Used in the high probability bound, i.e, with probability at least (1 - sigma) the confidence bound. So sigma should be very small
		self.lambda_ = lambda_
	def updateParameters(self, articlePicked, click,  userID, update='Inv'):
		X = vectorize(np.outer(articlePicked.contextFeatureVector, self.W.T[userID]))
		#print "X: " + str(X)
		change = np.outer(X, X)
		self.A += change
		self.b += click*X
		if update == 'Inv':
			self.AInv =  np.linalg.inv(self.A)
		else:
			self.AInv = self.AInv - float(np.dot(self.AInv, np.dot(outer, self.AInv)))/(1.0+np.dot(np.transpose(X), np.dot(self.AInv, X)  ))
		self.UserTheta = matrixize(np.dot(self.AInv, self.b), len(articlePicked.contextFeatureVector)) 
		self.CoTheta = np.dot(self.UserTheta, self.W)
		self.CCA = np.dot(np.dot(self.BigW , self.AInv), np.transpose(self.BigW))
	
	def getProb(self, alpha, article, userID):
		warnings.filterwarnings('error')
		TempFeatureM = np.zeros(shape =(len(article.contextFeatureVector), self.userNum))
		TempFeatureM.T[userID] = article.contextFeatureVector
		TempFeatureV = vectorize(TempFeatureM)
		
		mean = np.dot(self.CoTheta.T[userID], article.contextFeatureVector)	
		var = np.sqrt(np.dot(np.dot(TempFeatureV, self.CCA), TempFeatureV))

		#self.alpha_t = 0.01*np.sqrt(np.log(np.linalg.det(self.A)/float(self.sigma * self.lambda_) )) + np.sqrt(self.lambda_)
		try:
			self.alpha_t = 0.01*np.sqrt(np.log(np.linalg.det(self.A)/float(self.sigma * self.lambda_) )) + np.sqrt(self.lambda_)
		except:
			self.alpha_t = 0.0
		#pta = mean + alpha * var    # use emprically tuned alpha
		pta = mean + self.alpha_t *var   # use the theoretically computed alpha_t
		
		return pta

	def getUserCoTheta(self, userID):
		return self.CoTheta.T[userID]

	def getCCA(self):
		return self.CCA	
	def calculateAlphaT(self):
		warnings.filterwarnings('error')
		try:
			self.alpha_t = 0.01*np.sqrt(np.log(np.linalg.det(self.A)/float(self.sigma * self.lambda_) )) + np.sqrt(self.lambda_)
		except:
			self.alpha_t = 0.0
		return self.alpha_t

#---------------CoLinUCB(fixed user order) algorithms: Asynisized version and Synchorized version		
class CoLinUCBAlgorithm(BaseAlg):
	def __init__(self, arg_dict, update='inv'):  # n is number of users
		BaseAlg.__init__(self, arg_dict)
		self.update = update #default is inverse. Could be 'rankone' instead.
		self.USERS = CoLinUCBUserSharedStruct(arg_dict['dimension'], arg_dict['lambda_'], arg_dict['n_users'], arg_dict['W'])

	def decide_old(self, pool_articles, userID, exclude = []):
		maxPTA = float('-inf')
		articlePicked = None

		for x in pool_articles:
			x_pta = self.USERS.getProb(self.alpha, x, userID)
			# pick article with highest Prob
			if maxPTA < x_pta:
				articlePicked = x
				maxPTA = x_pta

		return [articlePicked]

	def decide(self, pool_articles, userID, k = 1):
		# MEAN
		art_features = np.empty([len(pool_articles), len(pool_articles[0].contextFeatureVector)])
		for i in range(len(pool_articles)):
			art_features[i, :] = pool_articles[i].contextFeatureVector
		user_features = self.USERS.CoTheta.T[userID]
		mean_matrix = np.dot(art_features, user_features)

		# VARIANCE
		art_temp_features = np.empty([len(pool_articles), len(pool_articles[0].contextFeatureVector)*self.n_users])
		for i in range(len(pool_articles)):
			TempFeatureM = np.zeros(shape =(len(pool_articles[0].contextFeatureVector), self.n_users))
			TempFeatureM.T[userID] = pool_articles[i].contextFeatureVector
			art_temp_features[i, :] = vectorize(TempFeatureM)
		var_matrix = np.sqrt(np.dot(np.dot(art_temp_features, self.USERS.CCA), art_temp_features.T))
		#self.USERS.calculateAlphaT()
		if self.use_alpha_t:

			self.USERS.calculateAlphaT()
			pta_matrix = mean_matrix + self.USERS.alpha_t*np.diag(var_matrix)
		else:
			pta_matrix = mean_matrix + self.alpha*np.diag(var_matrix)

		pool_positions = np.argsort(pta_matrix)[(k*-1):]
		articles = []
		for i in range(k):
			articles.append(pool_articles[pool_positions[i]])

		return articles

		#return pool_articles[pool_position]

	def updateParameters(self, articlePicked, click, userID, update='Inv'):
		self.USERS.updateParameters(articlePicked, click, userID, update)
		
	def getLearntParameters(self, userID):
		return self.USERS.UserTheta.T[userID]

	def getTheta(self, userID):
		return self.USERS.UserTheta.T[userID]

	def getCoTheta(self, userID):
		return self.USERS.CoTheta.T[userID]

	def getA(self):
		return self.USERS.A

	def getW(self, userID):
		return self.USERS.W



	
