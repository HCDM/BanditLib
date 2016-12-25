import numpy as np
from scipy.linalg import sqrtm
import math
import time
import datetime
from util_functions import vectorize, matrixize


class CoLinUCBUserSharedStruct(object):
	def __init__(self, featureDimension, lambda_, userNum, W):
		self.W = W
		self.userNum = userNum
		self.A = lambda_*np.identity(n = featureDimension*userNum)
		self.b = np.zeros(featureDimension*userNum)
		self.AInv =  np.linalg.inv(self.A)
		
		self.UserTheta = np.zeros(shape = (featureDimension, userNum))
		self.CoTheta = np.zeros(shape = (featureDimension, userNum))

		self.BigW = np.kron(np.transpose(W), np.identity(n=featureDimension))

		self.CCA = np.dot(np.dot(self.BigW , self.AInv), np.transpose(self.BigW))
	def updateParameters(self, articlePicked, click,  userID, update):
		pass
	
	def getProb(self, alpha, article, userID):
		
		TempFeatureM = np.zeros(shape =(len(article.contextFeatureVector), self.userNum))
		TempFeatureM.T[userID] = article.contextFeatureVector
		TempFeatureV = vectorize(TempFeatureM)
		
		mean = np.dot(self.CoTheta.T[userID], article.contextFeatureVector)	
		var = np.sqrt(np.dot(np.dot(TempFeatureV, self.CCA), TempFeatureV))
		
		pta = mean + alpha * var
		
		return pta

class AsyCoLinUCBUserSharedStruct(CoLinUCBUserSharedStruct):	
	def updateParameters(self, articlePicked, click,  userID, update='Inv'):	
		
		X = vectorize(np.outer(articlePicked.contextFeatureVector, self.W.T[userID])) 
		self.A += np.outer(X, X)	
		self.b += click*X
		if update == 'Inv':
			self.AInv =  np.linalg.inv(self.A)
		else:
			self.AInv = self.AInv - float(np.dot(self.AInv, np.dot(outer, self.AInv)))/(1.0+np.dot(np.transpose(X), np.dot(self.AInv, X)  ))
		

		self.UserTheta = matrixize(np.dot(self.AInv, self.b), len(articlePicked.contextFeatureVector)) 
		self.CoTheta = np.dot(self.UserTheta, self.W)
		self.CCA = np.dot(np.dot(self.BigW , self.AInv), np.transpose(self.BigW))
		

class SyCoLinUCBUserSharedStruct(CoLinUCBUserSharedStruct):
	def __init__(self, featureDimension, lambda_, userNum, W):
		CoLinUCBUserSharedStruct.__init__(self, featureDimension, lambda_, userNum, W)
		self.contextFeatureVectorMatrix = np.zeros(shape =(featureDimension, userNum))
		self.reward = np.zeros(userNum)
	def updateParameters(self, articlePicked, click, userID):	
		self.contextFeatureVectorMatrix.T[userID] = articlePicked.contextFeatureVector
		self.reward[userID] = click
		featureDimension = len(self.contextFeatureVectorMatrix.T[userID])
		
	def LateUpdate(self):
		featureDimension = self.contextFeatureVectorMatrix.shape[0]
		current_A = np.zeros(shape = (featureDimension* self.userNum, featureDimension*self.userNum))
		current_b = np.zeros(featureDimension*self.userNum)		
		for i in range(self.userNum):
			X = vectorize(np.outer(self.contextFeatureVectorMatrix.T[i], self.W.T[i])) 
			XS = np.outer(X, X)	
			current_A += XS
			current_b += self.reward[i] * X
		self.A += current_A
		self.b += current_b
		self.AInv =  np.linalg.inv(self.A)

		self.UserTheta = matrixize(np.dot(self.AInv, self.b), featureDimension) 
		self.CoTheta = np.dot(self.UserTheta, self.W)
		self.CCA = np.dot(np.dot(self.BigW , self.AInv), np.transpose(self.BigW))

		
		
#---------------CoLinUCB(fixed user order) algorithms: Asynisized version and Synchorized version		
class CoLinUCBAlgorithm:
	def __init__(self, dimension, alpha, lambda_, n, W, update='inv'):  # n is number of users
		self.update = update #default is inverse. Could be 'rankone' instead.
		self.USERS = CoLinUCBUserSharedStruct(dimension, lambda_, n, W)
		self.dimension = dimension
		self.alpha = alpha
		self.W = W

		self.CanEstimateUserPreference = False
		self.CanEstimateCoUserPreference = False 
		self.CanEstimateW = False
		self.CanEstimateV = False
	def decide(self, pool_articles, userID):
		maxPTA = float('-inf')
		articlePicked = None

		for x in pool_articles:
			x_pta = self.USERS.getProb(self.alpha, x, userID)
			# pick article with highest Prob
			if maxPTA < x_pta:
				articlePicked = x
				maxPTA = x_pta

		return articlePicked
	def updateParameters(self, articlePicked, click, userID, update='Inv'):
		self.USERS.updateParameters(articlePicked, click, userID, update)
		
	def getLearntParameters(self, userID):
		return self.USERS.UserTheta.T[userID]

	def getCoTheta(self, userID):
		return self.USERS.CoTheta.T[userID]

	def getA(self):
		return self.USERS.A


class AsyCoLinUCBAlgorithm(CoLinUCBAlgorithm):
	def __init__(self, dimension, alpha, lambda_, n, W, update='Inv'):
		CoLinUCBAlgorithm.__init__(self, dimension, alpha, lambda_, n, W, update)
		self.USERS = AsyCoLinUCBUserSharedStruct(dimension, lambda_, n, W)
		
class syncCoLinUCBAlgorithm(CoLinUCBAlgorithm):
	def __init__(self, dimension, alpha, lambda_, n, W):
		CoLinUCBAlgorithm.__init__(self, dimension, alpha, lambda_, n, W)
		self.USERS = SyCoLinUCBUserSharedStruct(dimension, lambda_, n, W)

	def LateUpdate(self):
		self.USERS.LateUpdate()

#-----------CoLinUCB select user algorithm(only has asynchorize version)-----
class CoLinUCB_SelectUserAlgorithm:
	def __init__(self, dimension, alpha, lambda_, n, W):  # n is number of users
		self.USERS = AsyCoLinUCBUserSharedStruct(dimension, lambda_, n, W)  
		self.dimension = dimension
		self.alpha = alpha
		self.W = W

	def decide(self, pool_articles, AllUsers):
		maxPTA = float('-inf')
		articlePicked = None
		userPicked = None

		for x in pool_articles:
			for user in AllUsers:
				x_pta = self.USERS.getProb(self.alpha, x, user.id)
				# pick article with highest Prob
				if maxPTA < x_pta:
					articlePicked = x
					userPicked = user
					maxPTA = x_pta

		return userPicked,articlePicked
	def updateParameters(self, articlePicked, click, userID):
		self.USERS.updateParameters(articlePicked, click, userID)
		
	def getLearntParameters(self, userID):
		return self.USERS.UserTheta.T[userID]

	def getCoThetaFromCoLinUCB(self, userID):
		return self.USERS.CoTheta.T[userID]

	def getA(self):
		return self.USERS.A
	
