import numpy as np
from scipy.linalg import sqrtm
import math
import time
import datetime

from util_functions import vectorize, matrixize
from CoLin import CoLinUCBAlgorithm, CoLinUCB_SelectUserAlgorithm




class GOBLinSharedStruct:
	def __init__(self, featureDimension, lambda_, userNum, W):
		self.W = W
		self.userNum = userNum
		self.A = lambda_*np.identity(n = featureDimension*userNum)
		self.b = np.zeros(featureDimension*userNum)
		self.AInv = np.linalg.inv(self.A)

		self.theta = np.dot(self.AInv , self.b)
		self.STBigWInv = sqrtm( np.linalg.inv(np.kron(W, np.identity(n=featureDimension))) )
		self.STBigW = sqrtm(np.kron(W, np.identity(n=featureDimension)))
	def updateParameters(self, articlePicked, click, userID):
		featureVectorM = np.zeros(shape =(len(articlePicked.contextFeatureVector), self.userNum))
		featureVectorM.T[userID] = articlePicked.contextFeatureVector
		featureVectorV = vectorize(featureVectorM)

		CoFeaV = np.dot(self.STBigWInv, featureVectorV)
		self.A += np.outer(CoFeaV, CoFeaV)
		self.b += click * CoFeaV

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
	def __init__(self, dimension, alpha, lambda_, n, W):
		CoLinUCBAlgorithm.__init__(self, dimension, alpha, lambda_, n, W)
		self.USERS = GOBLinSharedStruct(dimension, lambda_, n, W)
	def getLearntParameters(self, userID):
		thetaMatrix =  matrixize(self.USERS.theta, self.dimension) 
		return thetaMatrix.T[userID]

#inherite from CoLinUCB_SelectUserAlgorithm
class GOBLin_SelectUserAlgorithm(CoLinUCB_SelectUserAlgorithm):
	def __init__(self, dimension, alpha, lambda_, n, W):
		CoLinUCB_SelectUserAlgorithm.__init__(self, dimension, alpha, lambda_, n, W)
		self.USERS = GOBLinSharedStruct(dimension, lambda_, n, W)
	def getLearntParameters(self, userID):
		thetaMatrix =  matrixize(self.USERS.theta, self.dimension) 
		return thetaMatrix.T[userID]