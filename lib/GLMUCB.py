import numpy as np
from scipy.linalg import sqrtm
import math
from util_functions import vectorize, matrixize
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
from scipy import stats
import numpy as np

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


def getbounds(dim):
	bnds = []
	for i in range(dim):
		bnds.append((0.00001, None))
	return tuple(bnds)


def Expontential_ParaEstimatation_Exp(X, Y, para):
	def fun(w, X = X, Y = Y):
		w = np.asarray(w)

		functionValue =0
		for i in range(len(X)):
			innerProduct = np.dot(X[i], w)
			functionValue = functionValue + ( Y[i]*np.exp(innerProduct) - innerProduct)
		return functionValue
	def evaluateGradient(w, X = X, Y = Y):
		w = np.asarray(w)
		X = np.asarray(X)
		Y = np.asarray(Y)
		grad = np.zeros(len(w))
		for i in range(len(X)):
			grad += ( X[i]*Y[i]*np.exp(np.dot(X[i], w)) - X[i] )
		return grad
	def evaluateHessian(w, X= X, Y = Y):

		w = np.asarray(w)
		X = np.asarray(X)
		Y = np.asarray(Y)
		Hessian = np.zeros([len(w), len(w)])
		for i in range(len(X)):
			Hessian +=  Y[i]*np.exp(np.dot(X[i], w) * np.outer(X[i],X[i]))  
		return Hessian

	res = minimize(fun, para, method ='L-BFGS-B',  jac = evaluateGradient, hess = evaluateHessian, options={'disp': False})
	NewPara = res.x
	#print 'success?', res.success
	return NewPara


def Expontential_ParaEstimatation(X, Y, para):
	def fun(w, X = X, Y = Y):
		w = np.asarray(w)
		#print 'w', w
		functionValue =0
		for i in range(len(X)):
			innerProduct = np.dot(X[i], w)
			#print 'WWW', w
			if innerProduct<=0:
				print innerProduct, X[i], w
			functionValue  += ( Y[i]*innerProduct -  math.log(innerProduct))
		return functionValue
	def evaluateGradient(w, X = X, Y = Y):
		w = np.asarray(w)
		X = np.asarray(X)
		Y = np.asarray(Y)
		grad = np.zeros(len(w))
		for i in range(len(X)):
			grad += ( X[i]*Y[i] - X[i]/float(np.dot(X[i], w)) )
		return grad
	#print 'para', para
	res = minimize(fun, para, method ='BFGS', jac = evaluateGradient, bounds=getbounds(len(para)), options={'disp': False})
	NewPara = res.x
	print 'success?', res.success

	return NewPara


class LogisticUserStruct:
	def __init__(self, featureDimension, userID, lambda_, RankoneInverse):
		self.userID = userID
		self.A = lambda_*np.identity(n = featureDimension)
		
		self.AInv = np.linalg.inv(self.A)
		self.UserTheta = np.zeros(featureDimension)
		self.RankoneInverse = RankoneInverse

		self.totalCount = 0
		self.totalClick = 0
		self.CTR = 0.0
		self.alpha = 0

		self.LogiModel = LogisticRegression()
		self.X = []
		self.Y = []

	def updateParameters(self, articlePicked, click):
		self.totalCount +=1
		if click == 1:
			self.totalClick +=1
		self.CTR = float(self.totalClick)/self.totalCount

		self.alpha = 0.1*np.sqrt(math.log(self.totalCount+1))

		self.X.append(articlePicked.featureVector)
		self.Y.append(click)

		featureVector = articlePicked.featureVector
		self.A += np.outer(featureVector, featureVector)
		
	def updateAInv(self):
		if self.RankoneInverse:
			temp = np.dot(self.AInv, featureVector)
			self.AInv = self.AInv - (np.outer(temp,temp))/(1.0+np.dot(np.transpose(featureVector),temp))
		else:
			self.AInv = np.linalg.inv(self.A)

	def getVar(self, featureVector):
		var = np.sqrt(np.dot(np.dot(featureVector, self.AInv), featureVector))
		return  var
	def getClickProb_arr(self, alpha, users, article_Feature_arr):
		if len(self.X) ==0:
			ClickProb_arr = np.zeros(len(article_Feature_arr))
		else:
			if 1 not in self.Y:
				ClickProb_arr = np.zeros(len(article_Feature_arr))
			elif 0 not in self.Y:
				ClickProb_arr = np.ones(len(article_Feature_arr))
			else:
				self.LogiModel.fit(self.X, self.Y)
				Prob_arr = self.LogiModel.predict_proba(article_Feature_arr)
				ClassList = self.LogiModel.classes_.tolist()
				ClickIndex = ClassList.index(1)
				ClickProb_arr = Prob_arr.T[ClickIndex]
				#print ClickProb_arr
		return ClickProb_arr


class PoissonStruct:
	def __init__(self,featureDimension, userID, lambda_, RankoneInverse):
		self.userID = userID
		self.A = lambda_*np.identity(n = featureDimension)
		self.AInv = np.linalg.inv(self.A)
		self.UserTheta = np.zeros(featureDimension)
		self.RankoneInverse = RankoneInverse
		self.count = 0


		self.para = np.zeros(featureDimension)   #zero initialization
		#self.para = np.random.random(featureDimension)  #random initialization
		self.X = []
		self.Y = []
		
	def updateParameters(self, articlePicked, time):
		self.count +=1
		self.X.append(articlePicked.featureVector)
		self.Y.append(time)
		self.alpha = 0.1*np.sqrt(math.log(self.count+1))

		returnFeature = articlePicked.featureVector
		self.A += np.outer(returnFeature, returnFeature)

		#self.AInv = np.linalg.inv(self.A)  #seperate the upate of matrix inverse to save computation

		currentpara = self.para
		ParaEstimatation = Expontential_ParaEstimatation_Exp(self.X, self.Y, currentpara)	
		self.para = ParaEstimatation
	def updateAInv(self):
		#seperate the upate of matrix inverse to save computation
		self.AInv = np.linalg.inv(self.A)

	def getReturnPro(self, article_FeatureVector, ReturnThreshold):
		linear = np.dot(self.para, article_FeatureVector)
		intensity = np.exp(linear)
	
		returnProb = 1-np.exp(-intensity*ReturnThreshold)
		returnVar = np.sqrt(np.dot(np.dot(article_FeatureVector, self.AInv), article_FeatureVector))
		return returnProb, returnVar

	def getReturnPro_Arr(self, a_FeatureVector_arr, ReturnThreshold):
		linear_arr = np.dot( a_FeatureVector_arr, self.para.T)
		intensity_arr = np.exp(linear_arr)
		returnProb_arr = 1-np.exp(-intensity_arr*ReturnThreshold)
		return returnProb_arr

	def getBeta(self):
		return self.para



class reward_GLMUCBAlgorithm:
	def __init__(self, dimension, alpha, lambda_, usealphaT = False, RankoneInverse = False):
		self.users = {}
		self.dimension = dimension
		self.alpha = alpha
		self.lambda_ = lambda_
		self.totalCount = 0.0

		self.usealphaT = usealphaT
		self.RankoneInverse = RankoneInverse
		self.CanEstimateUserPreference = True
		self.CanEstimateReturn = False

	def decide(self, pool_articles, userID, pool_article_Arr):
		if userID not in self.users:
			self.users[userID] =LogisticUserStruct(self.dimension, userID, self.lambda_ ,self.RankoneInverse)

		maxUCB = float('-inf')
		articlePicked = None
		returnProb = 1.0
		ClickProb_arr = self.users[userID].getClickProb_arr(self.alpha, self.users, pool_article_Arr)

		if self.usealphaT:
			self.alpha =  self.users[userID].alpha
		#print self.alpha
		i = 0
		for x in pool_articles:
			articleFeature = x.featureVector
			#get mean of click probability
			ClickProb = ClickProb_arr[i]
			i +=1
			#get variance of click probability
			var = self.users[userID].getVar(articleFeature)
			#get ucb of click probabilit
			click_ucb = ClickProb + self.alpha*var
			if maxUCB < click_ucb:
				articlePicked = x
				maxUCB = click_ucb

		return articlePicked
	def updateParameters(self, articlePicked, click, userID, time):
		self.users[userID].updateParameters(articlePicked, click)
		self.users[userID].updateAInv()	
	def getTheta(self, userID):
		return self.users[userID].UserTheta

class return_GLMUCBAlgorithm(reward_GLMUCBAlgorithm):
	def __init__(self, dimension, alpha, lambda_, ReturnThreshold, usealphaT = False, RankoneInverse = False):
		reward_GLMUCBAlgorithm.__init__(self, dimension, alpha, lambda_, usealphaT = usealphaT, RankoneInverse = False)
		self.ReturnThreshold = ReturnThreshold
		self.CanEstimateUserPreference = False
		self.CanEstimateReturn = True

	def decide(self, pool_articles, userID, pool_article_Arr):
		if userID not in self.users:
			self.users[userID] = PoissonStruct(self.dimension, userID, self.lambda_, self.RankoneInverse)
		maxUCB = float('-inf')
		articlePicked = None
		if self.usealphaT:
			self.alpha =  self.users[userID].alpha
		#print self.alpha
		for x in pool_articles:		
			articleFeature = x.featureVector	
			returnProb, returnVar = self.users[userID].getReturnPro(articleFeature, self.ReturnThreshold)
			return_ucb = returnProb + self.alpha*returnVar
			if maxUCB < return_ucb:
				articlePicked = x
				maxUCB = return_ucb
		return articlePicked
	def updateParameters(self, articlePicked, click, userID, time):
		self.users[userID].updateParameters(articlePicked, time)
		self.users[userID].updateAInv()	
	def getBeta(self,userID):
		return self.users[userID].getBeta()
	



