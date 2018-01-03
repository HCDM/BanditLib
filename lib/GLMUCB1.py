import numpy as np
from scipy.linalg import sqrtm
import math
from util_functions import vectorize, matrixize
from sklearn.linear_model import Ridge
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize


import statsmodels.api as sm
from scipy import stats
import numpy as np
import random
import LinUCB
from sklearn.linear_model import LogisticRegression
from GLMUCB import sigmoid, Expontential_ParaEstimatation_Exp, LogisticUserStruct, PoissonStruct


class UCB1_Article:
	def __init__(self, id_):
		self.article_id = id_
		self.totalClick = 0
		self.totalReturnTime = 0.0
		self.PlayCount_click = 0
		self.PlayCount_return = 0
		self.LogiModel = LogisticRegression()
		self.X = []
		self.Y = []
	def updateClickInfo(self, click):
		self.totalClick +=click
		self.PlayCount_click +=1

		self.X.append([1])
		self.Y.append(click)
	def updateReturnInfo(self, ReturnTime):
		self.totalReturnTime += ReturnTime
		self.PlayCount_return +=1
	def getAverageClick(self):
		self.averageClick = self.totalClick/float(self.PlayCount_click)
		return self.averageClick

	def getClickProb(self):
		if len(self.X) ==0:
			ClickProb = 0.0
		else:
			if 1 not in self.Y:
				#ClickProb = random.uniform(0,0.5)
				ClickProb = 0
			elif 0 not in self.Y:
				#ClickProb = random.uniform(0.5,1)
				ClickProb = 1
			else:
				sample = np.array([1]).reshape(1, -1)  # Reshpae to avoid error, 1-d feature
				self.LogiModel.fit(self.X, self.Y)
				Prob_arr = self.LogiModel.predict_proba(sample)
				ClassList = self.LogiModel.classes_.tolist()
				ClickIndex = ClassList.index(1)
				ClickProb = Prob_arr.T[ClickIndex]
		return ClickProb

	def getAverageReturnProb(self, ReturnThreshold):
		try:
			self.averageReturnTime = self.totalReturnTime/float(self.PlayCount_return)
		except ZeroDivisionError:
			self.averageReturnTime = 0.0
		#print self.totalReturnTime
		Intensity = float(self.PlayCount_return)/float(self.totalReturnTime)
		returnProb = 1-np.exp(-Intensity*ReturnThreshold)
		return returnProb


class UCB1UserStruct:
	def __init__(self,userID):
		self.userID = userID
		self.ArticleDic = {}
		self.totalCount = 0.0
		self.totalClick = 0.0
		self.CTR = 0.0
	def updateParameters(self, articlePicked, click, ReturnTime):
		articleID = articlePicked.id
		if articleID not in self.ArticleDic:
			self.ArticleDic[articleID] = UCB1_Article(articleID)
		self.ArticleDic[articleID].updateClickInfo(click)
		self.ArticleDic[articleID].updateReturnInfo(ReturnTime)

		self.totalCount+=1.0
		self.totalClick +=click
		self.CTR = self.totalClick/self.totalCount
	def getVar(self, articlePicked, totalCount):
		articleID = articlePicked.id
		articleChosenNum = self.ArticleDic[articleID].PlayCount_click
		var = np.sqrt(2*np.log(totalCount)/float(articleChosenNum))

		return var
		
class r2_GLMUCB1Algorithm:
	def __init__(self, dimension, alpha, lambda_, FutureWeight, ReturnThreshold, usealphaT, RankoneInverse):
		self.dimension = dimension
		self.alpha = alpha
		self.lambda_ = lambda_
		self.FutureWeight = FutureWeight
		self.ReturnThreshold = ReturnThreshold
		self.usealphaT = usealphaT
		self.RankoneInverse = RankoneInverse

		self.users = {}
		self.users_click = {}
		self.users_return ={}

		self.CanEstimateUserPreference = False
		self.CanEstimateReturn = False
		
	def decide(self, pool_articles, userID, pool_article_Arr ):
		if userID not in self.users:
			self.users[userID] = UCB1UserStruct(userID)
			self.users_click[userID]  =LogisticUserStruct(featureDimension = self.dimension, userID = userID, lambda_ = self.lambda_, RankoneInverse = self.RankoneInverse  )
			self.users_return[userID] = PoissonStruct(featureDimension = self.dimension, userID = userID, lambda_  = self.lambda_, RankoneInverse = self.RankoneInverse)

		if self.FutureWeight == None:
			FutureWeight = self.users[userID].CTR
			print 'ctr', FutureWeight
		else:
			FutureWeight = float(self.FutureWeight)

		maxUCB = float('-inf')
		articlePicked = None

		ClickProb_arr = self.users_click[userID].getClickProb_arr(self.alpha, self.users, pool_article_Arr)
		i = 0
		for x in pool_articles:
			x_ID = x.id
			articleFeature = x.featureVector
			if x_ID not in self.users[userID].ArticleDic:
				articlePicked = x
				return articlePicked
			else:
				clickProb = ClickProb_arr[i]
				returnProb = self.users_return[userID].getReturnPro(articleFeature, self.ReturnThreshold)[0]
				var = self.users[userID].getVar(x, self.users[userID].totalCount)
				ucb = clickProb +  FutureWeight*returnProb+ (1+FutureWeight)*self.alpha* var
				# pick article with highest Prob
				if maxUCB < ucb:
					articlePicked = x
					maxUCB = ucb
			i +=1
		return articlePicked
	def updateParameters(self, articlePicked, click, userID, ReturnTime ):
		self.users[userID].updateParameters(articlePicked, click, ReturnTime)
		self.users_click[userID].updateParameters(articlePicked, click)
		self.users_return[userID].updateParameters(articlePicked, ReturnTime)



