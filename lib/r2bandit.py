import numpy as np
from GLMUCB import LogisticUserStruct, PoissonStruct, reward_GLMUCBAlgorithm

class r2_banditAlgorithm:
	def __init__(self, dimension, alpha, lambda_, FutureWeight, ReturnThreshold,  usealphaT = False, RankoneInverse = False):
		self.users = {}
		self.dimension = dimension
		self.alpha = alpha
		self.lambda_ = lambda_
		self.totalCount = 0.0

		self.usealphaT = usealphaT
		self.RankoneInverse = RankoneInverse
		self.CanEstimateUserPreference = True
		self.CanEstimateReturn = True

		self.ReturnThreshold = ReturnThreshold
		self.FutureWeight = FutureWeight
		self.users = {}
		self.users_return ={}

	def decide(self, pool_articles, userID, pool_article_Arr):
		if userID not in self.users:
			self.users[userID] = LogisticUserStruct(self.dimension, userID, self.lambda_ ,self.RankoneInverse) 
		if userID not in self.users_return:
			self.users_return[userID] = PoissonStruct(self.dimension, userID, self.lambda_, self.RankoneInverse)

		# get the weight for future click
		# the weight can be specified as a input hyper-parameter.
		# If it is not specified, compute the weight according to the historical click
		if self.FutureWeight == None:
			FutureWeight = self.users[userID].CTR
		else:
			FutureWeight = float(self.FutureWeight)

		# 'usealphaT' is a flag to compute alpha on the fly, the default setting is to use a pre-specified alpha
		if self.usealphaT:
			self.alpha =  self.users[userID].alpha

		maxUCB = float('-inf')
		articlePicked = None
		ClickProb_arr = self.users[userID].getClickProb_arr(self.alpha, self.users, pool_article_Arr)
		i = 0
		for x in pool_articles:
			articleFeature = x.featureVector
			# get current user's click probability on this arm
			ClickProb = ClickProb_arr[i]
			# get current user's return probability after seeing this arm
			returnProb = self.users_return[userID].getReturnPro(articleFeature, self.ReturnThreshold)[0]
			
			# get the variance of click and return estimation
			# note that since both click and return probability are estimated using generalized linear model (GLM)
			# the variacne for both click and return probability estimation are of the same form
			var = self.users[userID].getVar(articleFeature)
			# combing the predicted immediate click probability, future click
			ucb = ClickProb + FutureWeight*returnProb + (1+FutureWeight)*self.alpha*var 
			i +=1
			if maxUCB < ucb:
				articlePicked = x
				maxUCB = ucb
		return articlePicked
	def updateParameters(self, articlePicked, click, userID, time):
		self.users[userID].updateParameters(articlePicked, click)
		self.users[userID].updateAInv()
		self.users_return[userID].updateParameters(articlePicked, time)
		#self.users_return[userID].updateAInv()
	def getBeta(self,userID):
		return self.users_return[userID].getBeta()
	def getTheta(self, userID):
		return self.users[userID].UserTheta




