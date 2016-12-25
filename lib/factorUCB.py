import numpy as np
from util_functions import vectorize, matrixize
class FactorUCBArticleStruct:
	def __init__(self, id, context_dimension, latent_dimension, lambda_, W, init="zero", context_feature=None):
		self.W = W
		self.id = id
		self.context_dimension = context_dimension
		self.latent_dimension = latent_dimension
		self.d = context_dimension+latent_dimension

		self.A2 = lambda_*np.identity(n = self.latent_dimension)
		self.b2 = np.zeros(self.latent_dimension)
		self.A2Inv = np.linalg.inv(self.A2)

		self.count = {}
		self.time = 0
		if (init=="random"):
			self.V = np.random.rand(self.d)
		else:
			self.V = np.zeros(self.d)

	def updateParameters(self, user, click, userID):
		self.time += 1
		if userID in self.count:
			self.count[userID] += 1
		else:
			self.count[userID] = 1

		self.A2 += np.outer(user.CoTheta.T[userID][self.context_dimension:], user.CoTheta.T[userID][self.context_dimension:])
		self.b2 += user.CoTheta.T[userID][self.context_dimension:]*(click - user.CoTheta.T[userID][:self.context_dimension].dot(self.V[:self.context_dimension]))
		self.A2Inv  = np.linalg.inv(self.A2)

		self.V[self.context_dimension:] = np.dot(self.A2Inv, self.b2)

	def getCount(self, user_id):
		if user_id in self.count:
			return self.count[user_id]
		else:
			return 0
class FactorUCBUserStruct:
	def __init__(self, context_dimension, latent_dimension, lambda_, userNum, W, init="zero"):
		self.W = W
		self.context_dimension = context_dimension
		self.latent_dimension = latent_dimension
		self.d = context_dimension+latent_dimension

		self.userNum = userNum
		self.A = lambda_*np.identity(n = self.d*userNum)
		self.CCA = np.identity(n = self.d*userNum)
		self.b = np.zeros(self.d*userNum)
		self.AInv = np.linalg.inv(self.A)


		self.count = []
		for i in range(userNum):
			self.count.append({})

		self.time = 0
		if (init=="random"):
			self.UserTheta = np.random.rand(self.d, userNum)
			self.CoTheta = np.dot(self.UserTheta, self.W)
		else:
			self.UserTheta = np.zeros(shape = (self.d, userNum))
			self.CoTheta = np.zeros(shape = (self.d, userNum))

		self.BigW = np.kron(np.transpose(W), np.identity(n=self.d))
		# self.U = np.zeros(self.d)
	def updateParameters(self, articles, clicks, userID):
		self.time += len(articles)
		for article, click in zip(articles, clicks):
			if article.id in self.count[userID]:
				self.count[userID][article.id] += 1
			else:
				self.count[userID][article.id] = 1
			X = vectorize(np.outer(article.V, self.W.T[userID])) 
			self.A += np.outer(X, X)	
			self.b += click*X

		self.AInv =  np.linalg.inv(self.A)

		self.UserTheta = matrixize(np.dot(self.AInv, self.b), len(articles[0].V)) 
		self.CoTheta = np.dot(self.UserTheta, self.W)
		self.CCA = np.dot(np.dot(self.BigW , self.AInv), np.transpose(self.BigW))				
	
	def getA(self):
		return self.A

	def getProb(self, alpha, alpha2, article, userID):
		if alpha == -1:
			alpha = 0.1*np.sqrt(np.log(self.time+1))+0.1*(1-0.8**self.time)
			alpha2 = 0.1*np.sqrt(np.log(article.time+1))+0.1*(1-0.8**article.time)

		TempFeatureM = np.zeros(shape =(len(article.V), self.userNum))
		TempFeatureM.T[userID] = article.V
		TempFeatureV = vectorize(TempFeatureM)

		mean = np.dot(self.CoTheta.T[userID], article.V)	
		var = np.sqrt(np.dot(np.dot(TempFeatureV, self.CCA), TempFeatureV))
		var2 = np.sqrt(np.dot(np.dot(self.CoTheta.T[userID][self.context_dimension:], article.A2Inv),  self.CoTheta.T[userID][self.context_dimension:]))
		pta = mean + alpha * var + alpha2*var2
		return pta
	def getProb_plot(self, alpha, alpha2, article, userID):
		TempFeatureM = np.zeros(shape =(len(article.V), self.userNum))
		TempFeatureM.T[userID] = article.V
		TempFeatureV = vectorize(TempFeatureM)

		mean = np.dot(self.CoTheta.T[userID], article.V)	
		var = np.sqrt(np.dot(np.dot(TempFeatureV, self.CCA), TempFeatureV))
		var2 = np.sqrt(np.dot(np.dot(self.CoTheta.T[userID][self.context_dimension:], article.A2Inv),  self.CoTheta.T[userID][self.context_dimension:]))
		pta = mean + alpha * var + alpha2*var2
		return pta, mean, alpha*var

	def getCount(self, article_id, userID):
		if article_id in self.count[userID]:
			return self.count[userID][article_id]
		else:
			return 0

class FactorUCBAlgorithm:
	def __init__(self, context_dimension, latent_dimension, alpha, alpha2, lambda_, n, itemNum, W, init="zero", window_size = 1, max_window_size = 10):  # n is number of users

		self.context_dimension = context_dimension
		self.latent_dimension = latent_dimension
		self.d = context_dimension + latent_dimension
		self.W = W
		
		self.USERS = FactorUCBUserStruct(context_dimension, latent_dimension, lambda_ , n, W, init)
		self.articles = []
		for i in range(itemNum):
			# print (i)
			self.articles.append(FactorUCBArticleStruct(i, context_dimension, latent_dimension, lambda_, W, init)) 

		self.alpha = alpha
		self.alpha2 = alpha2

		if window_size == -1:
			self.increase_window = True
			self.window_size = 1
		else:
			self.increase_window = False
			self.window_size = window_size
		self.max_window_size = max_window_size
		self.window = []
		self.time = 0
		self.CanEstimateUserPreference = False
		self.CanEstimateCoUserPreference = True 
		self.CanEstimateW = False
		self.CanEstimateV = True
	def decide(self, pool_articles, userID):
		maxPTA = float('-inf')
		articlePicked = None

		for x in pool_articles:
			self.articles[x.id].V[:self.context_dimension] = x.contextFeatureVector[:self.context_dimension]
			x_pta = self.USERS.getProb(self.alpha, self.alpha2, self.articles[x.id], userID)

			# pick article with highest Prob
			# print x_pta 
			if maxPTA < x_pta:
				articlePicked = x
				maxPTA = x_pta
				
		return articlePicked

	def getProb(self, pool_articles, userID):
		means = []
		vars = []
		for x in pool_articles:
			self.articles[x.id].V[:self.context_dimension] = x.contextFeatureVector[:self.context_dimension]
			x_pta, mean, var = self.USERS.getProb_plot(self.alpha, self.alpha2, self.articles[x.id], userID)
			means.append(mean)
			vars.append(var)
		return means, vars

	def updateParameters(self, articlePicked, click, userID):
		self.time += 1
		self.window.append((articlePicked, click, userID))
		if len(self.window)%self.window_size == 0:
			articles = []
			clicks = []
			for articlePicked, click, userID in self.window:
				articles.append(self.articles[articlePicked.id])
				clicks.append(click)
			self.USERS.updateParameters(articles, clicks, userID)
			for articlePicked, click, userID in self.window:
				self.articles[articlePicked.id].updateParameters(self.USERS, click, userID)
			# for articlePicked, click, userID in self.window:
			# 	article = self.articles[articlePicked.id]

			# 	#self.articles[articlePicked.id].A2 -= (article.getCount(userID))*np.outer(user.U[self.context_dimension:], user.U[self.context_dimension:])
			# 	self.USERS.updateParameters(self.articles[articlePicked.id], click, userID)
			
			# for articlePicked, click, userID in self.window:
			# 	#self.articles[articlePicked.id].A2 += (article.getCount(userID)-1)*np.outer(user.U[self.context_dimension:], user.U[self.context_dimension:])

			# 	# self.users[userID].A -= (user.getCount(articlePicked.id))*np.outer(article.V, article.V)
			# 	self.articles[articlePicked.id].updateParameters(self.USERS, click, userID)
			# 	article = self.articles[articlePicked.id]
			# 	# self.users[userID].A += (user.getCount(articlePicked.id)-1)*np.outer(article.V, article.V)
			self.window = []
			if self.increase_window == True:
				self.window_size = min(self.window_size+1, self.max_window_size)
	def increaseWindowSize(self):
		self.window_size = min(self.window_size+1, self.max_window_size)
	def getCoTheta(self, userID):
		return self.USERS.CoTheta.T[userID]
	def getTheta(self, userID):
		return self.USERS.UserTheta.T[userID]
	def getV(self, articleID):
		return self.articles[articleID].V