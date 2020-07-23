import numpy as np
import scipy.stats
from BaseAlg import BaseAlg


class UCBPMFArticleStruct:
	def __init__(self, id, dimension, sigma, sigmaV, init="zero"):
		self.id = id
		self.dimension = dimension
		self.sigma = sigma
		self.sigmaV = sigmaV

		self.A2 = 1.0/(self.sigmaV**2)*np.identity(n = self.dimension)
		self.b2 = np.zeros(self.dimension)
		self.A2Inv = np.linalg.inv(self.A2)
		self.Mu = 1.0/(self.sigma**2)*self.A2Inv.dot(self.b2)

		self.count = {}

		self.V = np.random.multivariate_normal(self.Mu, self.A2Inv)
	def updateParameters(self, user, click):
		if user.id in self.count:
			self.count[user.id] += 1
		else:
			self.count[user.id] = 1

		self.A2 += 1.0/(self.sigma**2)*np.outer(user.U, user.U)
		self.b2 += user.U*click
		self.A2Inv  = np.linalg.inv(self.A2)

		self.Mu = 1.0/(self.sigma**2)*self.A2Inv.dot(self.b2)
		#Sample V
		self.V = np.random.multivariate_normal(self.Mu, self.A2Inv)

	def getCount(self, user_id):
		if user_id in self.count:
			return self.count[user_id]
		else:
			return 0

class UCBPMFUserStruct:
	def __init__(self, id, dimension, sigma, sigmaU, init="zero"):
		self.id = id
		self.dimension = dimension
		self.sigma = sigma
		self.sigmaU = sigmaU

		self.A = 1.0/(self.sigmaU**2)*np.identity(n = self.dimension)
		self.b = np.zeros(self.dimension)
		self.AInv = np.linalg.inv(self.A)
		self.Mu = 1.0/(self.sigma**2)*np.dot(self.AInv, self.b)	

		self.count = {}

		self.U = np.random.multivariate_normal(self.Mu, self.AInv)


	def updateParameters(self, article, click):
		if article.id in self.count:
			self.count[article.id] += 1
		else:
			self.count[article.id] = 1

		self.A += 1.0/(self.sigma**2)*np.outer(article.V,article.V)
		self.b += article.V*click
		self.AInv = np.linalg.inv(self.A)				

		self.Mu = 1.0/(self.sigma**2)*np.dot(self.AInv, self.b)	
		#Sample U
		self.U = np.random.multivariate_normal(self.Mu, self.AInv)
	
	def getProb(self, alpha, article):
		mean = np.dot(self.U, article.V)
		var = np.sqrt(np.trace(self.AInv.T.dot(article.A2Inv) + self.AInv.T.dot(np.outer(article.V, article.V)) + np.outer(self.U, self.U).dot(article.A2Inv)))
		pta = mean + alpha * var
		return pta

class UCBPMFAlgorithm(BaseAlg):
	def __init__(self, arg_dict):  # n is number of users
		BaseAlg.__init__(self, arg_dict)
		self.users = []
		for i in range(self.n_users):
			self.users.append(UCBPMFUserStruct(i, self.dimension, self.sigma, self.sigmaU,))
		self.articles = []
		for i in range(self.itemNum):
			self.articles.append(UCBPMFArticleStruct(i, self.dimension, self.sigma, self.sigmaV,))
		self.time = 0

	def decide(self, pool_articles, userID, k = 1):
		articles = []
		for i in range(k):
			maxPTA = float('-inf')
			articlePicked = None

			for x in pool_articles:
				x_pta = self.users[userID].getProb(self.alpha, self.articles[x.id])
				# pick article with highest Prob
				if maxPTA < x_pta and x not in articles:
					articlePicked = x
					maxPTA = x_pta
			articles.append(articlePicked)
		return articles

	def updateParameters(self, articlePicked, click, userID):
		self.time += 1
		for i in range(10):
			self.users[userID].updateParameters(self.articles[articlePicked.id], click)
			self.articles[articlePicked.id].updateParameters(self.users[userID], click)

	def getCoTheta(self, userID):
		return self.users[userID].U
