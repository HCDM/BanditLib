import numpy as np
import scipy.stats

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
class UCBPMFAlgorithm:
	def __init__(self, dimension,  n, itemNum, sigma, sigmaU, sigmaV,alpha=0.0):  # n is number of users
		self.alpha = alpha
		self.sigma = sigma
		self.dimension = dimension
		self.users = []
		for i in range(n):
			self.users.append(UCBPMFUserStruct(i, dimension, sigma, sigmaU,))
		self.articles = []
		for i in range(itemNum):
			self.articles.append(UCBPMFArticleStruct(i, dimension, sigma, sigmaV,))
		self.time = 0

		self.CanEstimateUserPreference = False
		self.CanEstimateCoUserPreference = False
		self.CanEstimateW = False
		self.CanEstimateV = False
	def decide(self, pool_articles, userID):

		maxPTA = float('-inf')
		articlePicked = None

		for x in pool_articles:
			x_pta = self.users[userID].getProb(self.alpha, self.articles[x.id])
			# pick article with highest Prob
			# print x_pta 
			if maxPTA < x_pta:
				articlePicked = x
				maxPTA = x_pta				
		return articlePicked

	def updateParameters(self, articlePicked, click, userID):
		self.time += 1
		for i in range(10):
			self.users[userID].updateParameters(self.articles[articlePicked.id], click)
			self.articles[articlePicked.id].updateParameters(self.users[userID], click)

	def getCoTheta(self, userID):
		return self.users[userID].U
