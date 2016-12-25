import numpy as np

class HLinUCBArticleStruct:
	def __init__(self, id, context_dimension, latent_dimension, lambda_, init="zero", context_feature=None):
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
			# self.V = np.random.normal(0,0.2,self.d)
			# self.V = np.random.normal(0,0.5,self.d)
			# self.V = np.random.normal(0,1,self.d)
			# self.V = np.random.normal(0,2,self.d)
		else:
			self.V = np.zeros(self.d)

	def updateParameters(self, user, click):
		self.time += 1
		if user.id in self.count:
			self.count[user.id] += 1
		else:
			self.count[user.id] = 1

		self.A2 += np.outer(user.U[self.context_dimension:], user.U[self.context_dimension:])
		self.b2 += user.U[self.context_dimension:]*(click - user.U[:self.context_dimension].dot(self.V[:self.context_dimension]))
		self.A2Inv  = np.linalg.inv(self.A2)

		self.V[self.context_dimension:] = np.dot(self.A2Inv, self.b2)

	def getCount(self, user_id):
		if user_id in self.count:
			return self.count[user_id]
		else:
			return 0
class HLinUCBUserStruct:
	def __init__(self, id, context_dimension, latent_dimension, lambda_, init="zero"):
		self.id = id
		self.context_dimension = context_dimension
		self.latent_dimension = latent_dimension
		self.d = context_dimension+latent_dimension

		self.A = lambda_*np.identity(n = self.d)
		self.b = np.zeros(self.d)
		self.AInv = np.linalg.inv(self.A)

		self.count = {}
		self.time = 0
		if (init=="random"):
			self.U = np.random.rand(self.d)
		else:
			self.U = np.zeros(self.d)
		# self.U = np.zeros(self.d)
	def updateParameters(self, article, click):
		self.time += 1
		if article.id in self.count:
			self.count[article.id] += 1
		else:
			self.count[article.id] = 1

		self.A += np.outer(article.V,article.V)
		self.b += article.V*click
		self.AInv = np.linalg.inv(self.A)				

		self.U = np.dot(self.AInv, self.b)		
	def getTheta(self):
		return self.U
	
	def getA(self):
		return self.A

	def getProb(self, alpha, alpha2, article):
		if alpha == -1:
			alpha = 0.1*np.sqrt(np.log(self.time+1))+0.1*(1-0.8**self.time)
			alpha2 = 0.1*np.sqrt(np.log(article.time+1))+0.1*(1-0.8**article.time)
		mean = np.dot(self.U, article.V)
		var = np.sqrt(np.dot(np.dot(article.V, self.AInv),  article.V))
		var2 = np.sqrt(np.dot(np.dot(self.U[self.context_dimension:], article.A2Inv),  self.U[self.context_dimension:]))
		pta = mean + alpha * var + alpha2*var2
		return pta
	def getProb_plot(self, alpha, alpha2, article):
		mean = np.dot(self.U, article.V)
		var = np.sqrt(np.dot(np.dot(article.V, self.AInv),  article.V))
		var2 = np.sqrt(np.dot(np.dot(self.U[self.context_dimension:], article.A2Inv),  self.U[self.context_dimension:]))
		pta = mean + alpha * var + alpha*var2
		return pta, mean, alpha*var

	def getCount(self, article_id):
		if article_id in self.count:
			return self.count[article_id]
		else:
			return 0

class HLinUCBAlgorithm:
	def __init__(self, context_dimension, latent_dimension, alpha, alpha2, lambda_, n, itemNum, init="zero", window_size = 1, max_window_size = 50):  # n is number of users

		self.context_dimension = context_dimension
		self.latent_dimension = latent_dimension
		self.d = context_dimension + latent_dimension
		
		self.users = []
		#algorithm have n users, each user has a user structure
		for i in range(n):
			self.users.append(HLinUCBUserStruct(i, context_dimension, latent_dimension, lambda_ , init)) 
		self.articles = []
		for i in range(itemNum):
			self.articles.append(HLinUCBArticleStruct(i, context_dimension, latent_dimension, lambda_ , init)) 

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
			x_pta = self.users[userID].getProb(self.alpha, self.alpha2, self.articles[x.id])

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
			x_pta, mean, var = self.users[userID].getProb_plot(self.alpha, self.alpha2, self.articles[x.id])
			means.append(mean)
			vars.append(var)
		return means, vars

	def updateParameters(self, articlePicked, click, userID):
		self.time += 1
		self.window.append((articlePicked, click, userID))
		if len(self.window)%self.window_size == 0:
			for articlePicked, click, userID in self.window:
				article = self.articles[articlePicked.id]
				user = self.users[userID]

				#self.articles[articlePicked.id].A2 -= (article.getCount(userID))*np.outer(user.U[self.context_dimension:], user.U[self.context_dimension:])
				self.users[userID].updateParameters(self.articles[articlePicked.id], click)
			
			for articlePicked, click, userID in self.window:
				user = self.users[userID]
				#self.articles[articlePicked.id].A2 += (article.getCount(userID)-1)*np.outer(user.U[self.context_dimension:], user.U[self.context_dimension:])

				# self.users[userID].A -= (user.getCount(articlePicked.id))*np.outer(article.V, article.V)
				self.articles[articlePicked.id].updateParameters(self.users[userID], click)
				article = self.articles[articlePicked.id]
				# self.users[userID].A += (user.getCount(articlePicked.id)-1)*np.outer(article.V, article.V)
			self.window = []
			if self.increase_window == True:
				self.window_size = min(self.window_size+1, self.max_window_size)
	def getCoTheta(self, userID):
		return self.users[userID].U
	def getV(self, articleID):
		return self.articles[articleID].V