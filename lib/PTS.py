import numpy as np
import scipy.stats
from BaseAlg import BaseAlg

class PTSArticleStruct:
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
class PTSUserStruct:
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

class PTSParticleStruct:
	def __init__(self, dimension, n, itemNum, sigma, sigmaU, sigmaV, weight, ptsb = False):
		self.sigma = sigma
		self.weight = weight
		self.users = []
		for i in range(n):
			self.users.append(PTSUserStruct(i, dimension, sigma, sigmaU,))
		self.articles = []
		for i in range(itemNum):
			self.articles.append(PTSArticleStruct(i, dimension, sigma, sigmaV,))

class PTSAlgorithm(BaseAlg):
	def __init__(self, arg_dict):  # n is number of users
		BaseAlg.__init__(self, arg_dict)
		self.particles = [] # Particles
		for i in range(self.particle_num):
			self.particles.append(PTSParticleStruct(self.dimension, self.n, self.itemNum, self.sigma, self.sigmaU, self.sigmaV, 1.0/self.particle_num))
		self.time = 0

	def decide(self, pool_articles, userID, k = 1):

		#Sample a Particle
		d = np.random.choice(self.particle_num, p = [p.weight for p in self.particles])
		p = self.particles[d]
		#For PTS-B
		articles = []
		for i in range(k):
			maxPTA = float('-inf')
			articlePicked = None

			for x in pool_articles:
				x_pta = p.users[userID].U.dot(p.articles[x.id].V)
				# pick article with highest Prob
				if maxPTA < x_pta and x not in articles:
					articlePicked = x
					maxPTA = x_pta
			articles.append(articlePicked)
		return articles

	def updateParameters(self, articlePicked, click, userID):
		self.time += 1
		#Reweighting
		weights = []
		for d in range(self.particle_num):
			article = self.particles[d].articles[articlePicked.id]
			mean = article.V.dot(self.particles[d].users[userID].Mu)
			var = 1.0/(self.particles[d].sigma**2)+article.V.dot(article.A2Inv).dot(article.V)
			weights.append(scipy.stats.norm(mean, var).pdf(click))
		weights = np.array(weights)/sum(weights) # sum to 1

		# print self.time, weights
		#Resampling
		self.particles = [self.particles[i] for i in np.random.choice(self.particle_num, size = self.particle_num, p = weights)]
		for d in range(self.particle_num):
			self.particles[d].weight = 1.0/self.particle_num
		for d in range(self.particle_num):
			self.particles[d].users[userID].updateParameters(self.particles[d].articles[articlePicked.id], click)
			self.particles[d].articles[articlePicked.id].updateParameters(self.particles[d].users[userID], click)

	def getCoTheta(self, userID):
		#Sample a Particle
		d = np.random.choice(self.particle_num, p = [p.weight for p in self.particles])
		p = self.particles[d]
		return p.users[userID].U
