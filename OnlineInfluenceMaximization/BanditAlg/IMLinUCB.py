from random import choice, random, sample
import numpy as np
import networkx as nx
from BanditAlg.BanditAlgorithms import ArmBaseStruct

class LinUCBUserStruct:
	def __init__(self, featureDimension,lambda_, userID, RankoneInverse = False):
		self.userID = userID
		self.d = featureDimension
		self.A = lambda_*np.identity(n = self.d)
		self.b = np.zeros(self.d)
		self.AInv = np.linalg.inv(self.A)
		self.UserTheta = np.zeros(self.d)

		self.RankoneInverse = RankoneInverse

		self.pta_max = 1
		
	def updateParameters(self, articlePicked_FeatureVector, click):
		self.A += np.outer(articlePicked_FeatureVector,articlePicked_FeatureVector)
		self.b += articlePicked_FeatureVector*click
		if self.RankoneInverse:
			temp = np.dot(self.AInv, articlePicked_FeatureVector)
			self.AInv = self.AInv - (np.outer(temp,temp))/(1.0+np.dot(np.transpose(articlePicked_FeatureVector),temp))
		else:
			self.AInv =  np.linalg.inv(self.A)

		self.UserTheta = np.dot(self.AInv, self.b)
		
	def getTheta(self):
		return self.UserTheta
	
	def getA(self):
		return self.A

	def getProb(self, alpha, article_FeatureVector):
		mean = np.dot(self.UserTheta,  article_FeatureVector)
		var = np.sqrt(np.dot(np.dot(article_FeatureVector, self.AInv),  article_FeatureVector))
		pta = mean + alpha * var
		if pta > self.pta_max:
			pta = self.pta_max
		#print self.UserTheta
		#print article_FeatureVector
		#print pta, mean, alpha*var
		# if mean >0:
		# 	print 'largerthan0', mean
		return pta

class N_LinUCBAlgorithm:
	def __init__(self, G, P, parameter, seed_size, oracle, dimension, alpha,  lambda_ , FeatureDic, FeatureScaling, feedback = 'edge'):
		self.G = G
		self.trueP = P
		self.parameter = parameter		
		self.oracle = oracle
		self.seed_size = seed_size
		self.dimension = dimension
		self.alpha = alpha
		self.lambda_ = lambda_
		self.FeatureDic = FeatureDic
		self.FeatureScaling = FeatureScaling
		self.feedback = feedback

		self.currentP =nx.DiGraph()
		self.users = {}  #Nodes
		self.arms = {}
		for u in self.G.nodes():
			self.users[u] = LinUCBUserStruct(dimension, lambda_ , u)
			for v in self.G[u]:
				self.arms[(u,v)] = ArmBaseStruct((u,v))
				self.currentP.add_edge(u,v, weight=random())
		self.list_loss = []

	def decide(self):
		S = self.oracle(self.G, self.seed_size, self.currentP)
		return S

	def updateParameters(self, S, live_nodes, live_edges, _iter):
		count = 0
		loss_p = 0 
		loss_out = 0
		loss_in = 0
		for u in live_nodes:
			for (u, v) in self.G.edges(u):
				featureVector = self.FeatureScaling*self.FeatureDic[(u,v)]
				if (u,v) in live_edges:
					reward = live_edges[(u,v)]
				else:
					reward = 0
				self.arms[(u, v)].updateParameters(reward=reward)
				# reward = self.arms[(u, v)].averageReward    #####Average Reward
				self.users[u].updateParameters(featureVector, reward)
				self.currentP[u][v]['weight']  = self.users[v].getProb(self.alpha, featureVector)

				estimateP = self.currentP[u][v]['weight']
				trueP = self.trueP[u][v]['weight']
				loss_p += np.abs(estimateP-trueP)
				count += 1
		self.list_loss.append([loss_p/count])
	def getCoTheta(self, userID):
		return self.users[userID].UserTheta
	def getP(self):
		return self.currentP	
	def getLoss(self):
		return np.asarray(self.list_loss)	

class LinUCBAlgorithm:
	def __init__(self, G, seed_size, oracle, dimension, alpha,  lambda_ , FeatureDic, feedback = 'edge'):
		self.G = G
		self.oracle = oracle
		self.seed_size = seed_size

		self.dimension = dimension
		self.alpha = alpha
		self.lambda_ = lambda_
		self.FeatureDic = FeatureDic
		self.feedback = feedback

		self.currentP =nx.DiGraph()
		self.USER = LinUCBUserStruct(dimension, lambda_ , 0)
		for u in self.G.nodes():
			for v in self.G[u]:
				self.currentP.add_edge(u,v, weight=0)

	def decide(self):
		S = self.oracle(self.G, self.seed_size, self.currentP)
		return S

	def updateParameters(self, S, live_nodes, live_edges):
		for u in S:
			for (u, v) in self.G.edges(u):
				featureVector = self.FeatureDic[(u,v)]
				if (u,v) in live_edges:
					reward = live_edges[(u,v)]
				else:
					reward = 0
				self.USER.updateParameters(featureVector, reward)
				self.currentP[u][v]['weight']  = self.USER.getProb(self.alpha, featureVector)
	def getCoTheta(self, userID):
		return self.USER.UserTheta
	def getP(self):
		return self.currentP