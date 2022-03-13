from random import choice, random, sample
import numpy as np
import networkx as nx
from BanditAlg.BanditAlgorithms import ArmBaseStruct
import datetime

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
		
	def updateParameters(self, updated_A, updated_b):
		self.A += updated_A
		self.b += updated_b
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
	def __init__(self, parameter, node_list, seed_size, oracle, dimension, alpha,  lambda_ , feedback = 'edge'):
		self.param = parameter
		self.node_list = node_list
		self.oracle = oracle
		self.seed_size = seed_size
		self.dimension = dimension
		self.alpha = alpha
		self.lambda_ = lambda_
		self.feedback = feedback

		self.users = []  #Nodes
		self.Theta = np.zeros((len(node_list), dimension))
		for idx, u in enumerate(self.node_list):
			self.users.append(LinUCBUserStruct(dimension, lambda_ , u))
			self.Theta[idx, :] = self.users[-1].UserTheta

	def decide(self):
		n = len(self.node_list)
		MG = np.zeros((n, 2))
		MG[:, 0] = np.arange(n)
		influence_UCB = np.matmul(self.Theta, self.param[:, :self.dimension].T)
		np.fill_diagonal(influence_UCB, 1)
		np.clip(influence_UCB, 0, 1)
		MG[:, 1] = np.sum(influence_UCB, axis=1)
		# print('initialize time', datetime.datetime.now() - startTime)
		S = []
		args = []
		temp =  np.zeros(n)
		prev_spread = 0

		for k in range(self.seed_size):
			MG = MG[MG[:,1].argsort()]
			
			for i in range(0, n-k-1):
				iStartTime = datetime.datetime.now()
				select_node = int(MG[-1, 0])
				MG[-1, 1] = np.sum(np.maximum(influence_UCB[select_node, :], temp)) - prev_spread
				if MG[-1, 1] >= MG[-2, 1]:
					prev_spread = prev_spread + MG[-1, 1]
					break
				else:
					val = MG[-1, 1]
					idx = np.searchsorted(MG[:, 1], val)
					MG_new = np.zeros(MG.shape)
					MG_new[:idx, :] = MG[:idx, :]
					MG_new[idx, :] = MG[-1, :]
					MG_new[idx+1:	, :] = MG[idx:-1, :]
					MG = MG_new
			args.append(int(MG[-1, 0]))
			S.append(self.node_list[int(MG[-1, 0])])
			temp = np.amax(influence_UCB[np.array(args), :], axis=0)
			MG[-1, 1] = -1

		return S

	def updateParameters(self, S, live_nodes, live_edges, _iter):
		A_item = np.array([self.node_list.index(x) for x in self.node_list if x not in S])
		b_item = np.array([self.node_list.index(x) for x in live_nodes if x not in S])
		update_A = self.param[A_item, :self.dimension]
		add_A = np.sum(np.matmul(update_A[:, :, np.newaxis], update_A[:, np.newaxis,:]), axis=0)
		add_b = np.sum(self.param[b_item, :self.dimension], axis=0)
		for u in S:
			u_idx = self.node_list.index(u)
			self.users[u_idx].updateParameters(add_A, add_b)
			self.Theta[u_idx, :] = self.users[u_idx].UserTheta

	def getCoTheta(self, userID):
		return self.users[userID].UserTheta
	def getP(self):
		return self.currentP		

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