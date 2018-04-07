import numpy as np
from LinUCB import *
import math
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from BaseAlg import BaseAlg

class CLUBUserStruct(LinUCBUserStruct):
	def __init__(self,featureDimension,  lambda_, userID):
		LinUCBUserStruct.__init__(self,featureDimension = featureDimension, lambda_= lambda_)
		self.reward = 0
		self.CA = self.A
		self.Cb = self.b
		self.CAInv = np.linalg.inv(self.CA)
		self.CTheta = np.dot(self.CAInv, self.Cb)
		self.I = lambda_*np.identity(n = featureDimension)	
		self.counter = 0
		self.CBPrime = 0
		self.d = featureDimension
	def updateParameters(self, articlePicked_FeatureVector, click,alpha_2):
		#LinUCBUserStruct.updateParameters(self, articlePicked_FeatureVector, click)
		#alpha_2 = 1
		self.A += np.outer(articlePicked_FeatureVector,articlePicked_FeatureVector)
		self.b += articlePicked_FeatureVector*click
		self.AInv = np.linalg.inv(self.A)
		self.UserTheta = np.dot(self.AInv, self.b)
		self.counter+=1
		self.CBPrime = alpha_2*np.sqrt(float(1+math.log10(1+self.counter))/float(1+self.counter))

	def updateParametersofClusters(self,clusters,userID,Graph,users):
		self.CA = self.I
		self.Cb = np.zeros(self.d)
		#print type(clusters)

		for i in range(len(clusters)):
			if clusters[i] == clusters[userID]:
				self.CA += float(Graph[userID,i])*(users[i].A - self.I)
				self.Cb += float(Graph[userID,i])*users[i].b
		self.CAInv = np.linalg.inv(self.CA)
		self.CTheta = np.dot(self.CAInv,self.Cb)

	def getProb(self, alpha, article_FeatureVector,time):
		mean = np.dot(self.CTheta, article_FeatureVector)
		var = np.sqrt(np.dot(np.dot(article_FeatureVector, self.CAInv),  article_FeatureVector))
		pta = mean +  alpha * var*np.sqrt(math.log10(time+1))
		return pta

class CLUBAlgorithm(BaseAlg):
	def __init__(self, arg_dict):
		BaseAlg.__init__(self, arg_dict)
		self.time = 0
		#N_LinUCBAlgorithm.__init__(dimension = dimension, alpha=alpha,lambda_ = lambda_,n=n)
		self.users = []
		#algorithm have n users, each user has a user structure
		for i in range(self.n):
			self.users.append(CLUBUserStruct(self.dimension,self.lambda_, i)) 
		if (self.cluster_init=="Erdos-Renyi"):
			p = 3*math.log(self.n)/self.n
			self.Graph = np.random.choice([0, 1], size=(self.n,self.n), p=[1-p, p])
			self.clusters = []
			g = csr_matrix(self.Graph)
			N_components, components = connected_components(g)
		else:
			self.Graph = np.ones([self.n,self.n]) 
			self.clusters = []
			g = csr_matrix(self.Graph)
			N_components, components = connected_components(g)

			
	def decide(self,pool_articles,userID, k = 1):
		self.users[userID].updateParametersofClusters(self.clusters,userID,self.Graph, self.users)
		articles = []
		for i in range(k):
			maxPTA = float('-inf')
			articlePicked = None

			for x in pool_articles:
				x_pta = self.users[userID].getProb(self.alpha, x.contextFeatureVector[:self.dimension],self.time)
				# pick article with highest Prob
				if maxPTA < x_pta and x not in articles:
					articlePicked = x.id
					featureVectorPicked = x.contextFeatureVector[:self.dimension]
					picked = x
					maxPTA = x_pta
			articles.append(picked)
		self.time +=1

		return articles
	def updateParameters(self, articlePicked, click,userID):
		self.users[userID].updateParameters(articlePicked.contextFeatureVector[:self.dimension], click, self.alpha_2)
	def updateGraphClusters(self,userID, binaryRatio):
		n = len(self.users)
		for j in range(n):
			ratio = float(np.linalg.norm(self.users[userID].UserTheta - self.users[j].UserTheta,2))/float(self.users[userID].CBPrime + self.users[j].CBPrime)
			#print float(np.linalg.norm(self.users[userID].UserTheta - self.users[j].UserTheta,2)),'R', ratio
			if ratio > 1:
				ratio = 0
			elif binaryRatio == 'True':
				ratio = 1
			elif binaryRatio == 'False':
				ratio = 1.0/math.exp(ratio)
			#print 'ratio',ratio
			self.Graph[userID][j] = ratio
			self.Graph[j][userID] = self.Graph[userID][j]
		N_components, component_list = connected_components(csr_matrix(self.Graph))
		#print 'N_components:',N_components
		self.clusters = component_list
		return N_components
	def getLearntParameters(self, userID):
		return self.users[userID].UserTheta





