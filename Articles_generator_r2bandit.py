import cPickle
import numpy as np 
from util_functions import featureUniform, gaussianFeature, fileOverWriteWarning
from random import sample, randint, uniform
import json
import time
from random import *

class Article():	
	def __init__(self, aid, atype, FV=None):
		self.id = aid
		self.type = atype
		self.featureVector = FV
		

class ArticleManager():
	def __init__(self, dimension, n_articles, ArticleGroups, FeatureFunc, argv, userFeature_theta, userFeature_beta ):
		self.signature = "Article manager for simulation study"
		self.dimension = dimension
		self.n_articles = n_articles
		self.ArticleGroups = ArticleGroups
		self.FeatureFunc = FeatureFunc
		self.thetaFunc = FeatureFunc
		self.betaFunc = FeatureFunc
		self.argv = argv

		self.signature = "A-"+str(self.n_articles)+"+AG"+ str(self.ArticleGroups)+"+TF-"+self.FeatureFunc.__name__

		self.userFeature_theta = userFeature_theta
		self.userFeature_beta = userFeature_beta
	def saveArticles(self, Articles, filename, force = False):
		with open(filename, 'w') as f:
			for i in range(len(Articles)):
				f.write(json.dumps((Articles[i].id, Articles[i].type, Articles[i].featureVector)) + '\n')


	def loadArticles(self, filename):
		articles = []
		with open(filename, 'r') as f:
			for line in f:
				aid, atype, featureVector = json.loads(line)
				articles.append(Article(aid, atype,  np.array(featureVector)))
		return articles

	#automatically generate masks for articles, but it may generate same masks
	def generateMasks(self):
		mask = {}
		for i in range(self.ArticleGroups):
			mask[i] = np.random.randint(2, size = self.dimension)
		return mask

	def simulateArticlePool(self):
		articles = []
		
		articles_id = {}
		mask = self.generateMasks()

		for i in range(self.ArticleGroups):
			articles_id[i] = range((self.n_articles*i)/self.ArticleGroups, (self.n_articles*(i+1))/self.ArticleGroups)

			for key in articles_id[i]:
				featureVector = np.multiply(featureUniform(self.dimension, {}), mask[i])
				l2_norm = np.linalg.norm(featureVector, ord =2)
				articles.append(Article(key, featureVector/l2_norm ))

	
		return articles


	def small_small_Exp(self, theta, beta, small_bound):
		vector = np.array([2*np.random.uniform(-1,1) for _ in range(self.dimension)])
		#vector = self.FeatureFunc(self.dimension, argv = self.argv)
		vector_l2_norm = np.linalg.norm(vector, ord =2)
		final_vector_norm = np.asarray(vector)/float(vector_l2_norm)
		while np.dot(vector, theta) > small_bound or np.dot(vector, beta) > small_bound or np.linalg.norm(vector, ord =2) >10:
			#print vector, np.dot(vector, theta), np.dot(vector, beta)
			vector = np.array([2*np.random.uniform(-1,1) for _ in range(self.dimension)])
			vector_l2_norm = np.linalg.norm(vector, ord =2)

		vector_l2_norm = np.linalg.norm(vector, ord =2)
		final_vector_norm = np.asarray(vector)/float(vector_l2_norm)
			
		print 'small, small', np.exp(np.dot(vector, theta)), np.exp(np.dot(vector, beta))
		return vector

	def large_large_Exp(self, theta, beta, large_bound):
		vector = np.array([2*np.random.uniform(-1,1) for _ in range(self.dimension)])
		vector_l2_norm = np.linalg.norm(vector, ord =2)
		final_vector_norm = np.asarray(vector)/float(vector_l2_norm)
		while np.dot(vector, theta) < large_bound or np.dot(vector, beta) < large_bound or np.linalg.norm(vector, ord =2) > 10:
			vector = np.array([2*np.random.uniform(-1,1) for _ in range(self.dimension)])
			vector_l2_norm = np.linalg.norm(vector, ord =2)
		vector_l2_norm = np.linalg.norm(vector, ord =2)
		final_vector_norm = np.asarray(vector)/float(vector_l2_norm)
			
		print 'lagrge, large', np.exp(np.dot(final_vector_norm, theta)), np.exp(np.dot(final_vector_norm, beta))
		return vector

	def small_large_Exp(self, theta, beta, small_bound, large_bound):
		vector = np.array([2*np.random.uniform(-1,1) for _ in range(self.dimension)])
		vector_l2_norm = np.linalg.norm(vector, ord =2)
		final_vector_norm = np.asarray(vector)/float(vector_l2_norm)
		while np.dot(vector, theta) > large_bound or np.dot(vector, beta) < large_bound or np.linalg.norm(vector, ord =2) >10:
			#print vector, np.dot(vector, theta), np.dot(vector, beta)
			vector = np.array([2*np.random.uniform(-1,1) for _ in range(self.dimension)])
			vector_l2_norm = np.linalg.norm(vector, ord =2)
		vector_l2_norm = np.linalg.norm(vector, ord =2)
		final_vector_norm = np.asarray(vector)/float(vector_l2_norm)
			
		print 'small, large', np.exp(np.dot(final_vector_norm, theta)), np.exp(np.dot(final_vector_norm, beta))
		return vector

	def large_small_Exp(self, theta, beta, small_bound, large_bound):
		vector = np.array([2*np.random.uniform(-1,1) for _ in range(self.dimension)])
		vector_l2_norm = np.linalg.norm(vector, ord =2)
		final_vector_norm = np.asarray(vector)/float(vector_l2_norm)
		while np.dot(vector, theta) < large_bound or np.dot(vector, beta) > large_bound or vector_l2_norm >10:
			vector = np.array([2*np.random.uniform(-1,1) for _ in range(self.dimension)])
			vector_l2_norm = np.linalg.norm(vector, ord =2)
		vector_l2_norm = np.linalg.norm(vector, ord =2)
		final_vector_norm = np.asarray(vector)/float(vector_l2_norm)
			
		print 'lagrge, small', np.exp(np.dot(final_vector_norm, theta)), np.exp(np.dot(final_vector_norm, beta))
		return vector

	def simulateArticlePool_2SetOfFeature(self):
		articlesDic = {}		
		articles_id = {}
		centroids = [0.2, 0.9]
		
		articlesDic['small_small'] = []
		articlesDic['small_large'] = []
		articlesDic['large_small'] = []
		articlesDic['large_large'] = []
		small_bound = -0.5
		large_bound = 1
		# small_bound_click  = -1   #coressponds to 1/(1+e), which is smaller than 1/2
		# large_bound_click = 1     #coressponds to 1/(1+e^{-1}), which is larger than 1/2

		# small_bound_return = -1  # lambda = -1 coressponds to lambda = e^{-1}, expected return time t = e
		# large_bound_return = 1   # lambda = -1 coressponds to lambda = e^{-1}, expected return time t = 1/e
		print 'generating articles ...'
		for i in range(self.n_articles):
			small_theta_small_beta = list(self.small_small_Exp(self.userFeature_theta, self.userFeature_beta, small_bound))
			small_theta_large_beta = list(self.small_large_Exp(self.userFeature_theta, self.userFeature_beta, small_bound, large_bound))
			large_theta_small_beta = list(self.large_small_Exp(self.userFeature_theta, self.userFeature_beta, small_bound, large_bound))
			large_theta_large_beta = list(self.large_large_Exp(self.userFeature_theta, self.userFeature_beta, large_bound))

			articlesDic['small_small'].append(Article(4*i, 'smallTheta_smallBeta', small_theta_small_beta))
			articlesDic['small_large'].append(Article(4*i+1, 'smallTheta_largeBeta',small_theta_large_beta))
			articlesDic['large_small'].append(Article(4*i+2, 'largeTheta_smallBeta', large_theta_small_beta))
			articlesDic['large_large'].append(Article(4*i+3, 'largeTheta_largeBeta', large_theta_large_beta))
		print 'finish generating articles!'

		return articlesDic['small_small'], 	articlesDic['small_large'], articlesDic['large_small'], articlesDic['large_large']



