import copy
import numpy as np
#from random import sample, shuffle
from scipy.sparse import csgraph
import datetime #
import os.path #
#import matplotlib.pyplot as plt
import argparse
import yaml
from sklearn.decomposition import TruncatedSVD
from sklearn import cluster
from sklearn.decomposition import PCA
# local address to save simulated users, simulated articles, and results
from conf import sim_files_folder, save_address
from util_functions import featureUniform, gaussianFeature, createLinUCBDict, createCoLinDict, createHLinUCBDict
from Articles import ArticleManager
from Users.Users import UserManager
from Users.CoUsers import CoUserManager
from RewardManager import RewardManager
from DiffList.DiffManager import DiffManager

from lib.LinUCB import N_LinUCBAlgorithm, Uniform_LinUCBAlgorithm,Hybrid_LinUCBAlgorithm
from lib.hLinUCB import HLinUCBAlgorithm
from lib.factorUCB import FactorUCBAlgorithm
from lib.CoLin import CoLinUCBAlgorithm
from lib.GOBLin import GOBLinAlgorithm
from lib.CLUB import *
from lib.PTS import PTSAlgorithm
from lib.UCBPMF import UCBPMFAlgorithm

def pca_articles(articles, order):
	X = []
	for i, article in enumerate(articles):
		X.append(article.featureVector)
	pca = PCA()
	X_new = pca.fit_transform(X)
	# X_new = np.asarray(X)
	print('pca variance in each dim:', pca.explained_variance_ratio_) 

	print X_new
	#default is descending order, where the latend features use least informative dimensions.
	if order == 'random':
		np.random.shuffle(X_new.T)
	elif order == 'ascend':
		X_new = np.fliplr(X_new)
	elif order == 'origin':
		X_new = X
	for i, article in enumerate(articles):
		articles[i].featureVector = X_new[i]
	return

def generate_algorithms(alg_dict, W):
	algorithms = {}
	diffLists = DiffManager()
	for i in alg_dict:
		print str(i)
		if i == 'linUCB':
			linUCBDict = createLinUCBDict(alg_dict[i])
			algorithms[i] = N_LinUCBAlgorithm(linUCBDict)
		elif i == 'CoLin':
			coLinDict = createCoLinDict(alg_dict[i], W)
			algorithms[i] = CoLinUCBAlgorithm(coLinDict)
		elif i == 'GOBLin':
			# uses the same arguments as colin
			GOBLinDict = createCoLinDict(alg_dict[i], W)
			algorithms[i] = GOBLinAlgorithm(GOBLinDict)
		elif i == 'HLinUCB':
			hlinUCBDict = createHLinUCBDict(alg_dict[i])
			algorithms[i] = HLinUCBAlgorithm(hlinUCBDict)
		else:
			# Do not know that algorithm name, so skip it
			continue
		diffLists.add_algorithm(i, algorithms[i].getEstimateSettings())
	return algorithms, diffLists

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = '')
	parser.add_argument('--alg', dest='alg', help='Select a specific algorithm, could be CoLin, hLinUCB, factorUCB, LinUCB, etc.')
	parser.add_argument('--contextdim', type=int, help='Set dimension of context features.')
	parser.add_argument('--hiddendim', type=int, help='Set dimension of hidden features.')
	parser.add_argument('--config', dest='config', help='yaml config file')

	args = parser.parse_args()
	cfg = {}
	with open(args.config, 'r') as ymlfile:
		cfg = yaml.load(ymlfile)
	gen = cfg['general'] if cfg.has_key('general') else {}
	user = cfg['user'] if cfg.has_key('user') else {}
	article = cfg['article'] if cfg.has_key('article') else {}
	reco = cfg['reward'] if cfg.has_key('reward') else {}

	#algName = str(args.alg) if args.alg else gen['alg']

	rewardManagerDict = {}

	if args.contextdim:
		context_dimension = args.contextdim
	else:
		context_dimension = gen['context_dimension'] if gen.has_key('context_dimension') else 20
	rewardManagerDict['context_dimension'] = context_dimension
	if args.hiddendim:
		latent_dimension = args.hiddendim
	else:
		latent_dimension = gen['hidden_dimension'] if gen.has_key('hidden_dimension') else 0
	rewardManagerDict['latent_dimension'] = latent_dimension

	rewardManagerDict['training_iterations'] = gen['training_iterations'] if gen.has_key('training_iterations') else 0
	rewardManagerDict['testing_iterations'] = gen['testing_iterations'] if gen.has_key('testing_iterations') else 100
	rewardManagerDict['plot'] = gen['plot'] if gen.has_key('plot') else True
	
	rewardManagerDict['NoiseScale'] = .01

	

	alpha  = 0.3
	lambda_ = 0.1   # Initialize A
	rewardManagerDict['epsilon'] = 0 # initialize W
	eta_ = 0.5

	n_articles = 1000
	ArticleGroups = 5

	n_users = user['number'] if user.has_key('number') else 10
	UserGroups = user['groups'] if user.has_key('groups') else 5
	
	rewardManagerDict['poolArticleSize'] = 10
	rewardManagerDict['batchSize'] = 1

	# Matrix parameters
	matrixNoise = 0.01
	rewardManagerDict['matrixNoise'] = lambda : np.random.normal(scale = matrixNoise)
	rewardManagerDict['sparseLevel'] = n_users  # if smaller or equal to 0 or larger or enqual to usernum, matrix is fully connected


	# Parameters for GOBLin
	G_alpha = alpha
	G_lambda_ = lambda_
	rewardManagerDict['Gepsilon'] = 1
	
	user['default_file'] = os.path.join(sim_files_folder, "users_"+str(n_users)+"context_"+str(context_dimension)+"latent_"+str(latent_dimension)+ "Ugroups" + str(UserGroups)+".json")
	if user.has_key('collaborative') and user['collaborative']:
		UM = CoUserManager(context_dimension+latent_dimension, user, argv={'l2_limit':1, 'sparseLevel': n_users, 'matrixNoise': rewardManagerDict['matrixNoise']})
		UM.CoTheta()
	else:
		UM = UserManager(context_dimension+latent_dimension, user, argv={'l2_limit':1})
	rewardManagerDict['W'] = UM.getW()
	rewardManagerDict['users'] = UM.getUsers()
	
	articlesFilename = os.path.join(sim_files_folder, "articles_"+str(n_articles)+"context_"+str(context_dimension)+"latent_"+str(latent_dimension)+ "Agroups" + str(ArticleGroups)+".json")
	AM = ArticleManager(context_dimension+latent_dimension, n_articles=n_articles, ArticleGroups = ArticleGroups,
			FeatureFunc=featureUniform,  argv={'l2_limit':1})
	if article.has_key('load') and article['load']:
		articles = AM.loadArticles(articles['filename']) if articles.has_key('filename') else AM.loadArticles(articlesFilename)
	else:
		articles = AM.simulateArticlePool()
		if article.has_key('save') and article['save']:
			AM.saveArticles(articles, articlesFilename, force=False) 
	rewardManagerDict['k'] = reco['k'] if reco.has_key('k') else 1
	reward_type = reco['type'] if reco.has_key('type') else 'linear'
	
	#PCA
	pca_articles(articles, 'random')
	rewardManagerDict['articles'] = articles
	rewardManagerDict['testing_method'] = "online"
	rewardManagerDict['noise'] = lambda : np.random.normal(scale = rewardManagerDict['NoiseScale'])
	rewardManagerDict['type'] = "UniformTheta"
	rewardManagerDict['simulation_signature'] = AM.signature


	
	for i in range(len(articles)):
		articles[i].contextFeatureVector = articles[i].featureVector[:context_dimension]

	# TODO: Add in reward options dictionary
	simExperiment = RewardManager(arg_dict = rewardManagerDict, reward_type = reward_type, reward_options = reco['options'])

	print "Starting for ", simExperiment.simulation_signature

	algorithms, diffLists = generate_algorithms(cfg['alg'], UM.getW())
	# algorithms = {}
	
	# if algName == 'LinUCB':
	# 	algorithms['LinUCB'] = N_LinUCBAlgorithm(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, n = n_users)
	# if algName == 'hLinUCB':
	# 	algorithms['hLinUCB'] = HLinUCBAlgorithm(context_dimension = context_dimension, latent_dimension = latent_dimension, alpha = 0.1, alpha2 = 0.1, lambda_ = lambda_, n = n_users, itemNum=n_articles, init='zero', window_size = -1)	
	# 	algorithms['LinUCB'] = N_LinUCBAlgorithm(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, n = n_users)
	# if algName == 'PTS':
	# 	algorithms['PTS'] = PTSAlgorithm(particle_num = 10, dimension = 10, n = n_users, itemNum=n_articles, sigma = np.sqrt(.5), sigmaU = 1, sigmaV = 1)
	# if algName == 'HybridLinUCB':
	# 	algorithms['HybridLinUCB'] = Hybrid_LinUCBAlgorithm(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, userFeatureList=simExperiment.reward.generateUserFeature(UM.getW()))
	# if args.alg == 'UCBPMF':
	# 	algorithms['UCBPMF'] = UCBPMFAlgorithm(dimension = 10, n = n_users, itemNum=n_articles, sigma = np.sqrt(.5), sigmaU = 1, sigmaV = 1, alpha = 0.1) 
	# if args.alg == 'factorUCB':
	# 	algorithms['FactorUCB'] = FactorUCBAlgorithm(context_dimension = context_dimension, latent_dimension = 5, alpha = 0.05, alpha2 = 0.025, lambda_ = lambda_, n = n_users, itemNum=n_articles, W = UM.getW(), init='random', window_size = -1)	
	# 	algorithms['LinUCB'] = N_LinUCBAlgorithm(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, n = n_users)
	# if args.alg == 'CoLin':
	# 	algorithms['CoLin'] = AsyCoLinUCBAlgorithm(dimension=context_dimension, alpha = alpha, lambda_ = lambda_, n = n_users, W = UM.getW())
	# 	algorithms['LinUCB'] = N_LinUCBAlgorithm(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, n = n_users)
	# if algName == 'CLUB':
	# 	algorithms['CLUB'] = CLUBAlgorithm(dimension =context_dimension,alpha = alpha, lambda_ = lambda_, n = n_users, alpha_2 = 0.5, cluster_init = 'Erdos-Renyi')	
	# if algName == 'All':
	# 	algorithms['LinUCB'] = N_LinUCBAlgorithm(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, n = n_users)
	# 	algorithms['hLinUCB'] = HLinUCBAlgorithm(context_dimension = context_dimension, latent_dimension = 5, alpha = 0.1, alpha2 = 0.1, lambda_ = lambda_, n = n_users, itemNum=n_articles, init='random', window_size = -1)	
	# 	algorithms['PTS'] = PTSAlgorithm(particle_num = 10, dimension = 10, n = n_users, itemNum=n_articles, sigma = np.sqrt(.5), sigmaU = 1, sigmaV = 1)
	# 	algorithms['HybridLinUCB'] = Hybrid_LinUCBAlgorithm(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, userFeatureList=simExperiment.reward.generateUserFeature(UM.getW()))
	# 	algorithms['UCBPMF'] = UCBPMFAlgorithm(dimension = 10, n = n_users, itemNum=n_articles, sigma = np.sqrt(.5), sigmaU = 1, sigmaV = 1, alpha = 0.1) 
	# 	algorithms['CoLin'] = AsyCoLinUCBAlgorithm(dimension=context_dimension, alpha = alpha, lambda_ = lambda_, n = n_users, W = UM.getW())
	# 	algorithms['factorUCB'] = FactorUCBAlgorithm(context_dimension = context_dimension, latent_dimension = 5, alpha = 0.05, alpha2 = 0.025, lambda_ = lambda_, n = n_users, itemNum=n_articles, W = UM.getW(), init='zero', window_size = -1)	

	simExperiment.runAlgorithms(algorithms, diffLists)