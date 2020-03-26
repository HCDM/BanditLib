import copy
import numpy as np
from scipy.sparse import csgraph
import datetime #
import os.path #
import argparse
import yaml
from sklearn.decomposition import TruncatedSVD
from sklearn import cluster
from sklearn.decomposition import PCA
from conf import sim_files_folder, save_address
from util_functions import *
from Articles import ArticleManager
from Users.Users import UserManager
from Users.CoUsers import CoUserManager
from RewardManager import RewardManager
from DatasetRewardManager import DatasetRewardManager
from DiffList.DiffManager import DiffManager

from lib.LinUCB import LinUCBAlgorithm, Uniform_LinUCBAlgorithm,Hybrid_LinUCBAlgorithm
from lib.hLinUCB import HLinUCBAlgorithm
from lib.factorUCB import FactorUCBAlgorithm
from lib.CoLin import CoLinUCBAlgorithm
from lib.GOBLin import GOBLinAlgorithm
from lib.CLUB import *
from lib.PTS import PTSAlgorithm
from lib.UCBPMF import UCBPMFAlgorithm
from lib.FairUCB import FairUCBAlgorithm
from lib.ThompsonSampling import ThompsonSamplingAlgorithm
from lib.LinPHE import LinPHEAlgorithm

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


def generate_algorithms(alg_dict, W, system_params):
	gen = alg_dict['general'] if alg_dict.has_key('general') and alg_dict['general'] else {}
	algorithms = {}
	diffLists = DiffManager()
	for i in alg_dict['specific']:
		print str(i)
		try:
			tmpDict = globals()['create' + i + 'Dict'](alg_dict['specific'][i] if alg_dict['specific'][i] else {}, gen, W, system_params)
		except KeyError:
			tmpDict = createBaseAlgDict(alg_dict['specific'][i] if alg_dict['specific'][i] else {}, gen, W, system_params)
		try:
			algorithms[i] = globals()[i + 'Algorithm'](tmpDict)
		except KeyError:
			raise NotImplementedError(i + " not currently implemented")
		diffLists.add_algorithm(i, algorithms[i].getEstimateSettings())
	print algorithms
	return algorithms, diffLists

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = '')
	parser.add_argument('--alg', dest='alg', help='Select a specific algorithm, could be CoLin, hLinUCB, factorUCB, LinUCB, etc.')
	parser.add_argument('--contextdim', type=int, help='Set dimension of context features.')
	parser.add_argument('--hiddendim', type=int, help='Set dimension of hidden features.')
	parser.add_argument('--config', dest='config', help='yaml config file')
        parser.add_argument('--dataset', required=False, choices=['LastFM', 'Delicious'], help='Select dataset to run. No Selection resuts in simulated rewards')
        parser.add_argument('--clusterfile', dest="clusterfile", help="input an clustering label file", 
                        metavar="FILE", type=lambda x: is_valid_file(parser, x))

	args = parser.parse_args()
	cfg = {}




	with open(args.config, 'r') as ymlfile:
		cfg = yaml.load(ymlfile)
	gen = cfg['general'] if cfg.has_key('general') else {}
	user = cfg['user'] if cfg.has_key('user') else {}
	article = cfg['article'] if cfg.has_key('article') else {}
	reco = cfg['reward'] if cfg.has_key('reward') else {}
        


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

	

	rewardManagerDict['epsilon'] = 0 # initialize W

	n_articles = article['number'] if article.has_key('number') else 1000
	ArticleGroups = article['groups'] if article.has_key('groups') else 5
        
        if gen.has_key('dataset') and gen['dataset'] == 'LastFM':
                        n_users = 2100
                        n_articles = 1900
        elif user.has_key('number'):
                n_users = user['number'] 
        else:
                n_users =  10
	UserGroups = user['groups'] if user.has_key('groups') else 5
	
	rewardManagerDict['poolArticleSize'] = gen['pool_article_size'] if gen.has_key('pool_article_size') else 10
	rewardManagerDict['batchSize'] = gen['batch_size'] if gen.has_key('batch_size') else 1

	# Matrix parameters
	matrixNoise = 0.01
	rewardManagerDict['matrixNoise'] = lambda : np.random.normal(scale = matrixNoise)
	rewardManagerDict['sparseLevel'] = n_users  # if smaller or equal to 0 or larger or enqual to usernum, matrix is fully connected


	# Parameters for GOBLin
	rewardManagerDict['Gepsilon'] = 1
	
	user['default_file'] = os.path.join(sim_files_folder, "users_"+str(n_users)+"context_"+str(context_dimension)+"latent_"+str(latent_dimension)+ "Ugroups" + str(UserGroups)+".json")
	# Override User type 
	if gen.has_key('collaborative'):
		if gen['collaborative']:
			use_coUsers = True
			reward_type = 'SocialLinear'
		else:
			use_coUsers = False
			reward_type = 'Linear'
	else:
		use_coUsers = user.has_key('collaborative') and user['collaborative']
		reward_type = reco['type'] if reco.has_key('type') else 'linear'


	#if user.has_key('collaborative') and user['collaborative']:
	if use_coUsers:
		UM = CoUserManager(context_dimension+latent_dimension, user, argv={'l2_limit':1, 'sparseLevel': n_users, 'matrixNoise': rewardManagerDict['matrixNoise']})
	else:
		UM = UserManager(context_dimension+latent_dimension, user, argv={'l2_limit':1})
	UM.CoTheta()

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
	#reward_type = reco['type'] if reco.has_key('type') else 'linear'
	
	#PCA
	pca_articles(articles, 'random')
	rewardManagerDict['articles'] = articles
	rewardManagerDict['testing_method'] = gen['testing_method'] if gen.has_key('testing_method') else "online"
	rewardManagerDict['noise'] = lambda : np.random.normal(scale = rewardManagerDict['NoiseScale'])
	rewardManagerDict['type'] = "UniformTheta"
	rewardManagerDict['simulation_signature'] = AM.signature


	
	for i in range(len(articles)):
		articles[i].contextFeatureVector = articles[i].featureVector[:context_dimension]

	# TODO: Add in reward options dictionary
	if gen.has_key('dataset') and gen['dataset'] != 'None':
                experiment = DatasetRewardManager(arg_dict = rewardManagerDict,dataset = gen['dataset'], clusterfile=args.clusterfile)
        else:
                experiment = RewardManager(arg_dict = rewardManagerDict, reward_type = reward_type)


        #print(vars(experiment))
	print "Starting for ", experiment.simulation_signature
	system_params = {
		'context_dim': context_dimension,
		'latent_dim': latent_dimension,
		'n_users': n_users,
		'n_articles': n_articles
	}

	algorithms, diffLists = generate_algorithms(cfg['alg'], UM.getW(), system_params)

	experiment.runAlgorithms(algorithms, diffLists)
