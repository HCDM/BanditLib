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
# from util_functions import featureUniform, gaussianFeature, createLinUCBDict, \
# 	createCoLinUCBDict, createHLinUCBDict, createUCBPMFDict, createFactorUCBDict, \
# 	createCLUBDict, createPTSDict, createBaseAlgDict
from util_functions import *
from Articles import ArticleManager, ArticleManagerLastFm
from Users.Users import UserManager
from Users.CoUsers import CoUserManager
from RewardManager import RewardManager
from DiffList.DiffManager import DiffManager

from lib.LinUCB import LinUCBAlgorithm, Uniform_LinUCBAlgorithm,Hybrid_LinUCBAlgorithm
from lib.PrivateLinUCB import PrivateLinUCBAlgorithm
from lib.hLinUCB import HLinUCBAlgorithm
from lib.factorUCB import FactorUCBAlgorithm
from lib.CoLin import CoLinUCBAlgorithm
from lib.PrivateCoLin import PrivateCoLinUCBAlgorithm
from lib.GOBLin import GOBLinAlgorithm
from lib.CLUB import *
from lib.PTS import PTSAlgorithm
from lib.UCBPMF import UCBPMFAlgorithm
from lib.FairUCB import FairUCBAlgorithm

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


def generate_algorithms(global_dict, alg_dict, W, system_params):
	gen = alg_dict['general'] if alg_dict.has_key('general') and alg_dict['general'] else {}
	algorithms = {}
	diffLists = DiffManager()
	for alg in alg_dict['specific']:
		alg_name = alg['algorithm']
		alg_id = alg['id']
		try:
			tmpDict = globals()['create' + alg_name + 'Dict'](global_dict, alg if alg else {}, gen, W, system_params)
		except KeyError:
			tmpDict = createBaseAlgDict(global_dict, alg if alg else {}, gen, W, system_params)
		try:
			algorithms[alg_id] = {
				'name': alg_name,
				'algorithm': globals()[alg_name + 'Algorithm'](tmpDict)
			}
		except KeyError as e:
			if (alg_name + 'Algorithm') not in globals():
				raise NotImplementedError(alg_name + " not currently implemented")
			else:
				raise e
		diffLists.add_algorithm(alg_id, algorithms[alg_id]['algorithm'].getEstimateSettings())
	return algorithms, diffLists

def create_reward_manager_dict(config, clargs):
	def extract_context_dimension(general_cfg, clargs, default=20):
		context_dimension = default
		if clargs.contextdim:
			context_dimension = clargs.contextdim
		elif 'context_dimension' in general_cfg:
			context_dimension = general_cfg['context_dimension']
		return context_dimension

	def extract_latent_dimension(general_cfg, clargs, default=0):
		latent_dimension = default
		if clargs.hiddendim:
			latent_dimension = clargs.hiddendim
		elif 'hidden_dimension' in general_cfg:
			latent_dimension = general_cfg['hidden_dimension']
		return latent_dimension

	def extract_argument(parent_cfg, name, default):
		if name in parent_cfg:
			return parent_cfg[name]
		return default

	def create_filename(d, ext='.json', first_key=None):
		"""Generate default filename from set of key-values.

		The key-value pairs are ordered alphabetically with the exception of
		the first key, if specified.

		Args:
			d (dict): Set of key-value pairs to include in filename.
			ext (str): File extension, default to .json.
			first_key (Any): If specified, make this the first key in the filename.

		Returns:
			The full filepath string.
		"""
		parent_dir = sim_files_folder
		filename_array = []
		sorted_d_items = sorted(d.items(), key=lambda t: t[0])
		for k, v in sorted_d_items:
			if first_key and k == first_key:
				filename_array.insert(0, '{}_{}'.format(k, v))
			else:
				filename_array.append('{}_{}'.format(k, v))
		filename = ''.join(filename_array) + ext
		filepath = os.path.join(parent_dir, filename)
		return filepath
		

	reward_manager_dict = {}

	general_cfg = config['general'] if 'general' in config else {}
	article_cfg = config['article'] if 'article' in config else {}
	pool_cfg = config['pool'] if 'pool' in config else {}
	user_cfg = config['user'] if 'user' in config else {}
	reward_cfg = config['reward'] if 'reward' in config else {}

	# Other attributes
	n_articles = extract_argument(article_cfg, 'number', default=1000)
	n_article_groups = extract_argument(article_cfg, 'groups', default=5)
	n_users = extract_argument(user_cfg, 'number', default=10)
	n_users_groups = extract_argument(user_cfg, 'groups', default=5)

	if extract_argument(article_cfg, 'load', default=False):
		if extract_argument(article_cfg, 'format', default='default') == 'lastfm':
			# hardcoded for simplicity
			n_users = 1892
			n_articles = 12523
			user_cfg['number'] = 1892

	reward_manager_dict['n_articles'] = n_articles
	reward_manager_dict['n_article_groups'] = n_article_groups
	reward_manager_dict['n_users'] = n_users
	reward_manager_dict['n_users_groups'] = n_users_groups


	# Config-independent
	reward_manager_dict['NoiseScale'] = 0.01
	reward_manager_dict['noise'] = lambda : np.random.normal(scale = reward_manager_dict['NoiseScale'])
	reward_manager_dict['epsilon'] = 0
	reward_manager_dict['Gepsilon'] = 1  # parameter for GOBLin
	reward_manager_dict['matrixNoise'] = lambda : np.random.normal(scale = 0.01)  # matrix parameters
	reward_manager_dict['sparseLevel'] = n_users  # if smaller or equal to 0 or larger or enqual to usernum, matrix is fully connected
	reward_manager_dict['type'] = 'UniformTheta'

	# General
	context_dimension = extract_context_dimension(general_cfg, clargs)
	latent_dimension = extract_latent_dimension(general_cfg, clargs)
	reward_manager_dict['context_dimension'] = context_dimension
	reward_manager_dict['latent_dimension'] = latent_dimension
	reward_manager_dict['training_iterations'] = extract_argument(general_cfg, 'training_iterations', default=0)
	reward_manager_dict['testing_iterations'] = extract_argument(general_cfg, 'testing_iterations', default=100)
	reward_manager_dict['plot'] = extract_argument(general_cfg, 'plot', default=True)
	reward_manager_dict['poolArticleSize'] = extract_argument(general_cfg, 'pool_article_size', default=10)
	reward_manager_dict['batchSize'] = extract_argument(general_cfg, 'batch_size', default=1)
	reward_manager_dict['testing_method'] = extract_argument(general_cfg, 'testing_method', default='online')

	# Article
	default_article_filename = create_filename({
		'articles': n_articles,
		'context': context_dimension,
		'latent': latent_dimension,
		'Agroups': n_article_groups
	}, first_key='articles')
	article_filename = extract_argument(article_cfg, 'filename', default=default_article_filename)
	article_format = extract_argument(article_cfg, 'format', default='default')
	if article_format.lower() == 'lastfm':
		AM = ArticleManagerLastFm(context_dimension+latent_dimension, n_articles=n_articles, ArticleGroups = n_article_groups,
				FeatureFunc=featureUniform,  argv={'l2_limit':1})
	else:
		AM = ArticleManager(context_dimension+latent_dimension, n_articles=n_articles, ArticleGroups = n_article_groups,
				FeatureFunc=featureUniform,  argv={'l2_limit':1})
	if extract_argument(article_cfg, 'load', default=False):
		articles = AM.loadArticles(article_filename)
		n_articles = len(articles)  # Override n_articles
	else:
		articles = AM.simulateArticlePool()
		if extract_argument(article_cfg, 'save', default=False):
			AM.saveArticles(articles, article_filename, force=False)
	pca_articles(articles, 'ascend')
	reward_manager_dict['articles'] = articles
	reward_manager_dict['simulation_signature'] = AM.signature

	# Pool
	default_pool_filename = create_filename({
		'pool': reward_manager_dict['poolArticleSize'],
		'articles': n_articles,
		'iterations': reward_manager_dict['testing_iterations']
	}, first_key='pool')
	reward_manager_dict['pool_filename'] = extract_argument(pool_cfg, 'filename', default=default_pool_filename)
	reward_manager_dict['load_pool'] = extract_argument(pool_cfg, 'load', default=False)
	reward_manager_dict['save_pool'] = extract_argument(pool_cfg, 'save', default=False)
	reward_manager_dict['pool_format'] = extract_argument(pool_cfg, 'format', default='default')

	# User
	default_user_filename = create_filename({
		'users': n_users,
		'context': context_dimension,
		'latent': latent_dimension,
		'Ugroups': n_users_groups
	}, first_key='users')
	user_cfg['filename'] = extract_argument(user_cfg, 'filename', default=default_user_filename)
	
	if 'collaborative' in general_cfg:
		if extract_argument(general_cfg, 'collaborative', default=False):
			use_coUsers = True
			reward_manager_dict['reward_type'] = 'SocialLinear'
		else:
			use_coUsers = False
			reward_manager_dict['reward_type'] = 'Linear'
	else:
		use_coUsers = extract_argument(user_cfg, 'collaborative', default=False)
		reward_manager_dict['reward_type'] = extract_argument(reward_cfg, 'type', default='linear')

	if use_coUsers:
		UM = CoUserManager(context_dimension+latent_dimension, user_cfg, argv={'l2_limit':1, 'sparseLevel': n_users, 'matrixNoise': reward_manager_dict['matrixNoise']})
	else:
		UM = UserManager(context_dimension+latent_dimension, user_cfg, argv={'l2_limit':1})
	UM.CoTheta()

	reward_manager_dict['W'] = UM.getW()
	reward_manager_dict['users'] = UM.getUsers()

	# Reward
	reward_manager_dict['k'] = extract_argument(reward_cfg, 'k', default=1)
	# Reward noise
	reward_noise_cfg = reward_cfg['noise'] if 'noise' in reward_cfg else {}
	reward_manager_dict['load_reward_noise'] = extract_argument(reward_noise_cfg, 'load', default=False)
	reward_manager_dict['save_reward_noise'] = extract_argument(reward_noise_cfg, 'save', default=False)
	default_reward_noise_filename = create_filename({
		'reward_noise': reward_manager_dict['k'],
		'users': n_users,
		'iterations': reward_manager_dict['testing_iterations']
	}, first_key='reward_noise')
	reward_manager_dict['reward_noise_filename'] = extract_argument(reward_noise_cfg, 'filename', default=default_reward_noise_filename)
	# Reward noise resample
	if 'resample' in reward_noise_cfg:
		reward_manager_dict['reward_noise_resample_active'] = True
		reward_manager_dict['reward_noise_resample_round'] = extract_argument(reward_noise_cfg['resample'], 'round', default=0)
		reward_manager_dict['reward_noise_resample_change'] = extract_argument(reward_noise_cfg['resample'], 'change', default=0.5)
	else:
		reward_manager_dict['reward_noise_resample_active'] = False

	return reward_manager_dict

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

	# TODO: Add in reward options dictionary
	reward_manager_dict = create_reward_manager_dict(cfg, args)
	
	for i in range(len(reward_manager_dict['articles'])):
		reward_manager_dict['articles'][i].contextFeatureVector = reward_manager_dict['articles'][i].featureVector[:reward_manager_dict['context_dimension']]

	simExperiment = RewardManager(arg_dict = reward_manager_dict, reward_type = reward_manager_dict['reward_type'])

	print "Starting for ", simExperiment.simulation_signature
	system_params = {
		'context_dim': reward_manager_dict['context_dimension'],
		'latent_dim': reward_manager_dict['latent_dimension'],
		'n_users': reward_manager_dict['n_users'],
		'n_articles': reward_manager_dict['n_articles']
	}

	general_cfg = cfg['general'] if 'general' in cfg else {}
	algorithms, diffLists = generate_algorithms(general_cfg, cfg['alg'], reward_manager_dict['W'], system_params)

	simExperiment.runAlgorithms(algorithms, diffLists)