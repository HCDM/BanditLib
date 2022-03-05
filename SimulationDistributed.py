import copy
import numpy as np
from random import sample, shuffle
import random
import datetime
import os.path
import matplotlib.pyplot as plt
import argparse
# local address to save simulated users, simulated articles, and results
from conf import sim_files_folder, save_address
from util_functions import featureUniform, gaussianFeature, dsigmoid, sigmoid
from Articles import ArticleManager
from Users.ClusteredUsers import UserManager

from lib.AsyncLinUCB import AsyncLinUCB
from lib.DisLinUCB import DisLinUCB
from lib.AsyncLinUCB_AM import AsyncLinUCB_AM
from lib.FedGLB_UCB import FedGLB_UCB

class simulateOnlineData(object):
	def __init__(self, context_dimension, testing_iterations, plot, articles,
				 users, noise=lambda: 0, reward_model='linear', signature='', NoiseScale=0.0, poolArticleSize=None):

		self.simulation_signature = signature

		self.context_dimension = context_dimension
		self.testing_iterations = testing_iterations
		self.batchSize = 1

		self.plot = plot

		self.noise = noise
		self.reward_model = reward_model
		self.NoiseScale = NoiseScale
		
		self.articles = articles
		self.users = users

		if poolArticleSize is None:
			self.poolArticleSize = len(self.articles)
		else:
			self.poolArticleSize = poolArticleSize

	def getTheta(self):
		Theta = np.zeros(shape = (self.context_dimension, len(self.users)))
		for i in range(len(self.users)):
			Theta.T[i] = self.users[i].theta
		return Theta
	
	def batchRecord(self, iter_):
		print("Iteration %d"%iter_, " Elapsed time", datetime.datetime.now() - self.startTime)

	def getReward(self, user, pickedArticle):
		inner_prod = np.dot(user.theta, pickedArticle.featureVector)
		if self.reward_model == 'linear':
			reward = inner_prod
		elif self.reward_model == 'sigmoid':
			reward = sigmoid(inner_prod)
		else:
			raise ValueError
		return reward

	def GetOptimalReward(self, user, articlePool):		
		maxReward = float('-inf')
		maxx = None
		for x in articlePool:	 
			reward = self.getReward(user, x)
			if reward > maxReward:
				maxReward = reward
				maxx = x
		if self.reward_model == 'linear':
			maxReward = maxReward
		elif self.reward_model == 'sigmoid':
			maxReward = sigmoid(maxReward)
		else:
			raise ValueError
		return maxReward, maxx
	
	def getL2Diff(self, x, y):
		return np.linalg.norm(x-y) # L2 norm

	def regulateArticlePool(self):
		# Randomly generate articles
		self.articlePool = sample(self.articles, self.poolArticleSize)

	def runAlgorithms(self, algorithms):
		self.startTime = datetime.datetime.now()
		timeRun = self.startTime.strftime('_%m_%d_%H_%M') 
		filenameWriteRegret = os.path.join(save_address, 'AccRegret' + timeRun + '.csv')
		filenameWriteCommCost = os.path.join(save_address, 'AccCommCost' + timeRun + '.csv')
		filenameWritePara = os.path.join(save_address, 'ParameterEstimation' + timeRun + '.csv')

		tim_ = []
		BatchCumlateRegret = {}
		CommCostList = {}
		AlgRegret = {}
		ThetaDiffList = {}
		ThetaDiff = {}
		
		# Initialization
		# userSize = len(self.users)
		for alg_name, alg in algorithms.items():
			AlgRegret[alg_name] = []
			CommCostList[alg_name] = []
			BatchCumlateRegret[alg_name] = []
			if alg.CanEstimateUserPreference:
				ThetaDiffList[alg_name] = []

		with open(filenameWriteRegret, 'w') as f:
			f.write('Time(Iteration)')
			f.write(',' + ','.join([str(alg_name) for alg_name in algorithms.keys()]))
			f.write('\n')

		with open(filenameWriteCommCost, 'w') as f:
			f.write('Time(Iteration)')
			f.write(',' + ','.join([str(alg_name) for alg_name in algorithms.keys()]))
			f.write('\n')

		with open(filenameWritePara, 'w') as f:
			f.write('Time(Iteration)')
			f.write(','+ ','.join([str(alg_name)+'Theta' for alg_name in ThetaDiffList.keys()]))
			f.write('\n')

		for iter_ in range(self.testing_iterations):
			# prepare to record theta estimation error
			for alg_name, alg in algorithms.items():
				if alg.CanEstimateUserPreference:
					ThetaDiff[alg_name] = 0

			# for u in self.users:
			u = random.choices(population=self.users, weights=None, k=1)[0]
			self.regulateArticlePool()
			noise = self.noise()
			#get optimal reward for user x at time t
			OptimalReward, OptimalArticle = self.GetOptimalReward(u, self.articlePool)
			OptimalReward += noise

			for alg_name, alg in algorithms.items():
				pickedArticle = alg.decide(self.articlePool, u.id)
				reward = self.getReward(u, pickedArticle) + noise
				alg.updateParameters(pickedArticle, reward, u.id)

				regret = OptimalReward - reward  # pseudo regret, since noise is canceled out
				AlgRegret[alg_name].append(regret)
				CommCostList[alg_name].append(alg.totalCommCost)

				#update parameter estimation record
				if alg.CanEstimateUserPreference:
					ThetaDiff[alg_name] += self.getL2Diff(u.theta, alg.getTheta(u.id))

			for alg_name, alg in algorithms.items():
				if alg.CanEstimateUserPreference:
					ThetaDiffList[alg_name] += [ThetaDiff[alg_name]]
		
			if iter_%self.batchSize == 0:
				self.batchRecord(iter_)
				tim_.append(iter_)
				for alg_name, alg in algorithms.items():
					cumRegret = sum(AlgRegret[alg_name])
					BatchCumlateRegret[alg_name].append(cumRegret)
					print("{0: <16}: cum_regret {1}, cum_comm {2}".format(alg_name, cumRegret, alg.totalCommCost))
				with open(filenameWriteRegret, 'a+') as f:
					f.write(str(iter_))
					f.write(',' + ','.join([str(BatchCumlateRegret[alg_name][-1]) for alg_name in algorithms.keys()]))
					f.write('\n')
				with open(filenameWriteCommCost, 'a+') as f:
					f.write(str(iter_))
					f.write(',' + ','.join([str(CommCostList[alg_name][-1]) for alg_name in algorithms.keys()]))
					f.write('\n')
				with open(filenameWritePara, 'a+') as f:
					f.write(str(iter_))
					f.write(','+ ','.join([str(ThetaDiffList[alg_name][-1]) for alg_name in ThetaDiffList.keys()]))
					f.write('\n')

		if (self.plot==True): # only plot
			# # plot the results
			fig, axa = plt.subplots(2, 1, sharex='all')
			# Remove horizontal space between axes
			fig.subplots_adjust(hspace=0)

			print("=====Regret=====")
			for alg_name in algorithms.keys():
				axa[0].plot(tim_, BatchCumlateRegret[alg_name],label = alg_name)
				print('%s: %.2f' % (alg_name, BatchCumlateRegret[alg_name][-1]))
			axa[0].legend(loc='upper left',prop={'size':9})
			axa[0].set_xlabel("Iteration")
			axa[0].set_ylabel("Accumulative Regret")
			axa[1].set_ylim(bottom=0, top=200)

			print("=====Comm Cost=====")
			for alg_name in algorithms.keys():
				axa[1].plot(tim_, CommCostList[alg_name],label = alg_name)
				print('%s: %.2f' % (alg_name, CommCostList[alg_name][-1]))

			axa[1].set_xlabel("Iteration")
			axa[1].set_ylabel("Communication Cost")
			axa[1].set_ylim(bottom=0, top=20000)
			plt.savefig(os.path.join(save_address, "regretAndcommCost" + "_" + str(timeRun) + '.png'), dpi=300, bbox_inches='tight', pad_inches=0.0)
			plt.show()

		finalRegret = {}
		for alg_name in algorithms.keys():
			finalRegret[alg_name] = BatchCumlateRegret[alg_name][:-1]
		return finalRegret

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = '')
	parser.add_argument('--T', dest='T', help='total number of iterations')
	parser.add_argument('--n', dest='n', help='total number of clients')
	parser.add_argument('--contextdim', type=int, help='Set dimension of context features.')
	parser.add_argument('--globaldim', dest='global_dim', help='Set dimension of globally shared context features.')
	parser.add_argument('--rewardmodel', dest='reward_model', help='Set reward model to be linear or sigmoid.')
	args = parser.parse_args()

	## Environment Settings ##
	config = {}
	if args.contextdim:
		config["context_dimension"] = int(args.contextdim)
	else:
		config["context_dimension"] = 25
	if args.T:
		config["testing_iterations"] = int(args.T)
	else:
		config["testing_iterations"] = 50000
	if args.n:
		config["n_users"] = int(args.n)
	else:
		config["n_users"] = 1000
	if args.global_dim:
		config["global_dim"] = int(args.global_dim)
		assert config["global_dim"] < config["context_dimension"]
	else:
		config["global_dim"] = config["context_dimension"]
	if args.reward_model:
		config["reward_model"] = args.reward_model
	else:
		config["reward_model"] = 'linear'
	config["NoiseScale"] = 0.1  # standard deviation of Gaussian noise
	config["n_articles"] = 1000
	poolArticleSize = 25

	## Set Up Simulation ##
	UM = UserManager(config["context_dimension"], config["n_users"], thetaFunc=gaussianFeature, argv={'l2_limit': 1})
	if config["global_dim"] == config["context_dimension"]:
		users = UM.simulateThetaForHomoUsers()
	else:
		users = UM.simulateThetaForHeteroUsers(global_dim=config["global_dim"])
	AM = ArticleManager(config["context_dimension"], n_articles=config["n_articles"], FeatureFunc=gaussianFeature, argv={'l2_limit': 1}, ArticleGroups=0)
	articles = AM.simulateArticlePool()

	simExperiment = simulateOnlineData(	context_dimension=config["context_dimension"],
										testing_iterations=config["testing_iterations"],
										plot=True,
										articles=articles,
										users = users,
										noise=lambda: np.random.normal(scale=config["NoiseScale"]),
										reward_model=config["reward_model"],
										signature=AM.signature,
										NoiseScale=config["NoiseScale"],
										poolArticleSize=poolArticleSize)

	## Initiate Bandit Algorithms ##
	algorithms = {}

	config["lambda_"] = 0.1
	config["delta"] = 1e-1
	S = 1
	R = 0.5
	c_mu = dsigmoid(S * 1)

	D2 = (config["testing_iterations"]) / (config["n_users"] * config["context_dimension"]* np.log(config["testing_iterations"]))
	algorithms['DisLinUCB'] = DisLinUCB(dimension=config["context_dimension"], alpha=-1, lambda_=config["lambda_"],
												   delta_=config["delta"],
												   NoiseScale=config["NoiseScale"], threshold=D2)
	algorithms['AsyncLinUCB'] = AsyncLinUCB(dimension=config["context_dimension"], alpha=-1, lambda_=config["lambda_"], delta_=config["delta"],
												   NoiseScale=config["NoiseScale"], gammaU=5, gammaD=5)
	algorithms['AsyncLinUCB_AM'] = AsyncLinUCB_AM(dimension_g=config["global_dim"], dimension_l=config["context_dimension"] - config["global_dim"], alpha=-1,
											lambda_=config["lambda_"],
											delta_=config["delta"],
											NoiseScale=config["NoiseScale"], gammaU=5, gammaD=5)
	algorithms['FedGLB_UCB'] = FedGLB_UCB(dimension=config["context_dimension"], lambda_=config["lambda_"], threshold=D2, n_users=config["n_users"], c_mu=c_mu, delta=config["delta"], S=S, R=R, alpha=None, alpha_t_scaling=0.1, init_x=np.zeros(config["context_dimension"]), max_iters=None)
	## Run Simulation ##
	print("Starting for ", simExperiment.simulation_signature)
	simExperiment.runAlgorithms(algorithms)