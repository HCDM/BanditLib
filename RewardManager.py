from Rewards.LinearReward import LinearReward
from Rewards.SocialLinearReward import SocialLinearReward
import numpy as np 
import datetime
import os.path
import copy
from conf import sim_files_folder, save_address
from random import sample, shuffle
import matplotlib.pyplot as plt

class RewardManager():
	def __init__(self, arg_dict, reward_type = 'linear'):
		for key in arg_dict:
			setattr(self, key, arg_dict[key])
		#self.W, self.W0 = self.constructAdjMatrix(self.sparseLevel)
		if(reward_type == 'social_linear'):
			self.reward = SocialLinearReward(self.k, self.W)
		else:
			self.reward = LinearReward(self.k)
	
	def batchRecord(self, iter_):
		print "Iteration %d"%iter_, "Pool", len(self.articlePool)," Elapsed time", datetime.datetime.now() - self.startTime

	def regulateArticlePool(self):
		# Randomly generate articles
		self.articlePool = sample(self.articles, self.poolArticleSize)   
	
	def getL2Diff(self, x, y):
		return np.linalg.norm(x-y) # L2 norm

	def runAlgorithms(self, algorithms, diffLists):
		self.startTime = datetime.datetime.now()
		timeRun = self.startTime.strftime('_%m_%d_%H_%M') 
		filenameWriteRegret = os.path.join(save_address, 'AccRegret' + timeRun + '.csv')
		filenameWritePara = os.path.join(save_address, 'ParameterEstimation' + timeRun + '.csv')

		# compute co-theta for every user
		tim_ = []
		BatchCumlateRegret = {}
		AlgRegret = {}
		CoThetaVDiffList = {}
		RDiffList ={}
		RVDiffList = {}

		CoThetaVDiff = {}
		RDiff ={}
		RVDiff = {}

		Var = {}
		
		# Initialization
		userSize = len(self.users)
		for alg_name, alg in algorithms.items():
			AlgRegret[alg_name] = []
			BatchCumlateRegret[alg_name] = []
			Var[alg_name] = []

		
		with open(filenameWriteRegret, 'w') as f:
			f.write('Time(Iteration)')
			f.write(',' + ','.join( [str(alg_name) for alg_name in algorithms.iterkeys()]))
			f.write('\n')
		
		with open(filenameWritePara, 'w') as f:
			f.write('Time(Iteration)')
			diffLists.initial_write(f)
			f.write('\n')
		
		# Training
		shuffle(self.articles)
		for iter_ in range(self.training_iterations):
			article = self.articles[iter_]										
			for u in self.users:
				noise = self.noise()	
				reward = self.reward.getReward(u, article)
				reward += noise										
				for alg_name, alg in algorithms.items():
					alg.updateParameters(article, reward, u.id)	

			if 'syncCoLinUCB' in algorithms:
				algorithms['syncCoLinUCB'].LateUpdate()	

		#Testing
		for iter_ in range(self.testing_iterations):
				
			for u in self.users:
				self.regulateArticlePool() # select random articles

				noise = self.noise()
				#get optimal reward for user x at time t
				pool_copy = copy.deepcopy(self.articlePool)
				#OptimalReward, OptimalArticle = self.reward.getOptimalReward(u, pool_copy) 
				OptimalReward = self.reward.getOptimalRecommendationReward(u, self.articlePool, self.k)
				OptimalReward += noise
							
				for alg_name, alg in algorithms.items():
					recommendation = alg.createRecommendation(self.articlePool, u.id, self.k)
						
					pickedArticle = recommendation.articles[0]
					reward, rewardList = self.reward.getRecommendationReward(u, recommendation, noise)
					if (self.testing_method=="online"):
						#alg.updateParameters(pickedArticle, reward, u.id)
						alg.updateRecommendationParameters(recommendation, rewardList, u.id)
						if alg_name =='CLUB':
							n_components= alg.updateGraphClusters(u.id,'False')

					regret = OptimalReward - reward
					AlgRegret[alg_name].append(regret)

					if u.id == 0:
						if alg_name in ['LBFGS_random','LBFGS_random_around','LinUCB', 'LBFGS_gradient_inc']:
							means, vars = alg.getProb(self.articlePool, u.id)
							Var[alg_name].append(vars[0])

					# #update parameter estimation record
					diffLists.update_parameters(alg_name, self, u, alg, pickedArticle, reward, noise)

			if 'syncCoLinUCB' in algorithms:
				algorithms['syncCoLinUCB'].LateUpdate()	
			diffLists.append_to_lists(userSize)
				
			if iter_%self.batchSize == 0:
				self.batchRecord(iter_)
				tim_.append(iter_)
				for alg_name in algorithms.iterkeys():
					BatchCumlateRegret[alg_name].append(sum(AlgRegret[alg_name]))

				with open(filenameWriteRegret, 'a+') as f:
					f.write(str(iter_))
					f.write(',' + ','.join([str(BatchCumlateRegret[alg_name][-1]) for alg_name in algorithms.iterkeys()]))
					f.write('\n')
				with open(filenameWritePara, 'a+') as f:
					f.write(str(iter_))
					diffLists.iteration_write(f)
					f.write('\n')

		if (self.plot==True): # only plot
			# plot the results	
			f, axa = plt.subplots(1, sharex=True)
			for alg_name in algorithms.iterkeys():	
				axa.plot(tim_, BatchCumlateRegret[alg_name],label = alg_name)
				print '%s: %.2f' % (alg_name, BatchCumlateRegret[alg_name][-1])
			axa.legend(loc='upper left',prop={'size':9})
			axa.set_xlabel("Iteration")
			axa.set_ylabel("Regret")
			axa.set_title("Accumulated Regret")
			plt.show()

			# plot the estimation error of co-theta
			f, axa = plt.subplots(1, sharex=True)
			time = range(self.testing_iterations)
			diffLists.plot_diff_lists(axa, time)
		
			axa.legend(loc='upper right',prop={'size':6})
			axa.set_xlabel("Iteration")
			axa.set_ylabel("L2 Diff")
			axa.set_yscale('log')
			axa.set_title("Parameter estimation error")
			plt.show()

		finalRegret = {}
		for alg_name in algorithms.iterkeys():
			finalRegret[alg_name] = BatchCumlateRegret[alg_name][:-1]
		return finalRegret