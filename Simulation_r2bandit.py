import numpy as np
from operator import itemgetter      #for easiness in sorting and finding max and stuff
from matplotlib.pylab import *
import matplotlib
matplotlib.use('Agg')
from random import sample, choice
from scipy.sparse import csgraph 
import os
# local address to save simulated users, simulated articles, and results
from conf import sim_files_folder, result_folder, save_address
from util_functions import *
from Articles_generator_r2bandit import *
from Users_generator_r2bandit import *

from lib.GLMUCB1 import r2_GLMUCB1Algorithm
from lib.GLMUCB import reward_GLMUCBAlgorithm, return_GLMUCBAlgorithm
from lib.r2bandit import r2_banditAlgorithm

from scipy.linalg import sqrtm
import math
import argparse
from scipy import stats
import random
import copy 

from sklearn.decomposition import TruncatedSVD



class SimArticle():
	def __init__(self, article, type_):
		self.article = article
		self.type_ = type_

class simulateOnlineData():
	def __init__(self, dimension, iterations, articletypes, users, 
					batchSize = 1000,
					noise_Click = lambda : 0,
					noise_Return = lambda : 0 ,
					type_ = 'UniformTheta', 
					signature = '', 
					poolArticleSize = 10, 
					noiseLevel = 0,
					epsilon = 1, FutureWeight = 0.3, ReturnThreshold = 0.5, alpha = 0.1, usealphaT = False):

		self.simulation_signature = signature
		self.type = type_

		self.dimension = dimension
		self.iterations = iterations
		self.noise_Click = noise_Click
		self.noise_Return = noise_Return
		
		self.users = users
		self.poolArticleSize = poolArticleSize
		self.batchSize = batchSize
		
		self.noiseLevel = noiseLevel
		self.W = np.identity(len(users))
		self.FutureWeight = FutureWeight

		self.articlePool = []
		self.atypes = articletypes
		self.alpha = alpha
		self.usealphaT = usealphaT

		self.ReturnThreshold = ReturnThreshold

	def getTheta(self):
		Theta = np.zeros(shape = (self.dimension, len(self.users)))
		for i in range(len(self.users)):
			Theta.T[i] = self.users[i].theta
		return Theta
	def generateUserFeature(self,W):
		svd = TruncatedSVD(n_components=5)
		result = svd.fit(W).transform(W)
		return result
	def sigmoid(self, x):
		return 1 / (1 + math.exp(-x))
	def getL2Diff(self, x, y):
		return np.linalg.norm(x-y) # L2 norm

	def batchRecord(self, iter_):
		print "Iteration %d"%iter_, "Pool", len(self.articlePool)," Elapsed time", datetime.datetime.now() - self.startTime

	def regulateArticlePool(self, currentUser):
		articlesList = currentUser.articlesList
		# randomly generate articles 
		# guarantee each type of article is in the arm pool
		ArticleTypeNum = len(articlesList)
		n= self.poolArticleSize/ArticleTypeNum
		self.articlePool = []
		for i in range(ArticleTypeNum):
			Pool = sample(articlesList[i], n)
			self.articlePool +=Pool
		self.articlePool = np.asarray(self.articlePool)
		random.shuffle(self.articlePool)
		
	def getClick(self, user, Article, noise):
		reward = np.dot(user.theta, Article.featureVector) + noise
		clickProb = self.sigmoid(reward)
		randomNum = random.uniform(0,1)
		#click_threshold = 0.7
		
		if (randomNum )<= clickProb:
			click = 1
		else:
			click = 0
		return click  #Binary

	def getReturnTime(self, user, Article, noise):
		Intensity = np.exp(np.dot(user.beta, Article.featureVector) + noise)

		#sample return time from exponential distribution (parameterized by 1/Intensity)
		SampledReturnTime = 0.0
		sample_num = 100        # sample 20 times
		for i in range(sample_num):
			t = np.random.exponential(1.0/Intensity)
			SampledReturnTime +=t
		SampledReturnTime = SampledReturnTime/float(sample_num)
		return  SampledReturnTime
	
	

	def runAlgorithms(self, algorithms):
		# get cotheta for each user
		self.startTime = datetime.datetime.now()
		timeRun = datetime.datetime.now().strftime('_%m_%d') 
		timeRun_Save = datetime.datetime.now().strftime('_%m_%d_%H_%M') 
		fileSig = ''
		filenameWriteReward = {}
		filenameWriteOptimalRatio = {}
		
		for alg_name, alg in algorithms.items():
			fileSig = str(alg_name) + '_UserNum'+ str(len(self.users)) +'Article' + str(len(self.users[0].articlesList)) + 'alpha' + str(self.alpha)  +'_Noise'+str(self.noiseLevel) + 'weight_' + str(self.FutureWeight) +'ReturnThreshold' + str(self.ReturnThreshold)
			if self.usealphaT:
				fileSig = fileSig + '_alphaT'
			filenameWriteReward[alg_name] = os.path.join(save_address,  fileSig+ 'Reward_' + timeRun_Save +'.csv')


		self.startTime = datetime.datetime.now()

		tim_ = []
		BatchAverageRegret = {}
		AccRegret = {}
		ThetaDiffList = {}
		BetaDiffList = {}

		
		ThetaDiffList_user = {}
		BetaDiffList_user = {}
		USERS = {}

		TotalReward = {}
		TotalTime = {}

		RewardList = {}
		TimeList = {}

		TotalRewardList = {}
		uncountRewardList ={}
		uncountRewardList_Time ={}

		SelectedArticleType = {}

		SelectedNum = {}
		SelectedOptNum = {}
		SelectRatioList = {}
		
		# Initialization
		for alg_name, alg in algorithms.items():
			BatchAverageRegret[alg_name] = []		
			AccRegret[alg_name] = {}
			USERS[alg_name] = self.users
			TotalReward[alg_name] = 0.0
			TotalTime[alg_name] = 0.0

			RewardList[alg_name] = []
			TimeList[alg_name] = []

			TotalRewardList[alg_name] = []
			TotalRewardList[alg_name].append(0)

			uncountRewardList[alg_name] = []
			uncountRewardList_Time[alg_name] = []

			SelectedArticleType[alg_name] = {}


			SelectedNum[alg_name] = {}
			SelectedOptNum[alg_name] =  {}
			SelectRatioList[alg_name] = {}

			for j in range(len(self.users)):
				SelectedNum[alg_name][j] = 0
				SelectedOptNum[alg_name][j] = 0
				SelectRatioList[alg_name][j] = []

			SelectedNum[alg_name]['all'] = 0
			SelectedOptNum[alg_name]['all'] = 0
			SelectRatioList[alg_name]['all'] = []


			for type_ in self.atypes:
				SelectedArticleType[alg_name][type_] = 0


			if alg.CanEstimateUserPreference:
				ThetaDiffList[alg_name] = []
			if alg.CanEstimateReturn:
				BetaDiffList[alg_name] = []


			for i in range(len(self.users)):
				AccRegret[alg_name][i] = []

			# with open(filenameWriteReward[alg_name], 'a+') as f:
			# 	f.write('Time(Iteration)')
			# 	f.write(',' + 'Reward')
			# 	f.write(',' + 'ReturnTime')
			# 	f.write(',' + 'ArticleType')
			# 	f.write('\n')

		
		userSize = len(self.users)
		checkPoint = -1
		# Loop begin	
		iter_ = 0
		while iter_ < self.iterations:
			iter_ +=1	
			# prepare to record theta estimation error
			for alg_name, alg in algorithms.items():
				if alg.CanEstimateUserPreference:
					ThetaDiffList_user[alg_name] = []
				if alg.CanEstimateReturn:
					BetaDiffList_user[alg_name] = []
			# generate the noise in click feedback and return time feedback
			noise_Click = self.noise_Click()
			noise_Return = self.noise_Return()
			for i in range(len(self.users)):
				currentUser = self.users[i]
				# generate arm pool for current interaction
				self.regulateArticlePool(currentUser) 
				pool_article_Arr = getPoolArticleArr(self.articlePool)

				for alg_name, alg in algorithms.items():
					u = USERS[alg_name][i]
					pickedArticle= alg.decide(self.articlePool, u.id, pool_article_Arr)
					SelectedNum[alg_name][i] +=1
					SelectedNum[alg_name]['all'] +=1
					if pickedArticle.type == 'largeTheta_largeBeta':
						SelectedOptNum[alg_name][i] +=1
						SelectedOptNum[alg_name]['all'] +=1
					SelectRatioList[alg_name][i].append(float(SelectedOptNum[alg_name][i])/float(SelectedNum[alg_name][i]))
					SelectedArticleType[alg_name][pickedArticle.type] +=1  #Update Selected article type

					# compute the corresponding click for the selected arm
					reward = self.getClick(u, pickedArticle, noise_Click)
					# compute the corresponding return time for the selected arm
					returnTime= self.getReturnTime(u, pickedArticle, noise_Return)

					print alg_name, 'click:', reward, 'returnTime:', returnTime, 'articletype:', pickedArticle.type

					# update model parameters according to the feedback
					alg.updateParameters(pickedArticle, reward, u.id, returnTime)

					TotalTime[alg_name] += returnTime
					RewardList[alg_name].append(reward)
					TimeList[alg_name].append(returnTime)

					# with open(filenameWriteReward[alg_name], 'a+') as f:
					# 	f.write(str(iter_))
					# 	f.write(',' + str(reward))
					# 	f.write(',' + str(returnTime))
					# 	f.write(',' + str(pickedArticle.type))
					# 	f.write('\n')

					if alg.CanEstimateUserPreference:
						ThetaDiffList_user[alg_name] += [self.getL2Diff(u.theta, alg.getTheta(u.id))]
					if alg.CanEstimateReturn:
						BetaDiffList_user[alg_name] +=[self.getL2Diff(u.beta, alg.getBeta(u.id))]
						#BetaDiffList_user[alg_name] +=[np.dot(pickedArticle.featureVector, u.beta) - np.dot( pickedArticle.featureVector, alg.getBeta(u.id))   ]
						#BetaDiffList_user[alg_name] +=[np.dot(pickedArticle.featureVector, u.beta) -  np.exp( np.dot( pickedArticle.featureVector, alg.getBeta(u.id)) )  ]

					
			for alg_name, alg in algorithms.items():
				SelectRatioList[alg_name]['all'].append(float(SelectedOptNum[alg_name]['all'])/float(SelectedNum[alg_name]['all']))

			for alg_name, alg in algorithms.items():
				if alg.CanEstimateUserPreference:
					ThetaDiffList[alg_name] += [sum(ThetaDiffList_user[alg_name])/float(userSize)]
				if alg.CanEstimateReturn:
					BetaDiffList[alg_name] +=[sum(BetaDiffList_user[alg_name])/float(userSize)]
				
			if iter_%self.batchSize == 0:
				self.batchRecord(iter_)
				tim_.append(iter_)
				for alg_name in algorithms.iterkeys():
					#print [u for u in AccRegret[alg_name].itervalues()]
					TotalAccRegret = sum(sum (u) for u in AccRegret[alg_name].itervalues())
					BatchAverageRegret[alg_name].append(TotalAccRegret)
				
		
		for alg_name in algorithms.iterkeys():
			checkPoint = 0
			TotalRewardList[alg_name].append(0)
			l = 0
			for t in range(1000):
				checkPoint += 0.5
				if checkPoint > sum(TimeList[alg_name]):
					TotalRewardList[alg_name].append(sum(RewardList[alg_name]))
				else:
					#l = 0
					r = len(RewardList[alg_name])-1
					for i in range(l , r):	
						if checkPoint >= sum(TimeList[alg_name][0:i]) and checkPoint < sum(TimeList[alg_name][0:(i+1)]):
							TotalRewardList[alg_name].append(sum(RewardList[alg_name][0:i]))
							l = i
							break


		# plot the results
		
		for alg_name in algorithms.iterkeys():
			print alg_name, TotalRewardList[alg_name][-1]
			print alg_name, SelectedArticleType[alg_name]
			plt.plot(TotalRewardList[alg_name],label = alg_name)
		plt.xlabel('time')
		plt.ylabel('reward')
		plt.legend(loc = 'lower right')
		plt.show()	

		for alg_name in algorithms.iterkeys():
			plt.bar(range(len(SelectedArticleType[alg_name])), SelectedArticleType[alg_name].values(), align='center')
			plt.xticks(range(len(SelectedArticleType[alg_name])), SelectedArticleType[alg_name].keys(), label = alg_name)
			plt.title(alg_name)
			plt.show()
		for i in range(len(self.users)):
			for alg_name in algorithms.iterkeys():
				plt.plot(SelectRatioList[alg_name][i], label = alg_name )  #label = alg_name + str(i)
		#plt.plot(SelectRatioList[alg_name]['all'], label = alg_name+'ALL')
		plt.xlabel('Time')
		plt.ylabel('Optimal article type Ratio')
		plt.legend(loc = 'lower right')
		plt.show()



if __name__ == '__main__':
	iterations = 500
	NoiseScale = .1

	dimension = 25
	#alpha  = 0.2
	alpha = 0.1
	lambda_ = 0.1   # Initialize A
	epsilon = 10 # initialize W
	eta_ = 0.1

	n_articles = 100
	ArticleGroups = 5

	n_users = 100

	poolSize = 20
	batchSize = 1

 
	eGreedy = 0.3
	#atypes = []

	parser = argparse.ArgumentParser(description = '')
	parser.add_argument('--alg', dest='alg', help='Select a specific algorithm, could be CoLin, GOBLin, AsyncCoLin, or SyncCoLin')

	parser.add_argument('--RankoneInverse', action='store_true',
	                help='Use Rankone Correction to do matrix inverse') 
	parser.add_argument('--userNum', dest = 'userNum', help = 'Set the userNum, can be 40, 80, 100')
	parser.add_argument('--NoiseScale', dest = 'NoiseScale', help = 'Set NoiseScale')
	parser.add_argument('--FutureWeight', dest = 'FutureWeight', help = 'Set NoiseScale')
	parser.add_argument('--ReturnThreshold', dest = 'ReturnThreshold', help = 'threshold of user return, which is defined as tau in the paper')
	parser.add_argument('--alpha', dest = 'alpha', help = 'Set NoiseScale')
	parser.add_argument('--usealphaT', action='store_true',
	                help='Use Rankone Correction to do matrix inverse') 
	#parser.add_argument('--WindowSize', dest = 'WindowSize', help = 'Set the Init WindowSize')
	args = parser.parse_args()

	algName = str(args.alg)
	n_users = int(args.userNum)
	NoiseScale = float(args.NoiseScale)
	if args.FutureWeight != None:
		FutureWeight = float(args.FutureWeight)
	else:
		FutureWeight = args.FutureWeight

	RankoneInverse =args.RankoneInverse
	ReturnThreshold = float(args.ReturnThreshold)
	alpha = float(args.alpha)
	usealphaT = args.usealphaT

	#WindowSize = int(WindowSize)
	
	userFilename = os.path.join(sim_files_folder, "r2bandit_users_" + 'featureUniform'+str(n_users)+"+dim--"+str(dimension) +".json")
	#"Run if there is no such file with these settings; if file already exist then comment out the below funciton"
	# we can choose to simulate users every time we run the program or simulate users once, save it to 'sim_files_folder', and keep using it.
	UM = UserManager(dimension, n_users, thetaFunc=featureUniform, betaFunc = featureUniform, argv={'l2_limit':1})
	
	# users = UM.simulateThetafromUsers()
	# UM.saveUsers(users, userFilename, force = False)
	users = UM.loadUsers(userFilename)

	userFeatureDic = {}
	u_dimension = len(users[0].userFeature)
	for i in range(len(users)):
		userFeatureDic[users[i].id] = users[i].userFeature

	atypes = ['smallTheta_smallBeta', 'smallTheta_largeBeta', 'largeTheta_smallBeta', 'largeTheta_largeBeta']
	for i in range(len(users)):
		articlesFilename = os.path.join(sim_files_folder,  'r2bandit_article_' + 'userindex' + str(len(users)) +'_' + str(i) + "articles_"  + 'featureUniform'+str(n_articles)+"+dim"+str(dimension) +".json")
		# Similarly, we can choose to simulate articles every time we run the program or simulate articles once, save it to 'sim_files_folder', and keep using it.
		AM = ArticleManager(dimension, n_articles=n_articles, ArticleGroups = ArticleGroups,
				FeatureFunc=gaussianFeature,  argv={'l2_limit':1}, userFeature_theta = users[i].theta, userFeature_beta= users[i].beta)
		
		articles_small_small, articles_small_large, articles_large_small, articles_large_large = AM.simulateArticlePool_2SetOfFeature()
		
		# #save articles into files for later use
		# AM.saveArticles(articles_small_small, articlesFilename+'_small_small', force=False)
		# AM.saveArticles(articles_small_large, articlesFilename+'_small_large', force=False)
		# AM.saveArticles(articles_large_small, articlesFilename+'_large_small', force=False)
		# AM.saveArticles(articles_large_large, articlesFilename + '_large_large', force=False)
		# #load articles from existing files

		# articles_small_small = AM.loadArticles(articlesFilename + '_small_small')
		# articles_small_large = AM.loadArticles(articlesFilename + '_small_large')
		# articles_large_small = AM.loadArticles(articlesFilename + '_large_small')
		# articles_large_large = AM.loadArticles(articlesFilename + '_large_large')
		articlesList = []
		
		
		articlesList.append(articles_large_small)
		articlesList.append(articles_large_large)
		articlesList.append(articles_small_large)
		articlesList.append(articles_small_small)
		
		users[i].getArticleList(articlesList)
	simExperiment = simulateOnlineData(dimension  = dimension,
						iterations = iterations,
						articletypes = atypes,
						users = users,		
						noise_Click = lambda : np.random.normal(scale = NoiseScale),
						noise_Return = lambda : np.random.normal(scale = NoiseScale),
						batchSize = batchSize,
						type_ = "UniformTheta", 
						signature = AM.signature,
						poolArticleSize = poolSize, noiseLevel = NoiseScale, epsilon = epsilon, FutureWeight = FutureWeight, ReturnThreshold = ReturnThreshold, alpha = alpha, usealphaT = usealphaT)
	print "Starting for ", simExperiment.simulation_signature

	algorithms = {}
	
	if algName == 'all':
		
		algorithms['GLM-UCB'] = reward_GLMUCBAlgorithm(dimension = dimension, alpha = alpha, lambda_ = lambda_,   usealphaT = usealphaT, RankoneInverse = RankoneInverse)
		algorithms['rGLM-UCB'] = return_GLMUCBAlgorithm(dimension = dimension, alpha = alpha, lambda_ = lambda_, ReturnThreshold = ReturnThreshold,  usealphaT = usealphaT, RankoneInverse = RankoneInverse)
		algorithms['r2bandit'] = r2_banditAlgorithm(dimension = dimension, alpha = alpha, lambda_ = lambda_,  FutureWeight = FutureWeight, ReturnThreshold = ReturnThreshold, usealphaT = usealphaT ,RankoneInverse = RankoneInverse)
		algorithms['r2GLMUCB1'] = r2_GLMUCB1Algorithm(dimension = dimension, alpha = alpha, lambda_ = lambda_,  FutureWeight = FutureWeight, ReturnThreshold = ReturnThreshold, usealphaT = usealphaT, RankoneInverse = RankoneInverse)  

	if algName == 'r2bandit':
		algorithms['r2bandit'] = r2_banditAlgorithm(dimension = dimension, alpha = alpha, lambda_ = lambda_, FutureWeight = FutureWeight, usealphaT = usealphaT ,RankoneInverse = RankoneInverse)
	if algName =='GLMUCB':
		algorithms['reward_GLMUCB'] = reward_GLMUCBAlgorithm(dimension = dimension, alpha = alpha, lambda_ = lambda_, FutureWeight = FutureWeight, usealphaT = usealphaT, RankoneInverse = RankoneInverse)
	if algName == 'rGLM-UCB':
		algorithms['return_GLMUCB'] = return_GLMUCBAlgorithm(dimension = dimension, alpha = alpha, lambda_ = lambda_, FutureWeight = FutureWeight, usealphaT = usealphaT, RankoneInverse = RankoneInverse)
	if algName == 'r2GLMUCB1':
		algorithms['r2GLMUCB1'] = r2_GLMUCB1Algorithm(dimension = dimension, alpha = alpha, lambda_ = lambda_,  FutureWeight = FutureWeight, usealphaT = usealphaT)  
	simExperiment.runAlgorithms(algorithms)



	
