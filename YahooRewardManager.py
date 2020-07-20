import sys
from conf import *     # it saves the address of data stored and where to save the data produced by algorithms
import argparse # For argument parsing
import time
import re             # regular expression library
from random import random, choice,shuffle     # for random strategy
from operator import itemgetter
import datetime
import numpy as np     
import sys
from scipy.sparse import csgraph
from scipy.spatial import distance
from YahooExp_util_functions import *
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt


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
from lib.MLP import * 

import warnings

# structure to save data from random strategy as mentioned in LiHongs paper
class randomStruct:
    def __init__(self):
        self.learn_stats = articleAccess()


class Article():    
    def __init__(self, aid, FV=None):
        self.id = aid
        self.featureVector = FV
        self.contextFeatureVector = FV


class YahooRewardManager():
	def __init__(self, arg_dict):
                for key in arg_dict:
                        setattr(self, key, arg_dict[key])
                self.nClusters = 100 

        def runAlgorithms(self, algorithms, diffLists):
		warnings.filterwarnings('ignore')
		# regularly print stuff to see if everything is going alright.
		# this function is inside main so that it shares variables with main and I dont wanna have large number of function arguments
		def printWrite():
			randomLearnCTR =  articles_random.learn_stats.cumulative_CTR_list[-1]
			for alg_name, alg in algorithms.items():
			    print(alg_name, algCTR, algCTR/randomLearnCTR, alg.learn_stats.accesses, alg.learn_stats.clicks)
		        print("")
			# write to file
			#save_to_file(fileNameWrite, recordedStats, tim) 


		def calculateStat():
			if click:        
			    for article in currentArticles:
				if article == article_chosen:
				    articleTruePositve[article_chosen] +=1
				else:
				    articleTrueNegative[article] +=1                
			else:
			    for article in currentArticles:
				if article == article_chosen:
				    articleFalsePositive[article_chosen] +=1

				   

		tstart = datetime.datetime.now()
		timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M')     # the current data time
		dataDays = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
#		dataDays = ['01', '02']
#		dataDays = ['01']
#		fileSig = str(algName)+str(clusterNum)+ 'SP'+ str(SparsityLevel)+algName
		batchSize = 100000
		statBatchSize = 200000                            # size of one batch


		totalObservations = 0
		articleTruePositve = {}
		articleFalseNegative = {}

		articleTrueNegative = {}
		articleFalsePositive = {}

		articles_random = randomStruct()

		tsave = 60*60*47 # Time interval for saving model.

		userNum = 160 
		written = False
	
		alg_accesses = {}
		alg_clicks = {}
		alg_CTR = {}
		for alg_name, alg in algorithms.items():
			#if not args.load:
		        alg.learn_stats = articleAccess()
		clusterNum = 160
		
		fileNameWriteCluster = os.path.join(Kmeansdata_address, '10kmeans_model'+str(clusterNum)+ '.dat')
		userFeatureVectors = getClusters(fileNameWriteCluster)
		#Yahoo dataset does not come with user labels
		# We group users my kmeans clustering
		
		total_observations = 0
		for dataDay in dataDays:
			fileName = Yahoo_address + "/ydata-fp-td-clicks-v1_0.200905" + dataDay + '.' + str(userNum) +'.userID'
			
			with open(fileName, 'r') as f:
			    # reading file line ie observations unning one at a time
			    for i, line in enumerate(f, 1):
				tim, article_chosen, click, currentUserID, pool_articles = parseLine_ID(line)

				#-----------------------------Pick an article (CoLinUCB, LinUCB, Random)-------------------------
				articlePool = []    
				currentArticles = []            
				for article in pool_articles:
				    article_featureVector = np.asarray(article[1:6])
				    user_featureVector = userFeatureVectors[currentUserID] 
				    combined_featureVector = np.outer(article_featureVector, user_featureVector).flatten()
				    #combined_featureVector= np.asarray(article[1:6])
				    if len(article_featureVector) == 5:
					    article_id = int(article[0])
					    articlePool.append(Article(article_id, combined_featureVector))
					    currentArticles.append(article_id)  
				shuffle(articlePool)	
				for article in currentArticles:
				    if article not in articleTruePositve:
					articleTruePositve[article] = 0
					articleTrueNegative[article] = 0
					articleFalsePositive[article] = 0
					articleFalseNegative[article] = 0
				
				# article picked by random strategy
				articles_random.learn_stats.addrecord(click)
				for alg_name, alg in algorithms.items():
				    pickedArticle = alg.createRecommendation(articlePool, currentUserID, self.k)
				    pickedArticle = pickedArticle.articles[0]
				    # reward = getReward(userID, pickedArticle) 
				    if (pickedArticle.id == article_chosen):
					alg.learn_stats.addrecord(click)
					alg.updateParameters(pickedArticle, click, currentUserID)
					calculateStat()

				total_observations += 1
				if total_observations % 2000 == 0:
				    articles_random.learn_stats.updateCTR()
				    for alg in algorithms.values():
					algCTR = alg.learn_stats.updateCTR()
			    #print stuff to screen and save parameters to file when the Yahoo! dataset file ends
			    print("Day " + dataDay + " Time Elapsed: " + str(datetime.datetime.now()-tstart))     # the current data time
			    plot_results(algorithms, articles_random)
			    printWrite()


def plot_results(algorithms, random):
	num_data_points = len(random.learn_stats.cumulative_CTR_list)
	tim_ = [i for i in range(num_data_points)]
	random_CTR_list = random.learn_stats.cumulative_CTR_list

	f, axa = plt.subplots(1, sharex=True)
	for alg_name, alg in algorithms.items():
		alg_CTR_list = alg.learn_stats.cumulative_CTR_list
		alg_normalized_CTR = [alg_CTR/rand_CTR for alg_CTR, rand_CTR in zip(alg_CTR_list, random_CTR_list)]
		axa.plot(tim_, alg_normalized_CTR, label=alg_name)
	plt.xlabel('time')
	plt.ylabel('CTR-Ratio')
	plt.legend(loc = 'lower right')
	plt.title('Yahoo 160 Users')
        filename = datetime.datetime.now().strftime("batch_results/yahoo_2layer/MLPMany5x25_lr01_1sample")

	plt.savefig(filename)





