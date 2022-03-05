import numpy as np
from util_functions import vectorize
# from .LinUCB import LinUCBUserStruct, LinUCB
import math
import random
import copy
from scipy.special import erfinv

class LinUCBUserStruct:
	def __init__(self, featureDimension, alpha, lambda_,  NoiseScale, init="zero"):
		self.d = featureDimension
		self.A = lambda_*np.identity(n = self.d)
		self.b = np.zeros(self.d)
		self.AInv = np.linalg.inv(self.A)
		self.NoiseScale = NoiseScale
		if (init=="random"):
			self.UserTheta = np.random.rand(self.d)
		else:
			self.UserTheta = np.zeros(self.d)
		self.time = 0

	def updateParameters(self, articlePicked_FeatureVector, click):
		self.A += np.outer(articlePicked_FeatureVector,articlePicked_FeatureVector)
		self.b += articlePicked_FeatureVector*click
		self.AInv = np.linalg.inv(self.A)
		self.UserTheta = np.dot(self.AInv, self.b)
		self.time += 1

	def getTheta(self):
		return self.UserTheta
	
	def getA(self):
		return self.A

	def getProb(self, alpha, article_FeatureVector):
		if alpha == -1:
			alpha = alpha = 0.1*np.sqrt(np.log(self.time+1))
		mean = np.dot(self.UserTheta,  article_FeatureVector)
		var = np.sqrt(np.dot(np.dot(article_FeatureVector, self.AInv),  article_FeatureVector))
		#self.alpha_t = self.NoiseScale *np.sqrt(np.log(np.linalg.det(self.A)/float(self.sigma * self.lambda_) )) + np.sqrt(self.lambda_)
		pta = mean + alpha * var
		return pta

	def getProbInfo(self, alpha, article_FeatureVector):
		if alpha == -1:
			alpha = alpha = 0.1*np.sqrt(np.log(self.time+1))
		mean = np.dot(self.UserTheta,  article_FeatureVector)
		var = np.sqrt(np.dot(np.dot(article_FeatureVector, self.AInv),  article_FeatureVector))
		self.alpha_t = self.NoiseScale *np.sqrt(np.log(np.linalg.det(self.A)/float(self.sigma * self.lambda_) )) + np.sqrt(self.lambda_)
		#pta = mean + alpha * var
		return {'mean':mean, 'var':var, 'alpha':alpha, 'alpha_t':self.alpha_t}

	def getProb_plot(self, alpha, article_FeatureVector):
		mean = np.dot(self.UserTheta,  article_FeatureVector)
		var = np.sqrt(np.dot(np.dot(article_FeatureVector, self.AInv),  article_FeatureVector))
		pta = mean + alpha * var
		return pta, mean, alpha * var

class MASTER:
    def __init__(self, createTime):
        self.SLAVEs = []
        self.activeLinUCBs = []
        self.inactiveLinUCBs = []
        self.ModelSelection = []
        self.newUCBs = []
        self.discardUCBs = []
        self.discardUCBIDs = []
        self.ModelSelection = []
        self.ModelUCB = []
        self.SwitchPoints = []
        self.SwitchPoints_pool = []
        self.ActiveLinUCBNum = []
        self.selectedAlgList = []
        self.selectedAlg = None
        self.time = createTime
        self.createTime = createTime

class SlaveLinUCBStruct(LinUCBUserStruct):
    def __init__(self, featureDimension, alpha, lambda_ , createTime, NoiseScale, delta_1, delta_2,  init="random"):
        LinUCBUserStruct.__init__(self, featureDimension = featureDimension, alpha = alpha, lambda_ = lambda_, NoiseScale = NoiseScale, init = init)
        self.alpha = alpha
        self.fail = 0.0
        self.success = 0.0
        self.failList = []
        self.plays = 0.0
        self.updates = 0.0
        self.emp_loss = 1.0
        self.createTime = createTime
        self.time =createTime
        self.lambda_  = lambda_
        self.NoiseScale = NoiseScale
        self.sigma = 1.e-2   #Used in the high probability bound, i.e, with probability at least (1 - sigma) the confidence bound. So sigma should be very small
        self.model_id = createTime
        self.ratio = 0.0

        self.badness = 0.0
        self.badness_CB = 0.5
        self.active = True
        self.update_num = 0.0

        self.delta_1 = delta_1
        self.delta_2 = delta_2

        self.alpha_t = self.NoiseScale ** 2 * np.sqrt(
            self.d * np.log(1 + self.update_num / (self.d * self.lambda_)) + 2 * np.log(1 / self.delta_1)) + np.sqrt(
            self.lambda_)

        self.history = []
        self.clicks = []
       
    def getBadness(self, ObservationInterval):
        if len(self.failList) < ObservationInterval:
            ObservationNum = len(self.failList) +1.0
            self.badness = sum(self.failList)/ObservationNum 
        else:
            ObservationNum = ObservationInterval + 1.0
            self.badness = sum(self.failList[-ObservationInterval :])/ObservationNum
            
        self.badness_CB =  math.sqrt(math.log(1.0/self.delta_2)/(2.0*ObservationNum))

        return self.badness, self.badness_CB
    
    def decide(self, pool_articles):
        maxPTA = float('-inf')
        articlePicked = None
        ## use alpha or use alpha_t
        for x in pool_articles:
            x_pta = self.getProb(self.alpha, x.contextFeatureVector[:self.d])
            # pick article with highest Prob
            if maxPTA < x_pta:
                articlePicked = x
                maxPTA = x_pta

        return articlePicked
    def updateParameters(self, articlePicked_FeatureVector, click):
        self.history.append(articlePicked_FeatureVector)
        self.clicks.append(click)

        self.A += np.outer(articlePicked_FeatureVector,articlePicked_FeatureVector)
        self.b += articlePicked_FeatureVector*click
        self.AInv = np.linalg.inv(self.A)
        self.UserTheta = np.dot(self.AInv, self.b)
        self.time += 1
        #self.DetA = np.linalg.det(self.A)
        self.update_num +=1.0

        self.alpha_t = self.NoiseScale ** 2 * np.sqrt(
            self.d * np.log(1 + self.update_num / (self.d * self.lambda_)) + 2 * np.log(1 / self.delta_1)) + np.sqrt(
            self.lambda_)



    def getSlavePredictionInfo(self, alpha, article_FeatureVector):
        if alpha == -1:
            alpha = alpha = 0.1*np.sqrt(np.log(self.time+1))
        mean = np.dot(self.UserTheta,  article_FeatureVector)
        var = np.sqrt(np.dot(np.dot(article_FeatureVector, self.AInv),  article_FeatureVector))
        # self.alpha_t =  self.NoiseScale*np.sqrt(self.d* np.log( (self.lambda_ + self.update_num)/float(self.sigma * self.lambda_) )) + np.sqrt(self.lambda_)
        self.alpha_t = self.NoiseScale ** 2 * np.sqrt(
            self.d * np.log(1 + self.update_num / (self.d * self.lambda_)) + 2 * np.log(1 / self.delta_1)) + np.sqrt(
            self.lambda_)
        #print 'alpha',  alpha, self.alpha_t
        return {'mean':mean, 'var':var, 'alpha':alpha, 'alpha_t': self.alpha_t}


class dLinUCB:
    def __init__(self, dimension, alpha, lambda_, NoiseScale, tau, delta_1 = 1e-2 , delta_2 = 1e-1, tilde_delta_1 =1e-2, eta=0.3):  # n is number of users
        self.dimension = dimension
        self.alpha = alpha
        self.lambda_ = lambda_
        self.global_time = 0
        self.users = {}
        self.NoiseScale = NoiseScale
        self.ObservationInterval = tau         #Size of the slding window
        self.delta_1 = delta_1
        self.delta_2 = delta_2
        self.tilde_delta_1 = tilde_delta_1  #self.tilde_delta_1 is between 0 and self.delta_1
        self.eta = eta
        self.CanEstimateUserPreference = True
        self.CanEstimateUserCluster= False
        
    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = MASTER(self.global_time)
            self.users[userID].SLAVEs.append(SlaveLinUCBStruct(featureDimension = self.dimension, alpha = self.alpha, lambda_ = self.lambda_, createTime= self.global_time, NoiseScale = self.NoiseScale, delta_1 = self.delta_1, delta_2 = self.delta_2))
            self.users[userID].newUCBs.append(self.users[userID].time)
        
        #Select a slave model according to the LCB of badness
        min_badness = float('+inf')
        min_alg = None
        SLAVEs = self.users[userID].SLAVEs
        pool_id = []
        for alg in SLAVEs:
            pool_id.append(alg.createTime)
            badness_LCB = alg.badness - math.sqrt(math.log(self.ObservationInterval)) * alg.badness_CB
            if badness_LCB < min_badness:
                min_badness = badness_LCB
                min_alg = alg
    
        # print("number of slave for user {}".format(len(self.users[userID].SLAVEs)))

        index = SLAVEs.index(min_alg)
        self.users[userID].SLAVEs[index].plays += 1
        self.users[userID].selectedAlg = min_alg

        #Record useful inforamtion about which slave model the algorithm selected and the switch points of different slave models 
        self.users[userID].SLAVEs[index].plays +=1
        self.users[userID].ModelSelection.append(self.users[userID].SLAVEs[index].createTime)
        if len(self.users[userID].ModelSelection) >1:
            if self.users[userID].ModelSelection[-1] != self.users[userID].ModelSelection[-2]:
                self.users[userID].SwitchPoints.append(len(self.users[userID].ModelSelection))
                self.users[userID].SwitchPoints_pool.append(pool_id)
                self.users[userID].selectedAlgList.append(self.users[userID].ModelSelection[-1])
        self.users[userID].ActiveLinUCBNum.append(len(self.users[userID].SLAVEs))

        #Select an arm according to each slave model's UCB
        return min_alg.decide(pool_articles)

    def updateParameters(self, articlePicked, click, userID):
        self.global_time +=1
        self.users[userID].time += 1
        # Create new slave model as long as the reward is beyond the reward confidence bound.
        # we only have out of bounds if EVERY one is out of bounds
        # i.e. NONE of them are in bounds
        # so if we find one that is in Bounds, we should set it to FALSE
        CreateNewFlag = True
        good_alg = []
        algs_to_remove = []
        for alg in self.users[userID].SLAVEs:
            data = alg.getSlavePredictionInfo(alg.alpha, articlePicked.featureVector)
            alg.TotalrewardDiff = abs(data['mean'] - click)
            # print("featureVector {}".format(articlePicked.featureVector))
            # print("data['mean'] {}".format(data['mean']))
            # print("click {}".format(click))
            # print("data['var']: {}".format(data['var']))
            # print("data['alpha']*data['var']+ self.eta: {}".format(data['var']*data['alpha']+ self.eta))
            # print("alg.TotalrewardDiff {}".format(alg.TotalrewardDiff))
            if alg.TotalrewardDiff <= data['var']*data['alpha'] + self.eta:  #0.5
                alg.success += 1
                good_alg.append(alg)
                alg.failList.append(0)
            else:
                # assert False
                alg.fail += 1
                alg.failList.append(1)
            alg_badness, alg_badness_CB = alg.getBadness(self.ObservationInterval)
            if alg_badness <= alg_badness_CB + self.tilde_delta_1:
                CreateNewFlag = False
            elif alg_badness > alg_badness_CB + self.delta_1:
                algs_to_remove.append(alg)

        #Discard bad slave models    
        for alg in algs_to_remove:
            # print("===========================discard model")
            # assert False
            self.users[userID].discardUCBs.append(self.users[userID].time)
            self.users[userID].discardUCBIDs.append(alg.createTime)
            self.users[userID].SLAVEs.remove(alg)
        
        #Create new slave model if necessary
        if CreateNewFlag or len(self.users[userID].SLAVEs) ==0:
            # print("==========================create model")
            self.users[userID].newUCBs.append(self.users[userID].time)
            new_ucb = SlaveLinUCBStruct(featureDimension = self.dimension, alpha = self.alpha, lambda_ = self.lambda_,  createTime = self.users[userID].time, NoiseScale = self.NoiseScale , delta_1 = self.delta_1, delta_2 = self.delta_2)
            self.users[userID].SLAVEs.append(new_ucb)

        #Update good slave models
        for alg in self.users[userID].SLAVEs:
            if alg in good_alg:
                alg.updateParameters(articlePicked.featureVector, click)
                alg.updates +=1

    def getTheta(self, userID):
        return self.users[userID].selectedAlg.UserTheta


