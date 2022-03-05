import numpy as np
from util_functions import vectorize
import math
import random

class AdaptiveThompsonUserStruct:
    def __init__(self, featureDimension, AdTS_Window, AdTS_CheckInter):
        self.B = np.identity(n = featureDimension)
        self.d = featureDimension
        self.mu = np.asarray([0] * self.d)
        self.f = np.asarray([0] * self.d)
        # self.epsilon = 0.44
        #self.windowSize = 100

        self.sigma = 0.1
        self.epsilon = 0.1
        self.windowSize = AdTS_Window
        self.history = []
        self.clicks = []
        self.time = 0
        self.distances = []
        self.changes = [0]

    def updateParameters(self, articlePicked_FeatureVector, click):
        #print(self.changes)
        self.time += 1
        self.history.append(articlePicked_FeatureVector)
        self.clicks.append(click)

        self.B = self.B + np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector.T)
        self.f = self.f + np.multiply(articlePicked_FeatureVector, click)
        self.mu = np.dot(np.linalg.inv(self.B), self.f)

        change = False
        if self.time % 50 == 0:
            print('trying it out')
            change = self.detectChange()

        if change:
            self.changes.append(self.changes[-1] + self.time)
            self.B = np.identity(n = self.d)
            self.mu = np.asarray([0] * self.d)
            self.f = np.asarray([0] * self.d)
            self.history = []
            self.clicks = []
            self.time = 0
            self.distances = []
            print("A change has been detected!")

        #print(self.mu)

    def getWindowInfo(self, w_start, w_end):
        cov = np.identity(n = self.d)
        mean = np.asarray([0] * self.d)
        local_f = np.asarray([0] * self.d)

        for i in range(w_start, w_end):
            cov = cov + np.outer(self.history[i], self.history[i].T)
            local_f = local_f + np.multiply(self.history[i], self.clicks[i])
            mean = np.dot(np.linalg.inv(cov), local_f)

        return mean, cov

    def detectChange(self):
        if self.time >= self.windowSize * 2:
            w1_mean, w1_cov = self.getWindowInfo(self.time - 2*self.windowSize, self.time - self.windowSize)
            w2_mean, w2_cov = self.getWindowInfo(self.time - self.windowSize, self.time)

            cov_avg = (w1_cov + w2_cov) / 2
            first = np.inner((w1_mean - w2_mean).T, cov_avg)

            distance = np.inner(first, w1_mean - w2_mean) 

            self.distances.append(distance)
            #print(len(self.distances))
            if len(self.distances) > 5:
                cum_sums = self.getCumSum(self.distances)

                s_diff = max(cum_sums) - min(cum_sums)

                #lets do 1000 samples
                sampleNum = 1000
                bootstraps = 0
                for i in range(sampleNum):

                    sample = random.sample(self.distances, len(self.distances))
                    sample_cum_sum = self.getCumSum(sample)
                    sample_diff = max(sample_cum_sum) - min(sample_cum_sum)
                    if sample_diff < s_diff:
                        bootstraps += 1

                print("Bootstraps: " + str(bootstraps))
                if bootstraps > 0.95 * sampleNum:
                    return True
                else:
                    return False
        else:
            return False

    def getCumSum(self, values):
        avg = 0.0
        for i in range(len(values)):
            avg += values[i]
        avg = avg / len(values)

        cum_sums = [0]
        for i in range(1, len(values)):
            cum_sums.append(cum_sums[i-1] + (values[i] - avg))

        return cum_sums

    def getProb(self, article_FeatureVector, v):
        #v = math.sqrt((24/self.epsilon)*self.d*math.log(1/self.sigma))
        # not sure why the calculated v is terrible but this one is much better
        v = v
        sampled_mu = np.random.multivariate_normal(mean=self.mu, cov=v*v*np.linalg.inv(self.B))
        return np.dot(article_FeatureVector.T, sampled_mu)

class AdaptiveThompson:
    def __init__(self, dimension, AdTS_Window = 500, AdTS_CheckInter = 50, sample_num = 1000, v = 0.4):  # n is number of users
        self.users = {}
        self.dimension = dimension
        self.AdTS_Window = AdTS_Window
        self.AdTS_CheckInter = AdTS_CheckInter
        self.v = v

        self.CanEstimateUserPreference = True
        self.CanEstimateCoUserPreference = False
        self.CanEstimateUserCluster= False
        self.CanEstimateW = False
        self.CanEstimateV = False

        self.CanEstimateBeta = False

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = AdaptiveThompsonUserStruct(self.dimension, self.AdTS_Window, self.AdTS_CheckInter)
        maxPTA = float('-inf')
        articlePicked = None

        for x in pool_articles:
            x_pta = self.users[userID].getProb(x.contextFeatureVector[:self.dimension], self.v)
            if maxPTA < x_pta:
                articlePicked = x
                maxPTA = x_pta

        return articlePicked

    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].updateParameters(articlePicked.contextFeatureVector[:self.dimension], click)

    def getTheta(self, userID):
        return self.users[userID].mu
