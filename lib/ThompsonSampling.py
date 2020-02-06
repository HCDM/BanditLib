# Working on adding thompson sampling to the library
import numpy as np
import math
from BaseAlg import BaseAlg


class ThompsonSamplingUserStruct:
    def __init__(self, featureDimension, lambda_, v_squared):
        self.d = featureDimension
        self.B = lambda_*np.identity(self.d)
        self.v_squared = v_squared
        self.f = np.zeros(self.d)
        self.theta_hat = np.zeros(self.d)
        self.theta_estimate = np.random.multivariate_normal(self.theta_hat, self.v_squared*np.linalg.inv(self.B))

    def updateParameters(self, articlePicked_FeatureVector, click):
        self.B += np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector)
        self.f += articlePicked_FeatureVector*click
        self.theta_hat = np.dot(np.linalg.inv(self.B), self.f)
        self.theta_estimate = np.random.multivariate_normal(self.theta_hat, self.v_squared*np.linalg.inv(self.B))

    def getProb(self, article_FeatureVector):
        return np.dot(self.theta_estimate, article_FeatureVector)  


class ThompsonSamplingAlgorithm(BaseAlg):
    def __init__(self, arg_dict):
        BaseAlg.__init__(self, arg_dict)
        self.users = [] 
        v_squared = self.get_v_squared(arg_dict['R'], arg_dict['epsilon'], arg_dict['delta'])
        print(v_squared)

        for _ in range(arg_dict['n_users']):
            self.users.append(ThompsonSamplingUserStruct(arg_dict['dimension'], arg_dict['lambda_'], v_squared))

    def decide(self, pool_articles, userID, k=1):
        maxPTA = float('-inf')
        articlePicked = None
        
        for x in pool_articles:
           x_pta = self.users[userID].getProb(x.contextFeatureVector[:self.dimension])
           if maxPTA < x_pta:
               articlePicked = x 
               maxPTA = x_pta 
        return [articlePicked]

    def updateParameters(self, article_picked, click, userID):
        self.users[userID].updateParameters(article_picked.contextFeatureVector[:self.dimension], click)
        
    def getTheta(self, userID):
        return self.users[userID].theta_hat
 
    def get_v_squared(self, R, epsilon, delta):
        v = R * math.sqrt(24*self.dimension/epsilon * math.log(1/delta))
        return v ** 2
