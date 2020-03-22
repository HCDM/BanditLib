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
        self.mu_hat = np.zeros(self.d)
        self.mu_estimate = np.random.multivariate_normal(self.mu_hat, self.v_squared*np.linalg.inv(self.B))
        self.time = 0

    def updateParameters(self, articlePicked_FeatureVector, click):
        self.B += np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector)
        self.f += articlePicked_FeatureVector*click
        self.mu_hat = np.dot(np.linalg.inv(self.B), self.f)
        self.mu_estimate = np.random.multivariate_normal(self.mu_hat, self.v_squared*np.linalg.inv(self.B))
        self.time += 1

    def getProb(self, article_FeatureVector):
        prob =  np.dot(self.mu_estimate, article_FeatureVector)  
        return np.dot(self.mu_hat, article_FeatureVector)  
 


class ThompsonSamplingAlgorithm(BaseAlg):
    def __init__(self, arg_dict):
        BaseAlg.__init__(self, arg_dict)
        self.users = [] 
        self.R = 0 
        v_squared = self.get_v_squared()
        print v_squared

        for _ in range(arg_dict['n_users']):
            self.users.append(ThompsonSamplingUserStruct(arg_dict['dimension'], arg_dict['lambda_'], 1))

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
        return self.users[userID].mu_hat
 
    def get_v_squared(self):
        delta = .1 # 1 - delta = confidece_paramater
        epsilon = .1 # Ideal error rate
        v = math.sqrt(24*self.dimension/epsilon * math.log(1/delta))
        return v ** 2

def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom         
