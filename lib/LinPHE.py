import numpy as np
from BaseAlg import BaseAlg
from scipy.stats import bernoulli


class LinPHEUserStruct:
        def __init__(self, featureDimension, lambda_, a):
                self.d = featureDimension
                self.lamda_ = lambda_
                self.a = a
                self.f = np.zeros(self.d)
                self.B = np.zeros((self.d, self.d))
                self.theta = np.zeros(self.d)
                self.G_additive = lambda_*(a+1)*np.identity(self.d)

        def updateParameters(self, articlePicked_featureVector, click):
                self.B += (self.a+1) * np.outer(articlePicked_featureVector, articlePicked_featureVector)
                G = self.B + self.G_additive

                perturbed_reward = .1*np.random.binomial(self.a, .5)
                self.f += articlePicked_featureVector * (click+ perturbed_reward)
                self.theta = np.dot(np.linalg.inv(G), self.f)

        def getProb(self, article_featureVector):
                return np.dot(self.theta, article_featureVector)


class LinPHEAlgorithm(BaseAlg):
        def __init__(self, arg_dict):
                BaseAlg.__init__(self, arg_dict)

                self.users = []
                for i in range(arg_dict['n_users']):
                        self.users.append(LinPHEUserStruct(arg_dict['dimension'], arg_dict['lambda_'], arg_dict['a']))                

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
                return self.users[userID].theta 
