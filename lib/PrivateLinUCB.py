import numpy as np
from util_functions import vectorize
from Recommendation import Recommendation
from BaseAlg import BaseAlg
"""Paper: Differentially Private Contextual Bandits
"""


class PrivateLinUCBUserStruct:
    def __init__(self, featureDimension, lambda_, eps, testing_iterations, variance, protect_context, noise_type, init="zero"):
        """In the paper, V = A, u = b
        """
        self.d = featureDimension
        self.M = lambda_ * np.identity(self.d + 1)
        self.V = self.M[:self.d, :self.d]
        self.u = self.M[:self.d, -1]
        self.Vinv = np.linalg.inv(self.V)
        self.Zs = []
        self.T = testing_iterations
        self.eps = eps
        self.variance = variance  # TODO: calculate from epsilon
        self.protect_context = protect_context
        self.noise_type = noise_type.lower()

        if (init == "random"):
            self.UserTheta = np.random.rand(self.d)
        else:
            self.UserTheta = np.zeros(self.d)
        self.time = 1

    def updateParameters(self, articlePicked_FeatureVector, click):
        change = np.outer(articlePicked_FeatureVector,
                          articlePicked_FeatureVector)

        num_partial_sums = bin(self.time).count('1')
        N = np.zeros(shape=(self.d + 1, self.d + 1))
        # Calculate noise
        if self.noise_type == 'gaussian':
            for _ in range(num_partial_sums):
                Z = np.random.normal(scale=np.sqrt(self.variance),
                                    size=(self.d + 1, self.d + 1))
                N += (Z + Z.T) / np.sqrt(2)
        elif self.noise_type == 'laplacian':
            for _ in range(num_partial_sums):
                N += np.random.laplace(scale=np.log(self.T) / self.eps,
                                    size=(self.d + 1, self.d + 1))
        elif self.noise_type == 'wishart':
            raise NotImplementedError()
        else:
            raise NotImplementedError()

        # Update M which encodes previous actions and rewards
        action_and_reward_vector = np.append(
            articlePicked_FeatureVector, click)
        self.M += np.outer(action_and_reward_vector, action_and_reward_vector)

        # Calculate user theta
        if self.protect_context:
            self.V = (self.M + N)[:self.d, :self.d]
            self.u = (self.M + N)[:self.d, -1]
        else:  # ICML
            self.V = (self.M + N)[:self.d, :self.d]
            self.u = self.M[:self.d, -1]
        self.Vinv = np.linalg.inv(self.V)
        self.UserTheta = np.dot(self.Vinv, self.u)
        self.time += 1

    def getProb(self, alpha, article_FeatureVector):
        if alpha == -1:
            alpha = alpha = 0.1 * np.sqrt(np.log(self.time + 1))
        mean = np.dot(self.UserTheta, article_FeatureVector)
        var = np.sqrt(np.dot(np.dot(article_FeatureVector,
                                    self.Vinv),  article_FeatureVector))
        pta = mean + alpha * var
        return pta

    def getProb_plot(self, alpha, article_FeatureVector):
        mean = np.dot(self.UserTheta,  article_FeatureVector)
        var = np.sqrt(np.dot(np.dot(article_FeatureVector,
                                    self.Vinv),  article_FeatureVector))
        pta = mean + alpha * var
        return pta, mean, alpha * var

    def getTheta(self):
        return self.UserTheta

    def getA(self):
        return self.A


class PrivateLinUCBAlgorithm(BaseAlg):
    def __init__(self, arg_dict, init="zero"):  # n is number of users
        BaseAlg.__init__(self, arg_dict)
        self.users = []
        # algorithm have n users, each user has a user structure
        for i in range(arg_dict['n_users']):
            self.users.append(PrivateLinUCBUserStruct(
                arg_dict['dimension'],
                arg_dict['lambda_'],
                arg_dict['eps'],
                arg_dict['T'],
                arg_dict['variance'],
                arg_dict['protect_context'],
                arg_dict['noise_type'],
                init))

    def decide(self, pool_articles, userID, k=1):
        # theta = user_features
        # x = article_features
        # V^-1 = self.users[userId].AInv
        # MEAN
        article_features = np.empty([len(pool_articles), len(
            pool_articles[0].contextFeatureVector[:self.dimension])])
        for i in range(len(pool_articles)):
            article_features[i,
                             :] = pool_articles[i].contextFeatureVector[:self.dimension]
        user_features = self.users[userID].UserTheta
        mean_matrix = np.dot(article_features, user_features)

        # VARIANCE
        var_matrix = np.sqrt(
            np.dot(np.dot(article_features, self.users[userID].Vinv), article_features.T).clip(0))
        pta_matrix = mean_matrix + self.alpha * np.diag(var_matrix)

        pool_positions = np.argsort(pta_matrix)[(k * -1):]
        articles = []
        for i in range(k):
            articles.append(pool_articles[pool_positions[i]])
        return articles

    def getProb(self, pool_articles, userID):
        means = []
        vars = []
        for x in pool_articles:
            x_pta, mean, var = self.users[userID].getProb_plot(
                self.alpha, x.contextFeatureVector[:self.dimension])
            means.append(mean)
            vars.append(var)
        return means, vars

    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].updateParameters(
            articlePicked.contextFeatureVector[:self.dimension], click)

    ##### SHOULD THIS BE CALLED GET COTHETA #####
    def getCoTheta(self, userID):
        return self.users[userID].UserTheta

    def getTheta(self, userID):
        return self.users[userID].UserTheta

    # def getW(self, userID):
    # 	return np.identity(n = len(self.users))
