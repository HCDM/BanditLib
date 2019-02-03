import numpy as np
from util_functions import vectorize
from Recommendation import Recommendation
from BaseAlg import BaseAlg
from scipy.stats import wishart
"""Paper: Differentially Private Contextual Bandits
"""

class PartialSum:
    def __init__(self, start, size, noise):
        self.start = start
        self.size = size
        self.noise = noise

class PrivateLinUCBUserStruct:
    def __init__(self, featureDimension, lambda_, hyperparameters, protect_context, noise_type, init="zero"):
        """In the paper, V = A, u = b
        """
        self.d = featureDimension
        self.M = lambda_ * np.identity(self.d + 1)
        self.V = self.M[:self.d, :self.d]
        self.u = self.M[:self.d, -1]
        self.Vinv = np.linalg.inv(self.V)

        self.alpha = hyperparameters['alpha']
        self.T = hyperparameters['T']
        self.eps = hyperparameters['eps']
        self.delta = hyperparameters['delta']

        self.protect_context = protect_context
        self.noise_type = noise_type.lower()
        self.noise = {}

        if init == "random":
            self.UserTheta = np.random.rand(self.d)
        else:
            self.UserTheta = np.zeros(self.d)
        self.time = 1

    def generate_noise(self):
        noise = np.zeros(shape=(self.d + 1, self.d + 1))
        if self.noise_type == 'gaussian':
            # Does NOT preserve (eps, delta)-DP
            #   Rather preserves (eps / sqrt(8mln(2/delta)), delta / 2m)-DP
            # For testing, try T = 200, delta = 5.6, eps = 75
            m = int(np.ceil(np.log2(self.T))) + 1  # max_number_of_p_sums
            max_feature_vector_l2_norm = 1  # assumed, but noise added to reward might be problematic
            max_reward_l1_norm = 1  # assumed
            L_tilde = np.sqrt(max_feature_vector_l2_norm**2 + max_reward_l1_norm**2)
            variance = 16 * m * L_tilde**4 * np.log(4 / self.delta)**2 / self.eps**2
            Z = np.random.normal(scale=np.sqrt(variance),
                                size=(self.d + 1, self.d + 1))
            upsilon = np.sqrt(2 * m * variance) * (4 * np.sqrt(self.d) + 2 * np.log(2 * self.T / self.alpha))
            noise = (Z + Z.T) / np.sqrt(2) + 2 * upsilon * np.identity(self.d + 1)
        elif self.noise_type == 'laplacian':
            #  Paper: Differential privacy under continual observation says add Lap(logT/e)
            #       Lap(scale=x) + Lap(scale=x) has a lower variance than Lap(scale=2x)
            #           So why not just add Lap(scale=1/eps) to each reward and then sum them up?
            #           Wouldn't that be more effective than adding one big lump sum of Lap(scale=T/eps)?
            #       Is noise resampled every time for each node or is it sampled once and stored?
            #           Does it make a difference?
            noise = np.random.laplace(scale=np.log(self.T) / self.eps,
                                size=(self.d + 1, self.d + 1))
        elif self.noise_type == 'wishart':
            # Does NOT preserve (eps, delta)-DP
            #   Rather preserves (eps / sqrt(8mln(2/delta)), delta / 2m)-DP
            m = int(np.ceil(np.log2(self.T))) + 1  # max_number_of_p_sums
            max_feature_vector_l2_norm = 1  # assumed, but noise added to reward might be problematic
            max_reward_l1_norm = 1  # assumed
            L_tilde = np.sqrt(max_feature_vector_l2_norm**2 + max_reward_l1_norm**2)
            df = int(self.d + 1 + np.ceil(224 * m * self.eps**-2 * np.log(8 * m / self.delta) * np.log(2 / self.delta)))
            scale = L_tilde * np.identity(self.d + 1)
            noise = wishart.rvs(df, scale)
        else:
            raise NotImplementedError()
        return noise

    def consolidate_partial_sums(self, time):
        prev_p_sum_time = self.noise[time].start - self.noise[time].size
        if prev_p_sum_time in self.noise:
            if self.noise[time].size == self.noise[prev_p_sum_time].size:
                self.noise[prev_p_sum_time] = PartialSum(prev_p_sum_time, self.noise[time].size*2, self.generate_noise())
                del self.noise[time]
                self.consolidate_partial_sums(prev_p_sum_time)

    def update_noise_tree(self):
        self.noise[self.time] = PartialSum(self.time, 1, self.generate_noise())
        self.consolidate_partial_sums(self.time)

    def get_total_noise(self):
        N = np.zeros(shape=(self.d + 1, self.d + 1))
        for p_sum in self.noise.values():
            N += p_sum.noise
        return N

    def update_user_theta(self, N):
        self.V = (self.M + N)[:self.d, :self.d]
        if self.protect_context:  # NIPS
            self.u = (self.M + N)[:self.d, -1]
        else:  # ICML
            self.u = self.M[:self.d, -1]
        self.Vinv = np.linalg.inv(self.V)
        self.UserTheta = np.dot(self.Vinv, self.u)

    def updateParameters(self, articlePicked_FeatureVector, click):
        self.update_noise_tree()
        N = self.get_total_noise()
        action_and_reward_vector = np.append(articlePicked_FeatureVector, click)
        self.M += np.outer(action_and_reward_vector, action_and_reward_vector)
        self.update_user_theta(N)
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
        hyperparameters = {
            'alpha': arg_dict['alpha'],
            'eps': arg_dict['eps'],
            'delta': arg_dict['delta'],
            'T': arg_dict['T'],
        }
        # algorithm have n users, each user has a user structure
        for i in range(arg_dict['n_users']):
            self.users.append(PrivateLinUCBUserStruct(
                arg_dict['dimension'],
                arg_dict['lambda_'],
                hyperparameters,
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
