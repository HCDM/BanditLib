from __future__ import division  # enforce float division with `/`
import numpy as np
from BaseAlg import BaseAlg
from scipy.stats import wishart

from .partial_sums import *
"""Paper: Differentially Private Contextual Bandits
"""


class PrivateLinUCBNoiseGenerator:
    def __init__(self, eps, delta, T, alpha, context_dimension, noise_type, is_theta_level):
        self.eps = eps
        self.delta = delta
        self.T = T
        self.alpha = alpha
        self.d = context_dimension
        self.noise_type = noise_type
        self.is_theta_level = is_theta_level
        self.max_feature_vector_L2 = 1  # an assumed constant
        self.max_reward_L1 = 1  # an assumed constant

    def zeros(self):
        """Generate a matrix of zeros."""
        return np.zeros(shape=(self.d + 1, self.d + 1))

    def laplacian(self, denominator, sens = 1):
        """Generate a matrix with noise sampled from a laplacian.

        The scale of the laplacian noise is 1 divided by the provided denominator.
        The size of the matrix is (context dimension + 1, context dimension + 1).

        Args:
            denominator (float): defines scale of noise as 1 / denominator
            sens (float): sensitivity

        Returns:
            A numpy matrix of noise sampled from the computed laplacian distribution

        """
        if self.is_theta_level:
            return np.random.laplace(scale=1 / (denominator * sens), size=(self.d + 1, self.d + 1))
        else:
            return np.random.laplace(scale=1 / denominator, size=(self.d + 1, self.d + 1))

    def laplacian_tree(self, eps, T):
        """Generate a matrix with noise sampled from a laplacian for tree-based algorithm.

        The scale of the laplacian noise is log2(T) / epsilon. The size of the matrix is
        (context dimension + 1, context dimension + 1).

        Returns:
            A numpy matrix of noise sampled from the computed laplacian distribution

        """
        return self.laplacian(eps / np.log2(T))

    def gaussian_tree(self, eps, delta, T, shifted=True):
        """Generate a symmetric matrix with noise sampled from a gaussian for tree-based algorithm.

        The variance of the gaussian distribution is calculated based on the setting
        variables - epsilon, delta, alpha, and T. The size of the matrix is
        (context dimension + 1, context dimension + 1). If shifted, then the matrix
        will maintain its positive semi-definiteness throughout the algorithm.

        Args:
            shifted (bool): if should maintain positive semi-definiteness of matrix

        Returns:
            A numpy matrix of noise sampled from the computed gaussian distribution
        """
        L_tilde = np.sqrt(self.max_feature_vector_L2**2 +
                          self.max_reward_L1**2)

        m = int(np.ceil(np.log2(T))) + 1  # max number of p sums
        variance = 16 * m * L_tilde**4 * \
            np.log(4 / delta)**2 / eps**2
        Z = np.random.normal(scale=np.sqrt(variance),
                             size=(self.d + 1, self.d + 1))
        sym_Z = (Z + Z.T) / np.sqrt(2)
        if not shifted:
            return sym_Z

        upsilon = np.sqrt(2 * m * variance) * \
            (4 * np.sqrt(self.d) + 2 * np.log(2 * T / self.alpha))
        shifted_Z = sym_Z + 2 * upsilon * np.identity(self.d + 1)
        return shifted_Z

    def wishart_tree(self, eps, delta, T, shifted=True):
        """Generate a matrix with noise sampled from a wishart distribution for tree-based algorithm.

        The degrees of freedom and scale of the wishart distribution is calculated based on the setting
        variables - epsilon, delta, alpha, and T. The size of the matrix is (context dimension + 1,
        context dimension + 1). If shifted, then the matrix will be shifted down by a computed factor.

        Args:
            shifted (bool): if noise matrix should be shifted down

        Returns:
            A numpy matrix of noise sampled from the computed wishart distribution
        """
        m = int(np.ceil(np.log2(T))) + 1  # max_number_of_p_sums
        L_tilde = np.sqrt(self.max_feature_vector_L2**2 +
                          self.max_reward_L1**2)
        df = int(self.d + 1 +
                 np.ceil(224 * m * eps**-2 * np.log(8 * m / delta) * np.log(2 / delta)))
        scale = L_tilde * np.identity(self.d + 1)
        noise = wishart.rvs(df, scale)
        if not shifted:
            return noise

        sqrt_m_df = np.sqrt(m * df)
        sqrt_d = np.sqrt(self.d)
        sqrt_2_ln8T_a = np.sqrt(2 * np.log2(8 * T / self.alpha))
        shift_factor = L_tilde**2 * (sqrt_m_df - sqrt_d - sqrt_2_ln8T_a)**2 - \
            4 * L_tilde**2 * sqrt_m_df * (sqrt_d + sqrt_2_ln8T_a)
        shifted_noise = noise - shift_factor * np.identity(self.d + 1)
        return shifted_noise

    def generate_noise_tree(self, eps, delta, T):
        """Generate one noise matrix N.

        If noise_type is "gaussian", then generate a shifted, symmetric, gaussian noise matrix.
        If noise_type is "unshifted gaussian", then generate a symmetric, gaussian noise matrix.
        If noise_type is "laplacian", then generate a Laplacian(log(T)/eps) noise matrix.
        If noise_type is "wishart", then generate a shifted, wishart noise matrix.
        If noise_type is "unshifted wishart", then generate a wishart noise matrix.
        If noise_type is anything else, raise an error.

        Args:
            eps (float): defining level of differential privacy
            delta (float): defining level of differential privacy

        Returns:
            A numpy array with noise sampled from the specified distribution
        """
        noise = np.zeros(shape=(self.d + 1, self.d + 1))
        if self.is_theta_level:
            noise = self.laplacian()
        elif self.noise_type == 'gaussian':
            noise = self.gaussian_tree(eps, delta, T, shifted=True)
        elif self.noise_type == 'unshifted gaussian':
            noise = self.gaussian_tree(eps, delta, T, shifted=False)
        elif self.noise_type == 'laplacian':
            noise = self.laplacian_tree(eps, T)
        elif self.noise_type == 'wishart':
            noise = self.wishart_tree(eps, delta, T, shifted=True)
        elif self.noise_type == 'unshifted wishart':
            noise = self.wishart_tree(eps, delta, T, shifted=False)
        else:
            raise NotImplementedError()
        return noise


"""
# noise types

1/eps
T/eps
logT/eps
1/2eps
1/eps sqrt(j)

"""

class PrivateLinUCBUserStruct:
    def __init__(self, featureDimension, lambda_, hyperparameters, protect_context, noise_type, release_method, is_theta_level, init="zero"):
        self.d = featureDimension
        self.M = lambda_ * np.identity(self.d + 1)
        self.A = self.M[:self.d, :self.d]
        self.b = self.M[:self.d, -1]
        self.Ainv = np.linalg.inv(self.A)

        self.alpha = hyperparameters['alpha']
        self.T = hyperparameters['T']
        self.eps = hyperparameters['eps']
        self.delta = hyperparameters['delta']

        self.protect_context = protect_context
        self.is_theta_level = is_theta_level

        noise_type = noise_type.lower()
        if noise_type != 'laplacian':
            raise NotImplementedError  # more refactoring needs to be done
        self.release_method = release_method.lower()
        noise_dim = self.d - 1 if self.is_theta_level else self.d
        self.noise_generator = PrivateLinUCBNoiseGenerator(
            self.eps, self.delta, self.T, self.alpha, noise_dim, noise_type, is_theta_level)

        self.noise_store = NoisePartialSumStore.get_instance(self.release_method, hyperparameters, self.noise_generator)

        if init == "random":
            self.UserTheta = np.random.rand(self.d)
        else:
            self.UserTheta = np.zeros(self.d)
        self.UserThetaNoise = np.zeros(self.d)
        self.time = 1

    def update_user_theta(self):
        self.noise_store.add(self.time)
        N = self.noise_store.release()

        if self.is_theta_level:
            self.UserThetaNoise = N[0]
            self.b = self.M[:self.d, -1]
            self.A = self.M[:self.d, :self.d]
            self.Ainv = np.linalg.inv(self.A)
            self.UserTheta = np.dot(self.Ainv, self.b)
        else:
            N = self.noise_store.release()
            self.b = (self.M + N)[:self.d, -1]
            if self.protect_context:  # NIPS
                self.A = (self.M + N)[:self.d, :self.d]
            else:  # ICML
                self.A = self.M[:self.d, :self.d]
            self.Ainv = np.linalg.inv(self.A)
            self.UserTheta = np.dot(self.Ainv, self.b)

    def updateParameters(self, articlePicked_FeatureVector, click):
        action_and_reward_vector = np.append(
            articlePicked_FeatureVector, click)
        self.M += np.outer(action_and_reward_vector, action_and_reward_vector)
        self.update_user_theta()
        self.time += 1

    def getProb(self, alpha, article_FeatureVector):
        if alpha == -1:
            alpha = alpha = 0.1 * np.sqrt(np.log(self.time + 1))
        mean = np.dot(self.UserTheta + self.UserThetaNoise, article_FeatureVector)
        var = np.sqrt(np.dot(np.dot(article_FeatureVector,
                                    self.Ainv),  article_FeatureVector))
        pta = mean + alpha * var
        return pta

    def getProb_plot(self, alpha, article_FeatureVector):
        mean = np.dot(self.UserTheta + self.UserThetaNoise, article_FeatureVector)
        var = np.sqrt(np.dot(np.dot(article_FeatureVector,
                                    self.Ainv),  article_FeatureVector))
        pta = mean + alpha * var
        return pta, mean, alpha * var

    def getTheta(self):
        return self.UserTheta + self.UserThetaNoise

    # def getA(self):
    #     return self.A


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
                arg_dict.get('protect_context', None),
                arg_dict['noise_type'],
                arg_dict['noise_method'],
                arg_dict.get('is_theta_level', None),
                init))

    def decide(self, pool_articles, userID, k=1):
        # theta = user_features
        # x = article_features
        # V^-1 = self.users[userId].AInv
        # MEAN
        article_features = np.empty([len(pool_articles), len(
            pool_articles[0].contextFeatureVector[:self.dimension])])
        for i in range(len(pool_articles)):
            article_features[i, :] = pool_articles[i].contextFeatureVector[:self.dimension]
        user_features = self.users[userID].getTheta()
        mean_matrix = np.dot(article_features, user_features)

        # VARIANCE
        var_matrix = np.sqrt(
            np.dot(np.dot(article_features, self.users[userID].Ainv), article_features.T).clip(0))
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
        return self.users[userID].getTheta()

    def getTheta(self, userID):
        return self.users[userID].getTheta()

    # def getW(self, userID):
    # 	return np.identity(n = len(self.users))
