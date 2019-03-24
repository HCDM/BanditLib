from __future__ import division  # enforce float division with `/`
import numpy as np
from util_functions import vectorize
from Recommendation import Recommendation
from BaseAlg import BaseAlg
from scipy.stats import wishart
"""Paper: Differentially Private Contextual Bandits
"""


class PrivateLinUCBNoiseGenerator:
    def __init__(self, eps, delta, T, alpha, context_dimension, noise_type):
        self.eps = eps
        self.delta = delta
        self.T = T
        self.alpha = alpha
        self.d = context_dimension
        self.noise_type = noise_type
        self.max_feature_vector_L2 = 1  # an assumed constant
        self.max_reward_L1 = 1  # an assumed constant

    def zeros(self):
        """Generate a matrix of zeros."""
        return np.zeros(shape=(self.d + 1, self.d + 1))

    def laplacian(self, denominator):
        """Generate a matrix with noise sampled from a laplacian.

        The scale of the laplacian noise is 1 divided by the provided denominator.
        The size of the matrix is (context dimension + 1, context dimension + 1).

        Args:
            denominator (float): defines scale of noise as 1 / denominator

        Returns:
            A numpy matrix of noise sampled from the computed laplacian distribution

        """
        return np.random.laplace(scale=1 / denominator, size=(self.d + 1, self.d + 1))

    def laplacian_tree(self):
        """Generate a matrix with noise sampled from a laplacian for tree-based algorithm.

        The scale of the laplacian noise is log2(T) / epsilon. The size of the matrix is
        (context dimension + 1, context dimension + 1).

        Returns:
            A numpy matrix of noise sampled from the computed laplacian distribution

        """
        return self.laplacian(self.eps / np.log2(self.T))

    def gaussian_tree(self, shifted=True):
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

        m = int(np.ceil(np.log2(self.T))) + 1  # max number of p sums
        variance = 16 * m * L_tilde**4 * \
            np.log(4 / self.delta)**2 / self.eps**2
        Z = np.random.normal(scale=np.sqrt(variance),
                             size=(self.d + 1, self.d + 1))
        sym_Z = (Z + Z.T) / np.sqrt(2)
        if not shifted:
            return sym_Z

        upsilon = np.sqrt(2 * m * variance) * \
            (4 * np.sqrt(self.d) + 2 * np.log(2 * self.T / self.alpha))
        shifted_Z = sym_Z + 2 * upsilon * np.identity(self.d + 1)
        return shifted_Z

    def wishart_tree(self, shifted=True):
        """Generate a matrix with noise sampled from a wishart distribution for tree-based algorithm.

        The degrees of freedom and scale of the wishart distribution is calculated based on the setting
        variables - epsilon, delta, alpha, and T. The size of the matrix is (context dimension + 1,
        context dimension + 1). If shifted, then the matrix will be shifted down by a computed factor.

        Args:
            shifted (bool): if noise matrix should be shifted down

        Returns:
            A numpy matrix of noise sampled from the computed wishart distribution
        """
        m = int(np.ceil(np.log2(self.T))) + 1  # max_number_of_p_sums
        L_tilde = np.sqrt(self.max_feature_vector_L2**2 +
                          self.max_reward_L1**2)
        df = int(self.d + 1 +
                 np.ceil(224 * m * self.eps**-2 * np.log(8 * m / self.delta) * np.log(2 / self.delta)))
        scale = L_tilde * np.identity(self.d + 1)
        noise = wishart.rvs(df, scale)
        if not shifted:
            return noise

        sqrt_m_df = np.sqrt(m * df)
        sqrt_d = np.sqrt(self.d)
        sqrt_2_ln8T_a = np.sqrt(2 * np.log2(8 * self.T / self.alpha))
        shift_factor = L_tilde**2 * (sqrt_m_df - sqrt_d - sqrt_2_ln8T_a)**2 - \
            4 * L_tilde**2 * sqrt_m_df * (sqrt_d + sqrt_2_ln8T_a)
        shifted_noise = noise - shift_factor * np.identity(self.d + 1)
        return shifted_noise

    def generate_noise_tree(self):
        """Generate one noise matrix N.

        If noise_type is "gaussian", then generate a shifted, symmetric, gaussian noise matrix.
        If noise_type is "unshifted gaussian", then generate a symmetric, gaussian noise matrix.
        If noise_type is "laplacian", then generate a Laplacian(log(T)/eps) noise matrix.
        If noise_type is "wishart", then generate a shifted, wishart noise matrix.
        If noise_type is "unshifted wishart", then generate a wishart noise matrix.
        If noise_type is anything else, raise an error.

        Args:
            noise_type (str): defines the type of noise matrix

        Returns:
            A numpy array with noise sampled from the specified distribution
        """
        noise = np.zeros(shape=(self.d + 1, self.d + 1))
        if self.noise_type == 'gaussian':
            noise = self.gaussian_tree(shifted=True)
        elif self.noise_type == 'unshifted gaussian':
            noise = self.gaussian_tree(shifted=False)
        elif self.noise_type == 'laplacian':
            noise = self.laplacian_tree()
        elif self.noise_type == 'wishart':
            noise = self.wishart_tree(shifted=True)
        elif self.noise_type == 'unshifted wishart':
            noise = self.wishart_tree(shifted=False)
        else:
            raise NotImplementedError()
        return noise


class NoisePartialSum:
    def __init__(self, start, size, noise):
        self.start = start
        self.size = size
        self.noise = noise

    def __str__(self):
        return 'NoisePartialSum(start=%i, size=%i)' % (self.start, self.size)


class NoisePartialSumStore:
    def __init__(self, noise_generator, release_method='tree'):
        if release_method != 'tree' and noise_generator.noise_type != 'laplacian':
            raise NotImplementedError
        if release_method not in ['tree', 'every', 'once', 'sqrt']:
            raise NotImplementedError
        self.noise_generator = noise_generator
        self.release_method = release_method
        self.store = {}
        self.START_TIME = 1


    def consolidate_for_once(self, time):
        """Delete all partial sums except at start.
        
        This is used for the 'once' release method. The first noise added
        will be of a magnitude great enough to protect privacy throughout
        the algorithm.

        Args:
            time (int): time of newly added partial sum
        """
        if time != self.START_TIME:
            del self.store[time]


    def consolidate_for_every(self, time):
        """Collapse all partial sums into one partial sum.
        
        This is used for the 'every' release method. Each new partial sum
        will have a small amount of noise that will accumulate in a bigger
        and bigger partial sum.

        Args:
            time (int): time of newly added partial sum
        """
        current_total_noise = self.store[self.START_TIME].noise
        newly_added_noise = self.store[time].noise
        new_total_noise = current_total_noise + newly_added_noise
        new_psum_size = time
        self.store[self.START_TIME] = NoisePartialSum(1, new_psum_size, new_total_noise)
        if time != self.START_TIME:
            del self.store[time]

    def consolidate_for_sqrt(self, time):
        """Collapse all block partial sums into one partial sum.

        This is used for the 'sqrt' release method. Theoretically, we compute partial sums
        of either a block's size or of a single item. If there are enough single items to
        create a block, consolidate them into one block. To achieve O(1) storage, we
        consolidate all block-level partial sums as well.

        Args:
            time (int): time of newly added partial sum
        """
        eps = self.noise_generator.eps
        block_size = int(np.sqrt(self.noise_generator.T))
        block_start = time - block_size + 1
        if block_start in self.store:
            for _t in range(block_start, time + 1):
                del self.store[_t]
            block_noise = self.noise_generator.laplacian(eps)
            if block_start == self.START_TIME:
                self.store[self.START_TIME] = NoisePartialSum(
                    self.START_TIME, block_size, block_noise)
            else:
                self.store[self.START_TIME] = NoisePartialSum(
                    self.START_TIME, self.store[self.START_TIME].size + block_size, self.store[self.START_TIME].noise + block_noise)

    def consolidate_for_tree(self, time):
        """Collapse all partial sums into "power of two"-sized blocks.
        
        This is used for the 'tree' release method. Instead of fixed-sized blocks like
        in the 'sqrt' release method, this consolidation technique will collapse sums
        into "power of two"-sized blocks. This method recursively combines equal-sized
        partial sums until there are no more.

        Args:
            time (int): time of newly added partial sum
        """
        prev_p_sum_time = self.store[time].start - self.store[time].size
        if prev_p_sum_time in self.store:
            if self.store[time].size == self.store[prev_p_sum_time].size:
                new_size = self.store[time].size * 2
                new_noise = self.noise_generator.generate_noise_tree()
                self.store[prev_p_sum_time] = NoisePartialSum(
                    prev_p_sum_time, new_size, new_noise)
                del self.store[time]
                self.consolidate_for_tree(prev_p_sum_time)

    def consolidate_store(self, time):
        if self.release_method == 'once':
            self.consolidate_for_once(time)
        elif self.release_method == 'every':
            self.consolidate_for_every(time)
        elif self.release_method == 'sqrt':
            self.consolidate_for_sqrt(time)
        elif self.release_method == 'tree':
            self.consolidate_for_tree(time)

    def add_noise(self, time):
        T = self.noise_generator.T
        eps = self.noise_generator.eps
        noise = self.noise_generator.zeros()
        if self.release_method == 'once':
            if len(self.store) == 0:
                noise = self.noise_generator.laplacian(eps / T)
        elif self.release_method == 'every':
            noise = self.noise_generator.laplacian(eps)
        elif self.release_method == 'sqrt':
            noise = self.noise_generator.laplacian(eps / 2)
        elif self.release_method == 'tree':
            noise = self.noise_generator.generate_noise_tree()
        self.store[time] = NoisePartialSum(start=time, size=1, noise=noise)
        self.consolidate_store(time)

    def release_noise(self):
        """Returns the sum of noise in all partial sums in store."""
        N = self.noise_generator.zeros()
        for p_sum in self.store.values():
            N += p_sum.noise
        return N


class PrivateLinUCBUserStruct:
    def __init__(self, featureDimension, lambda_, hyperparameters, protect_context, noise_type, release_method, init="zero"):
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

        noise_type = noise_type.lower()
        release_method = release_method.lower()
        noise_generator = PrivateLinUCBNoiseGenerator(
            self.eps, self.delta, self.T, self.alpha, self.d, noise_type)
        self.noise_store = NoisePartialSumStore(noise_generator, release_method)

        if init == "random":
            self.UserTheta = np.random.rand(self.d)
        else:
            self.UserTheta = np.zeros(self.d)
        self.START_TIME = 1
        self.time = self.START_TIME

    def update_user_theta(self, N):
        self.u = (self.M + N)[:self.d, -1]
        if self.protect_context:  # NIPS
            self.V = (self.M + N)[:self.d, :self.d]
        else:  # ICML
            self.V = self.M[:self.d, :self.d]
        self.Vinv = np.linalg.inv(self.V)
        self.UserTheta = np.dot(self.Vinv, self.u)

    def updateParameters(self, articlePicked_FeatureVector, click):
        self.noise_store.add_noise(self.time)
        N = self.noise_store.release_noise()
        action_and_reward_vector = np.append(
            articlePicked_FeatureVector, click)
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
                arg_dict['noise_method'],
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
