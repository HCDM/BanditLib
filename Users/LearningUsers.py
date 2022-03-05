import numpy as np


def norm2(x, axis=0, length=1.0):
    if axis == 1:
        # batch normalization
        return length * np.reshape(1 / np.linalg.norm(x, axis=axis), (x.shape[0], 1)) * x
    else:
        return length * x / np.linalg.norm(x)


def norm_matrix(x, P):
    return np.sqrt(x @ P @ x)


def get_eigen_vec(A, option='max', return_eigen_value=False):
    q = np.linalg.eigh(A)
    if option == 'max':
        i = q[0].argmax()
    elif option == 'min':
        i = q[0].argmin()
    elif option == 'max2':
        i = [q[0].argsort()[-1], q[0].argsort()[-2]]
    else:
        return None
    if return_eigen_value:
        return q[0][i], q[1][i]
    else:
        return q[1][i]


class User:
    def __init__(self, K, rho, alpha, determ_pro):
        self.K = K
        self.rho0 = rho
        self.alpha = alpha
        self.determ_pro = determ_pro
        self.num_acceptance_total = 0.0
        self.empirical_mean_per_arm = np.zeros(self.K)
        self.num_acceptance_per_arm = np.zeros(self.K)
        self.num_rejection_per_arm = np.zeros(self.K)
        self.accept_ratio = 0
        self.ucb = np.zeros(self.K)
        self.lcb = np.zeros(self.K)
        self.global_time = self.K  # total number of interactions

    def initialize(self, min_gap=0.1):
        self.generate_mu(min_gap=min_gap)
        # pull each arm once
        for i in range(self.K):
            sampled_reward = np.random.normal(self.mu[i], self.sigma[i])
            self.empirical_mean_per_arm[i] = sampled_reward
            self.num_acceptance_per_arm[i] = 1
            self.num_acceptance_total += 0
        self.ucb = self.empirical_mean_per_arm + np.sqrt(self.trust_func() / self.num_acceptance_per_arm)
        self.lcb = 2 * self.empirical_mean_per_arm - self.ucb
        self.accept_ratio = 1

    def rho(self):
        return self.rho0 + self.accept_ratio * 1.0

    def generate_mu(self, min_gap):
        p = np.random.rand(self.K)
        gap = sorted(p)[-1]-sorted(p)[-2]
        p[np.argmax(p)] = p[np.argmax(p)] - gap + min_gap
        self.mu = p
        self.sigma = np.ones_like(self.mu)  # vector of standard deviation for each arm's reward distribution

    def trust_func(self):
        """
        trust function / numerator in CB term
        :return:
        """
        return np.max([0, 2*self.alpha*np.log( self.rho() * max( self.num_acceptance_total, 1))])

    def isUserAccept(self, arm_id):
        """
        user decides whether accept the recommended arm or not
        by looking at whether there is another arm whose LCB is larger than UCB of the recommended arm
        :param arm_id:
        :return:
        """
        self.global_time += 1

        if self.determ_pro > np.random.rand():
            gamma = self.trust_func()
            CB = [np.sqrt(gamma / float(max(1, self.num_acceptance_per_arm[i]))) for i in range(self.K)]
            UCB_i = self.empirical_mean_per_arm[arm_id] + CB[arm_id]
            # find the arm with the highest lcb
            LCB_j = max([self.empirical_mean_per_arm[i] - CB[i] for i in range(self.K)])
            if LCB_j > UCB_i:
                return False
            # no other arm j has LCB larger than UCB of arm i
            self.num_acceptance_total += 1.0
            self.accept_ratio = (self.K + self.num_acceptance_total) / (self.K + self.global_time)
            # observe reward of arm i
            sampled_reward = np.random.normal(self.mu[arm_id], self.sigma[arm_id])
            self.empirical_mean_per_arm[arm_id] = (self.empirical_mean_per_arm[arm_id] * self.num_acceptance_per_arm[
                arm_id] + sampled_reward) / (self.num_acceptance_per_arm[arm_id] + 1.0)
            self.num_acceptance_per_arm[arm_id] += 1.0
            return True
        else:
            if np.random.rand() > 0.5:
                return False
            else:
                self.num_acceptance_total += 1.0
                self.accept_ratio = (self.K + self.num_acceptance_total) / (self.K + self.global_time)
                # observe reward of arm i
                sampled_reward = np.random.normal(self.mu[arm_id], self.sigma[arm_id])
                self.empirical_mean_per_arm[arm_id] = (self.empirical_mean_per_arm[arm_id] *
                                                       self.num_acceptance_per_arm[
                                                           arm_id] + sampled_reward) / (
                                                                  self.num_acceptance_per_arm[arm_id] + 1.0)
                self.num_acceptance_per_arm[arm_id] += 1.0
                return True

    def isUserAccept2(self, arm_id):
        """
        user decides whether accept the recommended arm or not
        by looking at whether there is another arm whose LCB is larger than UCB of the recommended arm
        :param arm_id:
        :return:
        """
        self.global_time += 1
        CB_i = np.sqrt(self.trust_func() / float(max(1, self.num_acceptance_per_arm[arm_id])))
        UCB_i = self.empirical_mean_per_arm[arm_id] + CB_i
        for j in range(self.K):
            if j != arm_id:
                CB_j = np.sqrt(self.trust_func() / float(max(1, self.num_acceptance_per_arm[j])))
                LCB_j = self.empirical_mean_per_arm[j] - CB_j
                if LCB_j > UCB_i:
                    return False
        # no other arm j has LCB larger than UCB of arm i
        self.num_acceptance_total += 1.0
        self.accept_ratio = (self.K + self.num_acceptance_total) / (self.K + self.global_time)
        # observe reward of arm i
        sampled_reward = np.random.normal(self.mu[arm_id], self.sigma[arm_id])
        self.empirical_mean_per_arm[arm_id] = (self.empirical_mean_per_arm[arm_id] * self.num_acceptance_per_arm[arm_id] + sampled_reward) / (self.num_acceptance_per_arm[arm_id] + 1.0)
        self.num_acceptance_per_arm[arm_id] += 1.0
        return True


class LinUser:
    def __init__(self, feature_dim, V0, sigma=0.1, gamma=0.0, init='random'):
        self.d = feature_dim
        self.init = init
        self.theta_star = norm2(np.random.randn(self.d))
        self.theta = norm2(np.random.randn(self.d))
        self.V0 = V0
        self.V = 0 + self.V0
        self.sigma = sigma
        self.gamma = gamma
        self.I_d = np.eye(self.d)
        self.t = 0
        self.MAX_ITE = 100000
        self.beta1 = [t ** (0.01) for t in range(1, self.MAX_ITE)]
        self.beta2 = [t ** (0.01) for t in range(1, self.MAX_ITE)]

        self.reset(diag=np.ones(self.d))

    def reset(self, diag=None):
        self.V = 0 + self.V0
        self.b = np.zeros(self.d)
        self.VInv = np.linalg.inv(self.V)
        self.var = None
        if diag is None:
            diag = np.ones(self.d)
        self.gen_prior(diag=diag)

    def gen_prior(self, diag):
        for i in range(self.d):
            x = self.I_d[i]
            for j in range(int(diag[i])):
                self.get_reward(x)
        self.t = 0

    def set_beta(self, beta_left, beta_right):
        self.beta1 = beta_left
        self.beta2 = beta_right

    def set_theta(self, theta):
        self.theta_star = theta

    def get_winner(self, left, right, var=0.0,  learning=False, return_score=False, rand_ratio=0):
        if learning:
            left_score = np.dot(left, self.theta) + self.beta1[self.t] * norm_matrix(left, self.VInv)
            right_score = np.dot(right, self.theta) + self.beta2[self.t] * norm_matrix(right, self.VInv)
        else:
            left_score = np.dot(left, self.theta_star) + var * np.random.rand()
            right_score = np.dot(right, self.theta_star) + var * np.random.rand()
        if np.dot(left - right, self.theta_star) * (left_score - right_score) > 0:
            correct = True
        else:
            correct = False
        if not return_score:
            if np.random.rand() < 1-rand_ratio:
                if left_score > right_score:
                    return 0, left, correct
                else:
                    return 1, right, correct
            else:
                if np.random.rand() < 0.5:
                    return 0, left, correct
                else:
                    return 1, right, correct
        else:
            if left_score > right_score:
                return 0, left, left_score, right_score
            else:
                return 1, right, left_score, right_score

    def get_reward(self, x):
        r = np.dot(x, self.theta_star) + self.sigma * np.random.rand()
        self.update_params(x, r)

    def update_params(self, x, r):
        self.V += np.outer(x, x)
        self.b += x * r
        self.VInv = np.linalg.inv(self.V)
        self.theta = np.dot(self.VInv, self.b)
        self.t += 1
