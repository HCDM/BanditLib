import numpy as np
import copy
import time
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
"""
FedGLB-UCB

"""

def sigmoid(x):
    return 1/(1+np.exp(-x))

def projection_in_norm(x, M):
    """Projection of x to simplex indiced by matrix M. Uses quadratic programming."""
    x = np.mat(x).T
    M = np.mat(M)
    m = M.shape[0]

    P = matrix(2 * M)
    q = matrix(-2 * M * x)
    G = matrix(-np.eye(m))
    h = matrix(np.zeros((m, 1)))
    A = matrix(np.ones((1, m)))
    b = matrix(1.0)
    sol = solvers.qp(P, q, G, h, A, b)
    return np.squeeze(sol["x"])

class LocalClient:
    def __init__(self, featureDimension, lambda_, c_mu, n_users, S, R, delta):
        self.d = featureDimension
        self.lambda_ = lambda_
        self.c_mu = c_mu
        self.n_users = n_users
        self.S = S 
        self.R = R
        self.delta = delta
        # Sufficient statistics stored on the client #
        # latest local sufficient statistics
        self.A = np.zeros((self.d, self.d))
        self.b = np.zeros(self.d)
        self.numObs_local = 0

        self.X = np.zeros((0, self.d))
        self.y = np.zeros((0,))

        # aggregated sufficient statistics to upload
        self.A_uploadbuffer = np.zeros((self.d, self.d))
        self.numObs_uploadbuffer = 0

        self.AInv = self.c_mu/self.lambda_ * np.identity(self.d)
        self.ThetaONS = np.zeros(self.d)  # ONS estimation

        self.ThetaRidge = np.zeros(self.d)  # center of confidence ellipsoid
        self.loss_diff_bound_B1 = 0
        self.loss_diff_bound_B2 = 0.5 * self.c_mu * self.lambda_ * self.S**2
        self.beta_t_global_part = 0
        self.beta_t_local_part = 0
        self.sum_z_sqr = 0

    def localUpdate_ONSStep(self, articlePicked_FeatureVector, click):
        # get predicted reward using ThetaONS
        z = articlePicked_FeatureVector.dot(self.ThetaONS)
        self.b += z * articlePicked_FeatureVector
        self.sum_z_sqr += z**2

        # use sherman-morrison formula to update AInv
        tmp = self.AInv.dot(articlePicked_FeatureVector)
        self.AInv -= np.outer(tmp, tmp) / (1 + articlePicked_FeatureVector.dot(tmp)) 

        # run one step of ONS with the new data point
        grad = -click + sigmoid(z)
        theta_prime = self.ThetaONS - grad / self.c_mu * self.AInv.dot(articlePicked_FeatureVector)
        self.ThetaONS = projection_in_norm(x=theta_prime, M=self.A+self.lambda_/self.c_mu * np.identity(n=self.d))

        # update parameters for confidence ellipsoid
        self.ThetaRidge = np.dot(self.AInv, self.b)
        self.loss_diff_bound_B2 += (0.5/self.c_mu) * np.dot(grad*articlePicked_FeatureVector, np.dot(self.AInv, grad*articlePicked_FeatureVector))
        self.beta_t_local_part = 1 + 4*self.loss_diff_bound_B2/self.c_mu + 8 * self.R**2 / self.c_mu**2 * np.log(self.n_users/self.delta * np.sqrt(4 + 8*self.loss_diff_bound_B2/self.c_mu+64*self.R**4/(self.c_mu**4 * 4 * self.delta**2)))

    def syncRoundTriggered(self, threshold):
        numerator = np.linalg.det(self.A+self.lambda_/self.c_mu * np.identity(n=self.d))
        denominator = np.linalg.det(self.A-self.A_uploadbuffer+self.lambda_/self.c_mu * np.identity(n=self.d))
        return np.log(numerator/denominator)*self.numObs_uploadbuffer >= threshold

    def getUCB(self, alpha, article_FeatureVector):
        mean = np.dot(self.ThetaRidge, article_FeatureVector)
        var = np.sqrt(np.dot(np.dot(article_FeatureVector, self.AInv), article_FeatureVector))
        pta = mean + alpha * var
        return pta

class FedGLB_UCB:
    def __init__(self, dimension, lambda_, threshold, n_users, c_mu=0.25, delta=1e-2, S=1, R=0.5, alpha=None, alpha_t_scaling=1, init_x=None, max_iters=None):
        self.dimension = dimension
        self.alpha = alpha
        self.alpha_t_scaling = alpha_t_scaling
        self.lambda_ = lambda_
        self.c_mu = c_mu
        self.S = S  # upper bound for norm of theta_star
        self.R = R
        self.delta = delta
        self.n_users = n_users
        self.threshold = threshold
        
        # aggregated sufficient statistics of all clients
        self.A_g = self.lambda_/self.c_mu * np.identity(n=self.dimension)
        self.b_g = np.zeros(self.dimension)
        self.sum_z_sqr_global = 0

        self.GlobalTheta = np.zeros(self.dimension)

        # initialization for AGD
        self.init_x = init_x
        self.max_iters = max_iters
        self.numObs_g = 0

        self.clients = {}

        # records
        self.totalCommCost = 0
        self.CanEstimateUserPreference = False  # set to true if want to record parameter estimation error

    # def decide(self, arm_matrix, currentClientID):
    #     # start = time.time()
    #     if currentClientID not in self.clients:
    #         self.clients[currentClientID] = LocalClient(featureDimension=self.dimension, lambda_=self.lambda_, c_mu=self.c_mu, n_users=self.n_users, S=self.S, R=self.R, delta=self.delta)
            
    #     ucbs = np.sqrt((np.matmul(arm_matrix, self.clients[currentClientID].AInv) * arm_matrix).sum(axis=1))

    #     if self.alpha is not None:
    #         alpha_t = self.alpha
    #     else:
    #         alpha_t = np.sqrt(self.lambda_/self.c_mu*self.S**2 +self.clients[currentClientID].beta_t_local_part+self.clients[currentClientID].beta_t_global_part-self.clients[currentClientID].sum_z_sqr+np.dot(self.clients[currentClientID].ThetaRidge,self.clients[currentClientID].b))
    #         alpha_t = self.alpha_t_scaling * alpha_t
    #     # print(alpha_t)
    #     # Compute UCB
    #     mu = np.matmul(arm_matrix, self.clients[currentClientID].ThetaRidge) + alpha_t * ucbs
    #     # Argmax breaking ties randomly
    #     arm = np.random.choice(np.flatnonzero(mu == mu.max()))
    #     # end = time.time()
    #     # print("v0 select takes: {}".format(end - start))
    #     return arm_matrix[arm], arm

    def decide(self, pool_articles, clientID):
        if clientID not in self.clients:
            self.clients[clientID] = LocalClient(featureDimension=self.dimension, lambda_=self.lambda_, c_mu=self.c_mu, n_users=self.n_users, S=self.S, R=self.R, delta=self.delta)

        maxPTA = float('-inf')
        articlePicked = None

        if self.alpha is not None:
            alpha_t = self.alpha
        else:
            alpha_t = np.sqrt(self.lambda_/self.c_mu*self.S**2 +self.clients[clientID].beta_t_local_part+self.clients[clientID].beta_t_global_part-self.clients[clientID].sum_z_sqr+np.dot(self.clients[clientID].ThetaRidge,self.clients[clientID].b))
            alpha_t = self.alpha_t_scaling * alpha_t

        for x in pool_articles:
            x_pta = self.clients[clientID].getUCB(alpha_t, x.featureVector)
            # print(x_pta)
            # pick article with highest UCB score
            if maxPTA < x_pta:
                articlePicked = x
                maxPTA = x_pta
        # print(maxPTA)
        return articlePicked


    def updateParameters(self, articlePicked, click, currentClientID):
        articlePicked_FeatureVector = articlePicked.featureVector
        # start = time.time()
        # update local dataset, sufficient statistics, and upload buffer
        self.clients[currentClientID].numObs_local += 1
        self.clients[currentClientID].numObs_uploadbuffer += 1
        self.clients[currentClientID].A += np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector)
        self.clients[currentClientID].A_uploadbuffer += np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector)
        self.clients[currentClientID].X = np.concatenate((self.clients[currentClientID].X, articlePicked_FeatureVector.reshape(1, self.dimension)), axis=0)
        self.clients[currentClientID].y = np.concatenate((self.clients[currentClientID].y, np.array([click])), axis=0)

        if not self.clients[currentClientID].syncRoundTriggered(self.threshold):
            ## local update with ONS ##
            self.clients[currentClientID].localUpdate_ONSStep(articlePicked_FeatureVector, click)
        else:
            ## global update using AGD ##
            # first collect the local updates from all the clients
            for clientID, clientModel in self.clients.items():
                if clientID != currentClientID:
                    # self.totalCommCost += 1
                    self.totalCommCost += (self.dimension*self.dimension + 1)
                self.A_g += clientModel.A_uploadbuffer
                self.numObs_g += clientModel.numObs_uploadbuffer
                clientModel.A_uploadbuffer = np.zeros((self.dimension, self.dimension))
                clientModel.numObs_uploadbuffer = 0

            # decide how to initialize AGD
            if self.init_x is None:
                # use last global model to initialize AGD
                x = self.GlobalTheta
            else:
                # use specified model to initialize AGD, e.g. zero vector
                x = self.init_x

            lambda_prev = 0
            lambda_curr = 1
            gamma = 1
            y_prev = x
            # step_size = 1e-1
            step_size = 1/(0.25+self.lambda_/self.numObs_g) * 0.1
            
            if self.max_iters is None:
                max_iters = self.numObs_g*2
            else:
                max_iters = self.max_iters
            for iter in range(max_iters):
                # collect and aggregate local gradients w.r.t. iterate x
                gradient = np.zeros(self.dimension)
                for _, clientModel in self.clients.items():
                    z = np.dot(clientModel.X, x)
                    gradient += np.dot(np.transpose(clientModel.X), -clientModel.y + sigmoid(z))
                gradient += self.lambda_ * x
                gradient = gradient / self.numObs_g

                ## stopping criteria to guarantee f(theta_t) - f(theta_hat^MLE) <= epsilon_t := 1/numObs**2 ##
                ## based on the (lambda/numObs)-strongly-convexity property of f() ##
                if np.linalg.norm(gradient) <= np.sqrt((2*self.lambda_)/self.numObs_g**3):
                    break
                # one step of AGD update
                y_curr = x - step_size * gradient
                x = (1 - gamma) * y_curr + gamma * y_prev
                y_prev = y_curr
                lambda_tmp = lambda_curr
                lambda_curr = (1 + np.sqrt(1 + 4 * lambda_prev * lambda_prev)) / 2
                lambda_prev = lambda_tmp
                gamma = (1 - lambda_prev) / lambda_curr

            # self.totalCommCost += iter*(len(self.clients)-1)*2
            self.totalCommCost += iter*(len(self.clients)-1)*2*self.dimension

            ## update parameters for confidence ellipsoid ##
            loss_diff_bound_B1 = self.lambda_ * 0.5 +  1.0 / self.numObs_g
            confidence_width_for_ThetaONS =  1 / self.c_mu * np.sqrt(np.dot(np.dot(gradient*self.numObs_g, self.A_g), gradient*self.numObs_g)) + self.R / self.c_mu * np.sqrt(self.dimension * np.log(1+self.numObs_g*self.c_mu/(self.dimension*self.lambda_))+2*np.log(1.0/self.delta)) + np.sqrt(self.lambda_/self.c_mu)*self.S
            loss_diff_bound_B2 = 0.5 / self.c_mu * confidence_width_for_ThetaONS**2
            log_det = np.log(1/self.delta) + self.dimension/2 * np.log(1+(1+self.numObs_g)/self.dimension)
            beta_t_global_part = 8 * self.R**2 / self.c_mu**2 * log_det + loss_diff_bound_B1 + 4*self.R/self.c_mu*np.sqrt(2*log_det)*(np.linalg.norm(x)+self.S+np.sqrt(loss_diff_bound_B1))
            self.sum_z_sqr_global = 0
            for _, clientModel in self.clients.items():
                z_per_client = np.dot(clientModel.X, x)
                self.sum_z_sqr_global += np.dot(z_per_client,z_per_client)

            self.b_g = np.dot(self.A_g - self.lambda_/self.c_mu * np.identity(n=self.dimension), x)
            AInv_g = np.linalg.inv(self.A_g)
            GlobalThetaRidge = np.dot(AInv_g, self.b_g)
            self.GlobalTheta = x
            ## synchronize the statistics on local bandit models ##
            for _, clientModel in self.clients.items():
                if clientID != currentClientID:
                    # self.totalCommCost += 1
                    self.totalCommCost += (self.dimension*self.dimension + self.dimension + 1)
                clientModel.A = self.A_g - self.lambda_/self.c_mu * np.identity(n=self.dimension) 
                # clientModel.b = self.b_g 
                # clientModel.AInv = AInv_g 
                # clientModel.numObs_local = self.numObs_g 
                # clientModel.ThetaRidge = GlobalThetaRidge # center of confidence ellipsoid
                # clientModel.ThetaONS = x  # ONS estimation
                clientModel.b = copy.deepcopy(self.b_g)
                clientModel.AInv = copy.deepcopy(AInv_g)
                clientModel.numObs_local = copy.deepcopy(self.numObs_g)
                clientModel.ThetaRidge = copy.deepcopy(GlobalThetaRidge)  # center of confidence ellipsoid
                clientModel.ThetaONS = copy.deepcopy(x)  # ONS estimation

                clientModel.loss_diff_bound_B1 = loss_diff_bound_B1 
                clientModel.loss_diff_bound_B2 = loss_diff_bound_B2
                clientModel.beta_t_global_part = beta_t_global_part
                clientModel.beta_t_local_part = loss_diff_bound_B2
                clientModel.sum_z_sqr = self.sum_z_sqr_global

        # end = time.time()
        # print("v0 update takes: {}".format(end - start))

    def getTheta(self, clientID):
        return self.clients[clientID].ThetaRidge


