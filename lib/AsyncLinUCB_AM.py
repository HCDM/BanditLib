import numpy as np
import copy

class LocalClient:
    def __init__(self, dimension_g, dimension_l, lambda_, delta_, NoiseScale):
        self.d_g = dimension_g
        self.d_l = dimension_l
        self.lambda_ = lambda_
        self.delta_ = delta_
        self.NoiseScale = NoiseScale

        # Sufficient statistics stored on the client for estimating global component #
        # latest local sufficient statistics
        self.A_local = np.zeros((self.d_g, self.d_g))  #lambda_ * np.identity(n=self.d)
        self.b_local = np.zeros(self.d_g)
        self.numObs_local = 0

        # aggregated sufficient statistics recently downloaded
        self.A_uploadbuffer = np.zeros((self.d_g, self.d_g))
        self.b_uploadbuffer = np.zeros(self.d_g)
        self.numObs_uploadbuffer = 0

        # Sufficient statistics stored on the client for estimating personalized local component #
        self.A_local_l = np.zeros((self.d_l, self.d_l))  #lambda_ * np.identity(n=self.d)
        self.b_local_l = np.zeros(self.d_l)

        # for computing UCB
        self.AInv_g = np.linalg.inv(self.A_local+self.lambda_ * np.identity(n=self.d_g))
        self.UserTheta_g = np.zeros(self.d_g)
        self.AInv_l = np.linalg.inv(self.A_local_l+self.lambda_ * np.identity(n=self.d_l))
        self.UserTheta_l = np.zeros(self.d_l)

        self.alpha_t_g = self.NoiseScale * np.sqrt(
            self.d_g * np.log(1 + (self.numObs_local) / (1+self.d_g * self.lambda_)) + 2 * np.log(1 / self.delta_)) + np.sqrt(
            self.lambda_)
        self.alpha_t_l = self.NoiseScale * np.sqrt(
            self.d_l * np.log(1 + (self.numObs_local) / (1+self.d_l * self.lambda_)) + 2 * np.log(
                1 / self.delta_)) + np.sqrt(
            self.lambda_)

    def getUCB(self, alpha, article_FeatureVector):
        # print("article_FeatureVector shape {}".format(article_FeatureVector.shape))
        if alpha == -1:
            alpha_g = self.alpha_t_g
            alpha_l = self.alpha_t_l
        else:
            alpha_g = alpha
            alpha_l = alpha
        article_FeatureVector_g = article_FeatureVector[:self.d_g]
        article_FeatureVector_l = article_FeatureVector[self.d_g:]
        mean_g = np.dot(self.UserTheta_g, article_FeatureVector_g)
        var_g = np.sqrt(np.dot(np.dot(article_FeatureVector_g, self.AInv_g), article_FeatureVector_g))
        mean_l = np.dot(self.UserTheta_l, article_FeatureVector_l)
        var_l = np.sqrt(np.dot(np.dot(article_FeatureVector_l, self.AInv_l), article_FeatureVector_l))
        pta = mean_g + alpha_g * var_g + mean_l + alpha_l * var_l
        return pta

    def localUpdate(self, articlePicked_FeatureVector, click, inner_iters=100):
        # run alternating minimization to get estimated partial rewards
        x_g = articlePicked_FeatureVector[:self.d_g]
        x_l = articlePicked_FeatureVector[self.d_g:]

        theta_g = self.UserTheta_g
        theta_l = self.UserTheta_l
        y_l = click - np.dot(theta_g, x_g)
        y_g = click - np.dot(theta_l, x_l)
        for iter in range(inner_iters):
            theta_g = np.dot(np.linalg.pinv(self.A_local+np.outer(x_g, x_g)), self.b_local+x_g*y_g)
            l2_norm = np.linalg.norm(theta_g, ord=2)
            theta_g = theta_g / max(l2_norm, 1)

            y_l = click - np.dot(theta_g, x_g)
            theta_l = np.dot(np.linalg.pinv(self.A_local_l + np.outer(x_l, x_l)), self.b_local_l + x_l * y_l)
            l2_norm = np.linalg.norm(theta_l, ord=2)
            theta_l = theta_l / max(l2_norm, 1)
            y_g = click - np.dot(theta_l, x_l)

        self.A_local += np.outer(x_g, x_g)
        self.b_local += x_g * y_g
        self.numObs_local += 1

        self.A_uploadbuffer += np.outer(x_g, x_g)
        self.b_uploadbuffer += x_g * y_g
        self.numObs_uploadbuffer += 1

        self.A_local_l += np.outer(x_l, x_l)
        self.b_local_l += x_l * y_l

        self.AInv_g = np.linalg.inv(self.A_local+self.lambda_ * np.identity(n=self.d_g))
        self.UserTheta_g = np.dot(self.AInv_g, self.b_local)
        self.AInv_l = np.linalg.inv(self.A_local_l+self.lambda_ * np.identity(n=self.d_l))
        self.UserTheta_l = np.dot(self.AInv_l, self.b_local_l)

        self.alpha_t_g = self.NoiseScale * np.sqrt(
            self.d_g * np.log(1 + (self.numObs_local) / (1+self.d_g * self.lambda_)) + 2 * np.log(1 / self.delta_)) + np.sqrt(
            self.lambda_)
        self.alpha_t_l = self.NoiseScale * np.sqrt(
            self.d_l * np.log(1 + (self.numObs_local) / (1+self.d_l * self.lambda_)) + 2 * np.log(1 / self.delta_)) + np.sqrt(
            self.lambda_)

    def getTheta(self):
        return np.concatenate([self.UserTheta_g, self.UserTheta_l])

    def uploadCommTriggered(self, gammaU):
        numerator = np.linalg.det(self.A_local+self.lambda_ * np.identity(n=self.d_g))
        denominator = np.linalg.det(self.A_local-self.A_uploadbuffer+self.lambda_ * np.identity(n=self.d_g))
        return numerator/denominator >= gammaU

class AsyncLinUCB_AM:
    def __init__(self, dimension_g, dimension_l, alpha, lambda_, delta_, NoiseScale, gammaU, gammaD):
        self.dimension_g = dimension_g
        self.dimension_l = dimension_l
        self.alpha = alpha
        self.lambda_ = lambda_
        self.delta_ = delta_
        self.NoiseScale = NoiseScale
        self.gammaU = gammaU
        self.gammaD = gammaD
        self.CanEstimateUserPreference = True

        self.clients = {}
        # aggregated sufficient statistics of all clients
        self.A_aggregated = np.zeros((self.dimension_g, self.dimension_g))
        self.b_aggregated = np.zeros(self.dimension_g)
        self.numObs_aggregated = 0
        # aggregated sufficient statistics that haven't been sent to each client
        self.A_downloadbuffer = {}
        self.b_downloadbuffer = {}
        self.numObs_downloadbuffer = {}

        # records
        self.totalCommCost = 0

    def decide(self, pool_articles, clientID):
        if clientID not in self.clients:
            self.clients[clientID] = LocalClient(self.dimension_g, self.dimension_l, self.lambda_, self.delta_, self.NoiseScale)
            self.A_downloadbuffer[clientID] = copy.deepcopy(self.A_aggregated)
            self.b_downloadbuffer[clientID] = copy.deepcopy(self.b_aggregated)
            self.numObs_downloadbuffer[clientID] = copy.deepcopy(self.numObs_aggregated)

        maxPTA = float('-inf')
        articlePicked = None

        for x in pool_articles:
            x_pta = self.clients[clientID].getUCB(self.alpha, x.featureVector)
            # pick article with highest UCB score
            if maxPTA < x_pta:
                articlePicked = x
                maxPTA = x_pta

        return articlePicked

    def decide_realData(self, pool_articles, clientID):
        if clientID not in self.clients:
            self.clients[clientID] = LocalClient(self.dimension_g, self.dimension_l, self.lambda_, self.delta_, self.NoiseScale)
            self.A_downloadbuffer[clientID] = copy.deepcopy(self.A_aggregated)
            self.b_downloadbuffer[clientID] = copy.deepcopy(self.b_aggregated)
            self.numObs_downloadbuffer[clientID] = copy.deepcopy(self.numObs_aggregated)

        maxPTA = float('-inf')
        articlePicked = None

        for x in pool_articles:
            x_pta = self.clients[clientID].getUCB(self.alpha, np.concatenate([x.featureVector, x.featureVector]))
            # pick article with highest UCB score
            if maxPTA < x_pta:
                articlePicked = x
                maxPTA = x_pta

        return articlePicked

    def updateParameters(self, articlePicked, click, currentClientID):
        # update local ss, and upload buffer
        self.clients[currentClientID].localUpdate(articlePicked.featureVector, click)
        if self.clients[currentClientID].uploadCommTriggered(self.gammaU):
            # self.totalCommCost += 1
            self.totalCommCost += (self.dimension_g**2 + self.dimension_g)
            # update server's aggregated ss
            self.A_aggregated += self.clients[currentClientID].A_uploadbuffer
            self.b_aggregated += self.clients[currentClientID].b_uploadbuffer
            self.numObs_aggregated += self.clients[currentClientID].numObs_uploadbuffer
            # update server's download buffer for other clients
            for clientID in self.A_downloadbuffer.keys():
                if clientID != currentClientID:
                    self.A_downloadbuffer[clientID] += self.clients[currentClientID].A_uploadbuffer
                    self.b_downloadbuffer[clientID] += self.clients[currentClientID].b_uploadbuffer
                    self.numObs_downloadbuffer[clientID] += self.clients[currentClientID].numObs_uploadbuffer
            # clear client's upload buffer
            self.clients[currentClientID].A_uploadbuffer = np.zeros((self.dimension_g, self.dimension_g))
            self.clients[currentClientID].b_uploadbuffer = np.zeros(self.dimension_g)
            self.clients[currentClientID].numObs_uploadbuffer = 0

            # check download triggering event for all clients
            for clientID, clientModel in self.clients.items():
                if self.downloadCommTriggered(self.gammaD, clientID):
                    # self.totalCommCost += 1
                    self.totalCommCost += (self.dimension_g**2 + self.dimension_g)
                    # update client's local ss, and clear server's download buffer
                    clientModel.A_local += self.A_downloadbuffer[clientID]
                    clientModel.b_local += self.b_downloadbuffer[clientID]
                    clientModel.numObs_local += self.numObs_downloadbuffer[clientID]
                    self.A_downloadbuffer[clientID] = np.zeros((self.dimension_g, self.dimension_g))
                    self.b_downloadbuffer[clientID] = np.zeros(self.dimension_g)
                    self.numObs_downloadbuffer[clientID] = 0

    def updateParameters_realData(self, articlePicked, click, currentClientID):
        # update local ss, and upload buffer
        self.clients[currentClientID].localUpdate(np.concatenate([articlePicked.featureVector,articlePicked.featureVector]), click)
        if self.clients[currentClientID].uploadCommTriggered(self.gammaU):
            # self.totalCommCost += 1
            self.totalCommCost += (self.dimension_g**2 + self.dimension_g)
            # update server's aggregated ss
            self.A_aggregated += self.clients[currentClientID].A_uploadbuffer
            self.b_aggregated += self.clients[currentClientID].b_uploadbuffer
            self.numObs_aggregated += self.clients[currentClientID].numObs_uploadbuffer
            # update server's download buffer for other clients
            for clientID in self.A_downloadbuffer.keys():
                if clientID != currentClientID:
                    self.A_downloadbuffer[clientID] += self.clients[currentClientID].A_uploadbuffer
                    self.b_downloadbuffer[clientID] += self.clients[currentClientID].b_uploadbuffer
                    self.numObs_downloadbuffer[clientID] += self.clients[currentClientID].numObs_uploadbuffer
            # clear client's upload buffer
            self.clients[currentClientID].A_uploadbuffer = np.zeros((self.dimension_g, self.dimension_g))
            self.clients[currentClientID].b_uploadbuffer = np.zeros(self.dimension_g)
            self.clients[currentClientID].numObs_uploadbuffer = 0

            # check download triggering event for all other clients
            for clientID, clientModel in self.clients.items():
                if self.downloadCommTriggered(self.gammaD, clientID):
                    # self.totalCommCost += 1
                    self.totalCommCost += (self.dimension_g**2 + self.dimension_g)
                    # update client's local ss, and clear server's download buffer
                    clientModel.A_local += self.A_downloadbuffer[clientID]
                    clientModel.b_local += self.b_downloadbuffer[clientID]
                    clientModel.numObs_local += self.numObs_downloadbuffer[clientID]
                    self.A_downloadbuffer[clientID] = np.zeros((self.dimension_g, self.dimension_g))
                    self.b_downloadbuffer[clientID] = np.zeros(self.dimension_g)
                    self.numObs_downloadbuffer[clientID] = 0

    def downloadCommTriggered(self, gammaD, clientID):
        numerator = np.linalg.det(self.A_aggregated+self.lambda_ * np.identity(n=self.dimension_g))
        denominator = np.linalg.det(self.A_aggregated-self.A_downloadbuffer[clientID]+self.lambda_ * np.identity(n=self.dimension_g))
        return numerator/denominator >= gammaD

    def getTheta(self, clientID):
        return self.clients[clientID].getTheta()


