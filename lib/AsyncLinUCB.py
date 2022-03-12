import numpy as np
import copy

class LocalClient:
    def __init__(self, featureDimension, lambda_, delta_, NoiseScale):
        self.d = featureDimension
        self.lambda_ = lambda_
        self.delta_ = delta_
        self.NoiseScale = NoiseScale

        # Sufficient statistics stored on the client #
        # latest local sufficient statistics
        self.A_local = np.zeros((self.d, self.d))  #lambda_ * np.identity(n=self.d)
        self.b_local = np.zeros(self.d)
        self.numObs_local = 0

        # aggregated sufficient statistics recently downloaded
        self.A_uploadbuffer = np.zeros((self.d, self.d))
        self.b_uploadbuffer = np.zeros(self.d)
        self.numObs_uploadbuffer = 0

        # for computing UCB
        self.AInv = np.linalg.inv(self.A_local+self.lambda_ * np.identity(n=self.d))
        self.UserTheta = np.zeros(self.d)

        self.alpha_t = self.NoiseScale * np.sqrt(
            self.d * np.log(1 + (self.numObs_local) / (self.d * self.lambda_)) + 2 * np.log(1 / self.delta_)) + np.sqrt(
            self.lambda_)


    def getUCB(self, alpha, article_FeatureVector):
        if alpha == -1:
            alpha = self.alpha_t

        mean = np.dot(self.UserTheta, article_FeatureVector)
        var = np.sqrt(np.dot(np.dot(article_FeatureVector, self.AInv), article_FeatureVector))
        pta = mean + alpha * var
        return pta

    def localUpdate(self, articlePicked_FeatureVector, click):
        self.A_local += np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector)
        self.b_local += articlePicked_FeatureVector * click
        self.numObs_local += 1

        self.A_uploadbuffer += np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector)
        self.b_uploadbuffer += articlePicked_FeatureVector * click
        self.numObs_uploadbuffer += 1

        self.AInv = np.linalg.inv(self.A_local+self.lambda_ * np.identity(n=self.d))
        self.UserTheta = np.dot(self.AInv, self.b_local)

        self.alpha_t = self.NoiseScale * np.sqrt(
            self.d * np.log(1 + (self.numObs_local) / (self.d * self.lambda_)) + 2 * np.log(1 / self.delta_)) + np.sqrt(
            self.lambda_)

    def getTheta(self):
        return self.UserTheta

    def uploadCommTriggered(self, gammaU):
        numerator = np.linalg.det(self.A_local+self.lambda_ * np.identity(n=self.d))
        denominator = np.linalg.det(self.A_local-self.A_uploadbuffer+self.lambda_ * np.identity(n=self.d))
        return numerator/denominator > gammaU

class AsyncLinUCB:
    def __init__(self, dimension, alpha, lambda_, delta_, NoiseScale, gammaU, gammaD):
        self.dimension = dimension
        self.alpha = alpha
        self.lambda_ = lambda_
        self.delta_ = delta_
        self.NoiseScale = NoiseScale
        self.gammaU = gammaU
        self.gammaD = gammaD
        self.CanEstimateUserPreference = True

        self.clients = {}
        # aggregated sufficient statistics of all clients
        self.A_aggregated = np.zeros((self.dimension, self.dimension))
        self.b_aggregated = np.zeros(self.dimension)
        self.numObs_aggregated = 0
        # aggregated sufficient statistics that haven't been sent to each client
        self.A_downloadbuffer = {}
        self.b_downloadbuffer = {}
        self.numObs_downloadbuffer = {}

        # records
        self.totalCommCost = 0

    def decide(self, pool_articles, clientID):
        if clientID not in self.clients:
            self.clients[clientID] = LocalClient(self.dimension, self.lambda_, self.delta_, self.NoiseScale)
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

    def updateParameters(self, articlePicked, click, currentClientID):
        # update local ss, and upload buffer
        self.clients[currentClientID].localUpdate(articlePicked.featureVector, click)
        if self.clients[currentClientID].uploadCommTriggered(self.gammaU):
            # self.totalCommCost += 1
            self.totalCommCost += (self.dimension**2 + self.dimension)
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
            self.clients[currentClientID].A_uploadbuffer = np.zeros((self.dimension, self.dimension))
            self.clients[currentClientID].b_uploadbuffer = np.zeros(self.dimension)
            self.clients[currentClientID].numObs_uploadbuffer = 0

            # check download triggering event for all clients
            for clientID, clientModel in self.clients.items():
                # if clientID != currentClientID:
                if self.downloadCommTriggered(self.gammaD, clientID):
                    # self.totalCommCost += 1
                    self.totalCommCost += (self.dimension**2 + self.dimension)
                    # update client's local ss, and clear server's download buffer
                    clientModel.A_local += self.A_downloadbuffer[clientID]
                    clientModel.b_local += self.b_downloadbuffer[clientID]
                    clientModel.numObs_local += self.numObs_downloadbuffer[clientID]
                    self.A_downloadbuffer[clientID] = np.zeros((self.dimension, self.dimension))
                    self.b_downloadbuffer[clientID] = np.zeros(self.dimension)
                    self.numObs_downloadbuffer[clientID] = 0

    def downloadCommTriggered(self, gammaD, clientID):
        numerator = np.linalg.det(self.A_aggregated+self.lambda_ * np.identity(n=self.dimension))
        denominator = np.linalg.det(self.A_aggregated-self.A_downloadbuffer[clientID]+self.lambda_ * np.identity(n=self.dimension))
        return numerator/denominator > gammaD

    def getTheta(self, clientID):
        return self.clients[clientID].getTheta()


