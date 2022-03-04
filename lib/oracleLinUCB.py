import numpy as np

class globalClusterModel:
    # stores statistics for estimating local user parameter
    # stores statistics for change detection
    # stores statistics for cluster estimation
    def __init__(self, dimension, alpha, lambda_, NoiseScale, delta_1):
        self.d = dimension
        self.alpha = alpha  # use constant alpha, instead of the one defined in LinUCB
        self.lambda_ = lambda_
        self.delta_1 = delta_1


        # LinUCB statistics
        self.A = lambda_ * np.identity(n=self.d)
        self.b = np.zeros(self.d)
        self.AInv = np.linalg.inv(self.A)

        self.NoiseScale = NoiseScale
        self.update_num = 0  # number of times this user has been updated
        self.alpha_t = self.NoiseScale ** 2 * np.sqrt(
            self.d * np.log(1 + self.update_num / (self.d * self.lambda_)) + 2 * np.log(1 / self.delta_1)) + np.sqrt(
            self.lambda_)

        self.UserTheta = np.zeros(self.d)

    def updateParameters(self, articlePicked_FeatureVector, click):
        # update LinUCB statistics
        self.A += np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector)
        self.b += articlePicked_FeatureVector * click
        self.AInv = np.linalg.inv(self.A)
        self.UserTheta = np.dot(self.AInv, self.b)

        self.update_num += 1.0

        self.alpha_t = self.NoiseScale ** 2 * np.sqrt(
            self.d * np.log(1 + self.update_num / (self.d * self.lambda_)) + 2 * np.log(1 / self.delta_1)) + np.sqrt(
            self.lambda_)

    def getCB(self, x, useConstantAlpha = False):
        # print("x shape {}".format(x.shape))
        # print("AInv shape {}".format(self.AInv.shape))
        var = np.sqrt(np.dot(np.dot(x, self.AInv), x))

        if useConstantAlpha:
            return self.alpha * var
        else:
            return self.alpha_t * var

    def getProb(self, article_FeatureVector):
        mean = np.dot(self.UserTheta, article_FeatureVector)
        CB = self.getCB(article_FeatureVector, useConstantAlpha=False)
        pta = mean + CB
        return pta

class oracleLinUCB:
    """
    algorithm that has access to ground truth change point and cluster
    we initialize this algorithm with the globalClusterModels
    And each time we serve a user, we observe the ground truth index of unique parameter
    Therefore, we just create a LinUCB model for each unique parameter
    """
    def __init__(self, dimension, alpha, lambda_, NoiseScale, delta_1):
        self.d = dimension
        self.alpha = alpha
        self.lambda_ = lambda_
        self.NoiseScale = NoiseScale
        self.delta_1 = delta_1
        self.globalClusterModels = {}

        self.usersClusterIndexSeq = {}
        self.CanEstimateUserPreference = True
        self.CanEstimateUserCluster = True
        self.cluster = []  # stores current user's neighbors

    def decide(self, pool_articles, userID, clusterIndex):
        if userID not in self.usersClusterIndexSeq:
            self.usersClusterIndexSeq[userID] = []
        self.usersClusterIndexSeq[userID].append(clusterIndex)

        if clusterIndex not in self.globalClusterModels:
            self.globalClusterModels[clusterIndex] = globalClusterModel(dimension=self.d, alpha=self.alpha,
                                                                        lambda_=self.lambda_,
                                                                        NoiseScale=self.NoiseScale,
                                                                        delta_1=self.delta_1)

        maxPTA = float('-inf')
        articlePicked = None

        for x in pool_articles:
            x_pta = self.globalClusterModels[clusterIndex].getProb(x.contextFeatureVector[:self.d])
            # pick article with highest Prob
            if maxPTA < x_pta:
                articlePicked = x
                maxPTA = x_pta


        self.cluster = []
        for u in self.usersClusterIndexSeq.keys():
            if self.usersClusterIndexSeq[u][-1] == clusterIndex:
                self.cluster.append(u)

        return articlePicked

    def updateParameters(self, articlePicked, click, userID, clusterIndex):
        self.globalClusterModels[clusterIndex].updateParameters(articlePicked.contextFeatureVector[:self.d], click)

    def getTheta(self, userID):
        return self.globalClusterModels[self.usersClusterIndexSeq[userID][-1]].UserTheta