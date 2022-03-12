import numpy as np
from scipy.special import erfinv
import math
from bidict import bidict
from scipy.stats import chi2
from copy import deepcopy

class localUserModel:
    def __init__(self, userID, dimension, alpha, lambda_, NoiseScale, delta_1, delta_2, createTime, eta, change_detection_alpha=0.01):
        self.userID = userID
        self.outDated = False
        self.d = dimension
        self.alpha = alpha  # use constant alpha, instead of the one defined in LinUCB
        self.lambda_ = lambda_
        self.change_detection_alpha = change_detection_alpha
        self.delta_1 = delta_1
        self.delta_2 = delta_2

        # LinUCB statistics
        self.A = lambda_ * np.identity(n=self.d)
        self.b = np.zeros(self.d)
        self.AInv = np.linalg.inv(self.A)

        self.NoiseScale = NoiseScale
        self.update_num = 0  # number of times this user has been updated
        self.alpha_t = self.NoiseScale * np.sqrt(
            self.d * np.log(1 + self.update_num / (self.d * self.lambda_)) + 2 * np.log(1 / self.delta_1)) + np.sqrt(
            self.lambda_)

        # history data
        self.X = np.zeros((0, self.d))
        self.y = np.zeros((0,))

        self.UserTheta = np.zeros(self.d)
        self.UserThetaNoReg = np.zeros(self.d)

        self.rank = 0

        self.createTime = createTime        # global time when model is created
        self.time = 0                       # number of times this user has been served

        # for dLinUCB's change detector
        self.eta = eta  # upper bound of gaussian noise
        self.detectedChangePoints = [0]
        self.failList = []

    def resetLocalUserModel(self, createTime):
        self.outDated = False
        self.A = self.lambda_ * np.identity(n=self.d)
        self.b = np.zeros(self.d)
        self.AInv = np.linalg.inv(self.A)
        self.UserTheta = np.zeros(self.d)
        self.UserThetaNoReg = np.zeros(self.d)
        self.X = np.zeros((0, self.d))
        self.rank = 0
        self.y = np.zeros((0,))
        self.update_num = 0  # number of times this user has been updated
        self.alpha_t = self.NoiseScale * np.sqrt(
            self.d * np.log(1 + self.update_num / (self.d * self.lambda_)) + 2 * np.log(1 / self.delta_1)) + np.sqrt(
            self.lambda_)

        self.createTime = createTime        # global time when model is created
        self.failList = []

    def updateLocalUserModel(self, articlePicked_FeatureVector, click):
        # update LinUCB statistics
        self.A += np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector)
        self.b += articlePicked_FeatureVector * click
        self.AInv = np.linalg.inv(self.A)
        self.UserTheta = np.dot(self.AInv, self.b)
        self.UserThetaNoReg = np.dot(np.linalg.pinv(self.A-self.lambda_ * np.identity(n=self.d)), self.b)
        assert self.d == articlePicked_FeatureVector.shape[0]

        self.update_num += 1.0

        # update observation history
        self.X = np.concatenate((self.X, articlePicked_FeatureVector.reshape(1, self.d)), axis=0)
        self.y = np.concatenate((self.y, np.array([click])),axis=0)
        assert self.X.shape == (self.update_num, self.d)
        assert self.y.shape == (self.update_num, )
        self.rank = np.linalg.matrix_rank(self.X)
        self.alpha_t = self.NoiseScale * np.sqrt(
            self.d * np.log(1 + self.update_num / (self.d * self.lambda_)) + 2 * np.log(1 / self.delta_1)) + np.sqrt(
            self.lambda_)

    def getCB(self, x, useConstantAlpha = False):
        var = np.sqrt(np.dot(np.dot(x, self.AInv), x))
        if useConstantAlpha:
            return self.alpha * var
        else:
            return self.alpha_t * var

    def getInstantaneousBadness(self, articlePicked, click, method="ChiSquare"):
        # compute badness on (articlePicked, click)
        if method == "ConfidenceBound":  # This is the test statistic used in dLinUCB
            mean = np.dot(self.UserTheta, articlePicked.contextFeatureVector[:self.d])
            rewardEstimationError = np.abs(mean - click)
            if rewardEstimationError <= self.getCB(articlePicked.contextFeatureVector[:self.d]) + self.eta:
                e = 0
            else:
                e = 1
        elif method == "ChiSquare":
            if self.rank < self.d:
                e = 0
            else:
                x = articlePicked.contextFeatureVector[:self.d]
                if self.rank < self.d:
                    e = 0
                else:
                    mean = np.dot(self.UserThetaNoReg, x)
                    rewardEstimationError = (mean - click)**2
                    rewardEstimationErrorSTD = self.NoiseScale**2 * (1 + np.dot(np.dot(x, np.linalg.pinv(self.A-self.lambda_ * np.identity(n=self.d))), x))
                    df1 = 1

                    chiSquareStatistic = rewardEstimationError / rewardEstimationErrorSTD
                    p_value = chi2.sf(x=chiSquareStatistic, df=df1)
                    if p_value <= self.change_detection_alpha:  # upper bound probability of false alarm
                        e = 1
                    else:
                        e = 0
        # Update failList
        self.failList.append(e)
        return e

    def detectChangeBasedOnBadness(self, ObservationInterval):
        if len(self.failList) < ObservationInterval:
            ObservationNum = float(len(self.failList))
            badness = sum(self.failList) / ObservationNum
        else:
            ObservationNum = float(ObservationInterval)
            badness = sum(self.failList[-ObservationInterval:]) / ObservationNum
        badness_CB = math.sqrt(math.log(1.0 / self.delta_2) / (2.0 * ObservationNum))
        # test badness against threshold
        if badness > self.delta_1 + badness_CB:
            changeFlag = 1
        else:
            changeFlag = 0
        return changeFlag

class AggregatedClusterModel:
    def __init__(self, cluster):
        assert cluster != []
        self.d = cluster[0].d
        self.lambda_ = cluster[0].lambda_
        self.alpha = cluster[0].alpha
        self.delta_1 = cluster[0].delta_1

        self.A = self.lambda_ * np.identity(n=self.d)
        self.b = np.zeros(self.d)
        self.update_num = 0

        for user in cluster:
            self.A += (user.A - user.lambda_ * np.identity(n=user.d))
            self.b += user.b
            self.update_num += user.update_num

        self.AInv = np.linalg.inv(self.A)
        self.NoiseScale = cluster[0].NoiseScale
        # Estimate theta for aggregated cluster
        self.UserTheta = np.dot(self.AInv, self.b)

        self.alpha_t = self.NoiseScale * np.sqrt(
            self.d * np.log(1 + self.update_num / (self.d * self.lambda_)) + 2 * np.log(1 / self.delta_1)) + np.sqrt(
            self.lambda_)

    def getCB(self, x, useConstantAlpha = False):
        var = np.sqrt(np.dot(np.dot(x, self.AInv), x))
        if useConstantAlpha:
            return self.alpha * var
        else:
            return self.alpha_t * var

class DyClu:
    def __init__(self, dimension, alpha, lambda_, NoiseScale, tau_e, delta_1=1e-2, delta_2=1e-1, neighbor_identification_alpha=1e-2, change_detection_alpha=1e-2,
                 dataSharing = False, aggregationMethod="combine", useOutdated=True, maxNumOutdatedModels=None, disable_change_detector=False):
        self.dimension = dimension
        self.alpha = alpha
        self.lambda_ = lambda_             # regularization parameter
        self.NoiseScale = NoiseScale
        self.tau_e = tau_e                 # size of sliding window for change detection stats
        self.neighbor_identification_alpha = neighbor_identification_alpha
        self.change_detection_alpha = change_detection_alpha
        self.delta_1 = delta_1  # for reward estimation CB
        self.delta_2 = delta_2  # for change detection CB

        # only used if we detect change using dLinUCB's test statistic
        self.eta = math.sqrt(2.0)*(self.NoiseScale)*erfinv(1.0 -self.delta_1)

        self.users = {}           # store up-to-date user models
        self.outdatedModels = []  # store outdated user models
        # store clustering structure in up-to-date and outdated user models
        self.user_graph = -1 * np.ones((50, 50))
        self.userID2userIndex = bidict()
        self.cur_userIndex = 0
        self.user2OutdatedModel = {}

        # parameters to control different components of DyClu
        self.useOutdated = useOutdated  # whether or not reuse observations in outdated models
        self.maxNumOutdatedModels = maxNumOutdatedModels  # maximum number of outdated user models to keep
        self.disable_change_detector = disable_change_detector  # disable change detector

        # This is used to experiment with another way of aggregating past observations:
        # Instead of combining all observations of several independent user models to construct an estimator as in DyClu
        # e.g. by setting dataSharing = False and aggregationMethod = "combine"
        # this one shares observations among user models, and use averaged UCB of each model for arm selection
        # e.g. by setting dataSharing = True and aggregationMethod = "average"
        self.dataSharing = dataSharing
        self.aggregationMethod = aggregationMethod

        self.global_time = 0      # keep track of total iterations
        self.CanEstimateUserPreference = True
        self.CanEstimateUserCluster = True

    def decide(self, pool_articles, userID):
        # 1. Observe user and arm pool
        # if it's a new user, create a localUserModel for it
        if userID not in self.users:
            assert userID not in self.userID2userIndex
            self.users[userID] = localUserModel(userID, self.dimension, self.alpha, self.lambda_, self.NoiseScale,
                                                self.delta_1,
                                                self.delta_2, self.global_time, self.eta, change_detection_alpha=self.change_detection_alpha)
            self.userID2userIndex[userID] = self.cur_userIndex
            self.cur_userIndex += 1
            assert userID == self.users[userID].userID
            assert self.cur_userIndex == len(self.users)
            # self.cur_userIndex equals to the current number of users
            # initialize user connectedness with all others to be 1
            self.user_graph[self.userID2userIndex[userID], 0:self.cur_userIndex] = [1] * self.cur_userIndex
            # initialize all others connectedness with user to be 1
            self.user_graph[0:self.cur_userIndex, self.userID2userIndex[userID]] = [1] * self.cur_userIndex
            # check if self.user_graph is full, and resize into two times the original size, padded with -1
            assert self.user_graph.shape[0] == self.user_graph.shape[1]
            if self.user_graph.shape[0] == self.cur_userIndex:
                self.user_graph = np.pad(self.user_graph, [(0, self.user_graph.shape[0]), (0, self.user_graph.shape[0])], mode='constant', constant_values=-1)
                assert self.user_graph.shape[0] == 2 * self.cur_userIndex
            self.user2OutdatedModel[userID] = []
        # 2. Cluster estimation: loop over all other user models to compute cluster of current user
        self.cluster, clusterTheta, clusterCBsOnArmPool = self.getUserClusterStatsBasedOnUserGraph(userID, pool_articles)

        # 3. Arm selection using aggregated UCB score
        maxPTA = float('-inf')
        articlePicked = None
        for arm_index in range(len(pool_articles)):
            mean = np.dot(clusterTheta, pool_articles[arm_index].contextFeatureVector[:self.dimension])
            arm_pta = mean + clusterCBsOnArmPool[arm_index]
            # pick article with highest Prob
            if maxPTA < arm_pta:
                articlePicked = pool_articles[arm_index]
                maxPTA = arm_pta

        return articlePicked

    def updateParameters(self, articlePicked, click, userID):
        # 1. Obverse (articlePicked, click) from the current user userID
        self.global_time += 1
        self.users[userID].time += 1
        # 2. Compute one-sample homogeneity test variable
        if self.disable_change_detector:
            e = 0
        else:
            e = self.users[userID].getInstantaneousBadness(articlePicked, click, method="ChiSquare")

        if e == 0:  # if model is admissible for this observation
            # 3. User model Update
            # Update current user's parameter estimation stats and cluster estimation stats
            self.users[userID].updateLocalUserModel(articlePicked.contextFeatureVector[:self.dimension], click)
            self.resetUserConnectedness(userID)
            # 4. Cluster structure Update
            self.updateUserConnectedness(userID)
            if self.useOutdated:
                self.updateUserOutdatedModelConnectedness(userID)

        # 5. Change detection for current user
        if self.disable_change_detector:
            changeFlag = False
        else:
            changeFlag = self.users[userID].detectChangeBasedOnBadness(self.tau_e)
        if changeFlag:
            # Replace outdated user model with new one
            if self.useOutdated:
                self.users[userID].outDated = True
                self.outdatedModels.append(deepcopy(self.users[userID]))
            self.users[userID].resetLocalUserModel(self.global_time)
            self.users[userID].detectedChangePoints.append(self.users[userID].time)
            self.resetUserConnectedness(userID)

    def getUserClusterStatsBasedOnUserGraph(self, userID, pool_articles):
        cluster = []
        clusterThetaList = []
        neighborUserIndexes = np.where(self.user_graph[self.userID2userIndex[userID], 0:self.cur_userIndex] == 1)[0]
        for neighborUserIndex in neighborUserIndexes:
            neighborUser = self.users[self.userID2userIndex.inverse[neighborUserIndex]]
            cluster.append(neighborUser)
            clusterThetaList.append(neighborUser.UserTheta)
        cluster_out = []
        clusterThetaList_out = []
        if self.useOutdated:
            for outdatedModelIndex in self.user2OutdatedModel[userID]:
                cluster_out.append(self.outdatedModels[outdatedModelIndex])
                clusterThetaList_out.append(self.outdatedModels[outdatedModelIndex].UserTheta)

        if self.aggregationMethod == "average":
            if self.useOutdated:
                clusterTheta = np.mean(clusterThetaList+clusterThetaList_out, axis=0)
            else:
                clusterTheta = np.mean(clusterThetaList, axis = 0)
            assert clusterTheta.shape == (self.dimension,)
            clusterCB = []
            for x in pool_articles:
                clusterCB_x = []
                if self.useOutdated:
                    for neighborUser in cluster+cluster_out:
                        clusterCB_x.append(neighborUser.getCB(x.contextFeatureVector[:self.dimension]))
                    assert len(clusterCB_x) == len(cluster)+len(cluster_out)
                else:
                    for neighborUser in cluster:
                        clusterCB_x.append(neighborUser.getCB(x.contextFeatureVector[:self.dimension]))
                    assert len(clusterCB_x) == len(cluster)
                clusterCB.append(np.mean(clusterCB_x))
            assert len(clusterCB) == len(pool_articles)
        else:
            if self.useOutdated:
                aggregatedModel = AggregatedClusterModel(cluster+cluster_out)
            else:
                aggregatedModel = AggregatedClusterModel(cluster)
            clusterTheta = aggregatedModel.UserTheta
            assert clusterTheta.shape == (self.dimension,)
            clusterCB = []
            for x in pool_articles:
                clusterCB.append(aggregatedModel.getCB(x.contextFeatureVector[:self.dimension]))
            assert len(clusterCB) == len(pool_articles)
        return cluster, clusterTheta, clusterCB

    def updateUserConnectedness(self, userID):
        """
        test current user model with all other user models, and test if any one of them exceeds the threshold
        Then set the connectedness to 0 (both current to neighbor and neighbor to current)
        :param userID:
        :return:
        """
        currentUser = self.users[userID]
        neighborUserIndexes = [i for i in range(0, self.cur_userIndex)]
        for neighborUserIndex in neighborUserIndexes:
            userID_j = self.userID2userIndex.inverse[neighborUserIndex]
            if userID_j != userID: # no need to test current user with itself
                neighborUser = self.users[userID_j]
                connected = self.testConnectednessBetweenTwoUsers(currentUser, neighborUser)
                if not connected:
                    self.user_graph[self.userID2userIndex[userID], self.userID2userIndex[userID_j]] = 0
                    self.user_graph[self.userID2userIndex[userID_j], self.userID2userIndex[userID]] = 0

    def updateUserOutdatedModelConnectedness(self, userID):
        currentUser = self.users[userID]
        self.user2OutdatedModel[userID] = []
        # only maintain maxNumOutdatedModels outdated models if necessary
        if self.maxNumOutdatedModels is not None:
            if len(self.outdatedModels) > self.maxNumOutdatedModels:
                self.outdatedModels.pop(0)

        for outdatedModelIndex in range(len(self.outdatedModels)):
            neighborUser = self.outdatedModels[outdatedModelIndex]
            connected = self.testConnectednessBetweenTwoUsers(currentUser, neighborUser)
            if connected:
                self.user2OutdatedModel[userID].append(outdatedModelIndex)

    def testConnectednessBetweenTwoUsers(self, currentUser, neighborUser):
        """
        Cluster identification:
        Test whether two user models have the same ground-truth theta
        :param currentUser:
        :param neighborUser:
        :return:
        """
        n = currentUser.update_num
        m = neighborUser.update_num
        if n == 0 and m == 0:
            return False
        # Compute numerator
        theta_combine = np.dot(
            np.linalg.pinv(currentUser.A + neighborUser.A - 2 * self.lambda_ * np.identity(n=self.dimension)),
            currentUser.b + neighborUser.b)
        num = np.linalg.norm(np.dot(currentUser.X, (currentUser.UserThetaNoReg - theta_combine))) ** 2 + np.linalg.norm(
            np.dot(neighborUser.X, (neighborUser.UserThetaNoReg - theta_combine))) ** 2
        XCombinedRank = np.linalg.matrix_rank(np.concatenate((currentUser.X, neighborUser.X), axis=0))
        df1 = int(currentUser.rank + neighborUser.rank - XCombinedRank)
        chiSquareStatistic = num / (self.NoiseScale**2)
        p_value = chi2.sf(x=chiSquareStatistic, df=df1)
        if p_value <= self.neighbor_identification_alpha:  # upper bound probability of false alarm
            return False
        else:
            return True

    def resetUserConnectedness(self, userID):
        """
        set connectedness of userID to all others and all others to userID to 1
        :param userID:
        :return:
        """
        self.user_graph[self.userID2userIndex[userID], 0:self.cur_userIndex] = 1
        self.user_graph[0:self.cur_userIndex, self.userID2userIndex[userID]] = 1

    def getTheta(self, userID):
        return self.users[userID].UserTheta