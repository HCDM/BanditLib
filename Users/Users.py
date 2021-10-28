import numpy as np
from util_functions import featureUniform, gaussianFeature, fileOverWriteWarning
import json
from random import choice, randint


class User:
    def __init__(self, id, theta=None, CoTheta=None):
        self.id = id
        self.theta = theta
        self.CoTheta = CoTheta
        self.estimatedTheta = None

    def chooseArticle(self, recommendation):
        if self.estimatedTheta is None:
            self.initializeEstimatedTheta(len(self.theta), 0.1)
        bestArticle = None
        bestIncentive = 0
        bestReward = float("-inf")
        for i in range(len(recommendation.articles)):
            reward = np.dot(self.estimatedTheta, recommendation.articles[i].featureVector)
            # var = np.sqrt(np.dot(np.dot(recommendation.articles[i].featureVector, self.AInv),  recommendation.articles[i].featureVector))
            if bestReward < reward + recommendation.incentives[i]:
                bestReward = reward + recommendation.incentives[i]
                bestArticle = recommendation.articles[i]
                bestIncentive = recommendation.incentives[i]
        return bestArticle, bestIncentive

    def initializeEstimatedTheta(self, featureDimension, lambda_, init="zero"):
        self.d = featureDimension
        self.A = lambda_ * np.identity(n=self.d)
        self.b = np.zeros(self.d)
        self.AInv = np.linalg.inv(self.A)
        if init == "random":
            self.estimatedTheta = np.random.rand(self.d)
        else:
            self.estimatedTheta = np.zeros(self.d)
        self.time = 0

    def updateParameters(self, articlePicked_FeatureVector, click):
        change = np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector)
        self.A += np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector)
        self.b += articlePicked_FeatureVector * click
        self.AInv = np.linalg.inv(self.A)
        self.estimatedTheta = np.dot(self.AInv, self.b)
        self.time += 1


class UserManager:
    def __init__(self, dimension, user_dict, argv=None):
        self.dimension = dimension
        self.thetaFunc = (
            eval(user_dict["thetaFunc"]) if "thetaFunc" in user_dict else featureUniform
        )
        self.userNum = user_dict["number"] if "number" in user_dict else 10
        self.UserGroups = user_dict["groups"] if "groups" in user_dict else 5
        self.argv = argv
        self.signature = "A-" + "+PA" + "+TF-" + self.thetaFunc.__name__
        if "load" in user_dict and user_dict["load"]:
            # Load from user file
            self.users = (
                self.loadUsers(user_dict["filename"])
                if "filename" in user_dict
                else self.loadUsers(user_dict["default_file"])
            )
        else:
            # Simulate random users
            self.users = self.simulateThetafromUsers()
            if "save" in user_dict and user_dict["save"]:
                self.saveUsers(users, user_dict["default_file"], force=False)

        # How should W be set up for this type of Users
        self.W, self.W0 = self.constructZeroMatrix()

    def getUsers(self):
        return self.users

    def saveUsers(self, users, filename, force=False):
        fileOverWriteWarning(filename, force)
        with open(filename, "w") as f:
            for i in range(len(users)):
                print(users[i].theta)
                f.write(json.dumps((users[i].id, users[i].theta.tolist())) + "\n")

    def loadUsers(self, filename):
        users = []
        with open(filename, "r") as f:
            for line in f:
                id, theta = json.loads(line)
                users.append(User(id, np.array(theta)))
        return users

    def generateMasks(self):
        mask = {}
        for i in range(self.UserGroups):
            mask[i] = np.random.randint(2, size=self.dimension)
        return mask

    def simulateThetafromUsers(self):
        usersids = {}
        users = []
        mask = self.generateMasks()

        if self.UserGroups == 0:
            for key in range(self.userNum):
                thetaVector = self.thetaFunc(self.dimension, argv=self.argv)
                l2_norm = np.linalg.norm(thetaVector, ord=2)
                users.append(User(key, thetaVector / l2_norm))
        else:
            for i in range(self.UserGroups):
                usersids[i] = range(
                    self.userNum * i / self.UserGroups, (self.userNum * (i + 1)) / self.UserGroups
                )

                for key in usersids[i]:
                    thetaVector = np.multiply(
                        self.thetaFunc(self.dimension, argv=self.argv), mask[i]
                    )
                    l2_norm = np.linalg.norm(thetaVector, ord=2)
                    users.append(User(key, thetaVector / l2_norm))
        return users

    def constructZeroMatrix(self):
        n = len(self.users)
        # Identity Matrix instead, so CoLin equivalent to LinUCB
        W = np.identity(n=n)
        W0 = np.identity(n=n)
        return [W, W0]

    def getW(self):
        return self.W

    def getW0(self):
        return self.W0

    def CoTheta(self):
        for ui in self.users:
            ui.CoTheta = ui.theta
            print("Users", ui.id, "CoTheta", ui.CoTheta)
