import numpy as np 
from util_functions import featureUniform, gaussianFeature, fileOverWriteWarning
import json
from random import choice, randint

class User():
	def __init__(self, id, theta = None, CoTheta = None):
		self.id = id
		self.theta = theta
		self.CoTheta = CoTheta


class UserManager():
	def __init__(self, dimension, userNum,  UserGroups, thetaFunc, argv = None):
		self.dimension = dimension
		self.thetaFunc = thetaFunc
		self.userNum = userNum
		self.UserGroups = UserGroups
		self.argv = argv
		self.signature = "A-"+"+PA"+"+TF-"+self.thetaFunc.__name__

	def saveUsers(self, users, filename, force = False):
		fileOverWriteWarning(filename, force)
		with open(filename, 'w') as f:
			for i in range(len(users)):
				print users[i].theta
				f.write(json.dumps((users[i].id, users[i].theta.tolist())) + '\n')
				
	def loadUsers(self, filename):
		users = []
		with open(filename, 'r') as f:
			for line in f:
				id, theta = json.loads(line)
				users.append(User(id, np.array(theta)))
		return users

	def generateMasks(self):
		mask = {}
		for i in range(self.UserGroups):
			mask[i] = np.random.randint(2, size = self.dimension)
		return mask

	def simulateThetafromUsers(self):
		usersids = {}
		users = []
		mask = self.generateMasks()

		if (self.UserGroups == 0):
			for key in range(self.userNum):
				thetaVector = self.thetaFunc(self.dimension, argv = self.argv)
				l2_norm = np.linalg.norm(thetaVector, ord =2)
				users.append(User(key, thetaVector/l2_norm))
		else:
			for i in range(self.UserGroups):
				usersids[i] = range(self.userNum*i/self.UserGroups, (self.userNum*(i+1))/self.UserGroups)

				for key in usersids[i]:
					thetaVector = np.multiply(self.thetaFunc(self.dimension, argv = self.argv), mask[i])
					l2_norm = np.linalg.norm(thetaVector, ord =2)
					users.append(User(key, thetaVector/l2_norm))
		return users

