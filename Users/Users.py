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
	def __init__(self, dimension, user_dict, argv = None):
		self.dimension = dimension
		self.thetaFunc = eval(user_dict['thetaFunc']) if user_dict.has_key('thetaFunc') else featureUniform
		self.userNum = user_dict['number'] if user_dict.has_key('number') else 10
		self.UserGroups = user_dict['groups'] if user_dict.has_key('groups') else 5
		self.argv = argv
		self.signature = "A-"+"+PA"+"+TF-"+self.thetaFunc.__name__
		if user_dict.has_key('load') and user_dict['load']:
			# Load from user file
			self.users = self.loadUsers(user_dict['filename']) if user_dict.has_key('filename') else self.loadUsers(user_dict['default_file'])
		else:
			# Simulate random users
			self.users = self.simulateThetafromUsers()
			if user_dict.has_key('save') and user_dict['save']:
				self.saveUsers(users, user_dict['default_file'], force = False)

		# How should W be set up for this type of Users
		self.W, self.W0 = self.constructZeroMatrix()

	def getUsers(self):
		return self.users

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

	def constructZeroMatrix(self):
		n = len(self.users)	
        # Identity Matrix instead, so CoLin equivalent to LinUCB
		W = np.identity(n = n)
		W0 = np.identity(n = n)
		return [W, W0] 

	def getW(self):
		return self.W
	def getW0(self):
		return self.W0

	def CoTheta(self):
		for ui in self.users:
			ui.CoTheta = ui.theta
			print 'Users', ui.id, 'CoTheta', ui.CoTheta
