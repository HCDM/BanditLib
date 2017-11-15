import numpy as np 
from util_functions import featureUniform, gaussianFeature, fileOverWriteWarning
import json
from random import choice, randint
from Users import User, UserManager

class CoUser(User):
	def __init__(self, id, theta = None, CoTheta = None):
		User.__init__(self, id, theta = theta)
		self.CoTheta = CoTheta


class CoUserManager(UserManager):
	def __init__(self, dimension, user_dict, argv = None):
		UserManager.__init__(self, dimension, user_dict, argv = None)
		self.matrixNoise = argv['matrixNoise']
		self.sparseLevel = argv['sparseLevel']
		self.W, self.W0 = self.constructAdjMatrix(self.sparseLevel)

	def loadUsers(self, filename):
		users = []
		with open(filename, 'r') as f:
			for line in f:
				id, theta = json.loads(line)
				users.append(CoUser(id, np.array(theta)))
		return users

	def simulateThetafromUsers(self):
		usersids = {}
		users = []
		mask = self.generateMasks()

		if (self.UserGroups == 0):
			for key in range(self.userNum):
				thetaVector = self.thetaFunc(self.dimension, argv = self.argv)
				l2_norm = np.linalg.norm(thetaVector, ord =2)
				users.append(CoUser(key, thetaVector/l2_norm))
		else:
			for i in range(self.UserGroups):
				usersids[i] = range(self.userNum*i/self.UserGroups, (self.userNum*(i+1))/self.UserGroups)

				for key in usersids[i]:
					thetaVector = np.multiply(self.thetaFunc(self.dimension, argv = self.argv), mask[i])
					l2_norm = np.linalg.norm(thetaVector, ord =2)
					users.append(CoUser(key, thetaVector/l2_norm))
		return users


	def constructGraph(self):
		n = len(self.users)	

		G = np.zeros(shape = (n, n))
		for ui in self.users:
			for uj in self.users:
				G[ui.id][uj.id] = np.dot(ui.theta, uj.theta) # is dot product sufficient
		return G
		
	def constructAdjMatrix(self, m):
		n = len(self.users)	

		G = self.constructGraph()
		W = np.zeros(shape = (n, n))
		W0 = np.zeros(shape = (n, n)) # corrupt version of W
		for ui in self.users:
			for uj in self.users:
				W[ui.id][uj.id] = G[ui.id][uj.id]
				sim = W[ui.id][uj.id] + self.matrixNoise() # corrupt W with noise
				if sim < 0:
					sim = 0
				W0[ui.id][uj.id] = sim
				
			# find out the top M similar users in G
			if m>0 and m<n:
				similarity = sorted(G[ui.id], reverse=True)
				threshold = similarity[m]				
				
				# trim the graph
				for i in range(n):
					if G[ui.id][i] <= threshold:
						W[ui.id][i] = 0;
						W0[ui.id][i] = 0;
					
			W[ui.id] /= sum(W[ui.id])
			W0[ui.id] /= sum(W0[ui.id])

		return [W, W0]
	
	def getW(self):
		return self.W
	def getW0(self):
		return self.W0
	def getFullW(self):
		return self.FullW
	
	def CoTheta(self):
		for ui in self.users:
			ui.CoTheta = np.zeros(self.dimension)
			for uj in self.users:
				ui.CoTheta += self.W[uj.id][ui.id] * np.asarray(uj.theta)
			#print 'Users', ui.id, 'CoTheta', ui.CoTheta	