import numpy as np 
from util_functions import featureUniform, gaussianFeature, fileOverWriteWarning
import json
from random import choice, randint
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

class User():
	def __init__(self, id, theta = None,  beta= None,  userFeature = None, Theta_userFeature = None, Beta_userFeature = None):
		self.id = id
		self.theta = theta
		#self.CoTheta = CoTheta
		self.uncertainty = 0.0
		self.beta = beta
		self.returnProb = 1.0
		self.articlesList = []
		self.userFeature = userFeature
		self.Theta_userFeature = Theta_userFeature
		self.Beta_userFeature = Beta_userFeature
	def updateReturnProb(self, returnProb):
		self.returnProb = returnProb
	def getArticleList(self, articlesList):
		self.articlesList = articlesList
	def setUserFeature(self, userFeature):
		self.userFeature = userFeature
	def setTheta_userFeature(self, Theta_userFeature):
		self.Theta_userFeature = Theta_userFeature
	def setBeta_userFeature(self, Beta_userFeature):
		self.Beta_userFeature = Beta_userFeature


	'''
	def separateArticle(self, articles):
		for a in articles:
	'''


class UserManager():
	def __init__(self, dimension, userNum, thetaFunc, betaFunc, argv = None):
		self.dimension = dimension
		self.thetaFunc = thetaFunc
		self.betaFunc = betaFunc
		self.userNum = userNum
		self.argv = argv
		self.signature = "A-"+"+PA"+"+TF-"+self.thetaFunc.__name__ +'-'+self.betaFunc.__name__

	def saveUsers(self, users, filename, force = False):
		fileOverWriteWarning(filename, force)
		with open(filename, 'w') as f:
			for i in range(len(users)):
				print users[i].theta, users[i].userFeature
				f.write(json.dumps((users[i].id, users[i].theta.tolist(), users[i].beta.tolist(), users[i].userFeature.tolist() )) + '\n')
				
	def loadUsers(self, filename):
		users = []
		with open(filename, 'r') as f:
			for line in f:
				id, theta, beta, userFeature = json.loads(line)
				users.append(User(id, np.array(theta), np.array(beta), np.array(userFeature)))
		return users


	def simulateThetafromUsers(self):
		usersids = {}
		users = []
		Theta_userModel = []
		Beta_userModel = []
		userModel = []
		for i in range(self.userNum):
			thetaVector = self.thetaFunc(self.dimension, argv = self.argv)
			theta_l2_norm = np.linalg.norm(thetaVector, ord =2)

			betaVector = self.betaFunc(self.dimension, argv = self.argv)
			beta_l2_norm = np.linalg.norm(betaVector, ord =2) 

			FinalthetaModel = np.array(thetaVector)/float(theta_l2_norm)
			FinalbetaModel  = np.array(betaVector)/float(beta_l2_norm)

			#Generate UserFeature
			X =  FinalthetaModel.tolist() + FinalbetaModel.tolist() 
			userModel.append(X)
			Theta_userModel.append(FinalthetaModel.tolist())
			Beta_userModel.append(FinalbetaModel.tolist())
			
			users.append(User(i, FinalthetaModel , FinalbetaModel))
		#print userModel
		userModel = np.array(userModel)
		pca = PCA(n_components= 5)
		userFeature = pca.fit_transform(userModel)

		Theta_userModel = np.array(Theta_userModel)
		pca_theta = PCA(n_components= 5)
		Theta_userFeature = pca_theta.fit_transform(Theta_userModel)

		Beta_userModel = np.array(Beta_userModel)
		pca_beta= PCA(n_components= 5)
		Beta_userFeature = pca_beta.fit_transform(Beta_userModel)

		#print len(userFeature), userFeature
		for i in range(self.userNum):
			users[i].setUserFeature(userFeature[i])
			users[i].setTheta_userFeature(Theta_userFeature[i])
			users[i].setBeta_userFeature(Beta_userFeature[i])
	
		return users

