from Reward import Reward
import numpy as np

class SocialLinearReward(Reward):
	def __init__(self, arg_dict = {}, Gepsilon = 1):
		Reward.__init__(self, arg_dict)

		#self.GW = self.constructLaplacianMatrix(W, Gepsilon)

	def getReward(self, user, pickedArticle):
		# How to conditionally change
		return np.dot(user.CoTheta, pickedArticle.featureVector)

	# def getRecommendationReward(self, user, recommendation, noise, cotheta = False):
	# 	total = 0
	# 	rewardList = []
	# 	for i in recommendation.articles:
	# 		articleReward = np.dot(user.CoTheta, i.featureVector) + noise
	# 		total += articleReward
	# 		rewardList.append(articleReward)
	# 	return (total/self.k), rewardList

	def get_user_features(self, user):
		return user.CoTheta

	def constructLaplacianMatrix(self, W, Gepsilon):
		G = W.copy()
		#Convert adjacency matrix of weighted graph to adjacency matrix of unweighted graph
		for i in self.users:
			for j in self.users:
				if G[i.id][j.id] > 0:
					G[i.id][j.id] = 1	

		L = csgraph.laplacian(G, normed = False)
		print L
		I = np.identity(n = G.shape[0])
		GW = I + Gepsilon*L  # W is a double stochastic matrix
		print 'GW', GW
		return GW.T
	
	def getGW(self):
		return self.GW

	def getTheta(self):
		Theta = np.zeros(shape = (self.dimension, len(self.users)))
		for i in range(len(self.users)):
			Theta.T[i] = self.users[i].theta
		return Theta
	def generateUserFeature(self,W):
		svd = TruncatedSVD(n_components=20)
		result = svd.fit(W).transform(W)
		return result