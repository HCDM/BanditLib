from LinUCB import LinUCBAlgorithm
from Recommendation import IncentivizedRecommendation
import random
import numpy as np



class FairUCBAlgorithm(LinUCBAlgorithm):

	def __init__(self, arg_dict):
		LinUCBAlgorithm.__init__(self, arg_dict)


	def createIncentivizedRecommendation(self, pool_articles, userID, k):
		chain = []
		top_of_chain = float('-inf')
		bottom_of_chain = float('-inf')

		for x in pool_articles:
			x_pta, mean, var = self.users[userID].getProb_plot(self.alpha, x.contextFeatureVector[:self.dimension])
			if x_pta - 2*var > top_of_chain:
				chain = [(x, var)]
				top_of_chain = x_pta
				bottom_of_chain = x_pta - 2*var
			elif x_pta > bottom_of_chain:
				chain.append((x, var))
				if x_pta > top_of_chain:
					top_of_chain = x_pta
				if x_pta - 2*var < bottom_of_chain:
					bottom_of_chain = x_pta - 2*var

		articles, incentives = [], []
		if len(chain) == 0 :
			articles.append(chain[0][0])
			incentives.append(0)
		else:
			for i in range(len(chain)):
				articles.append(chain[i][0])
				incentives.append(0)

		recommendation = IncentivizedRecommendation(len(articles), articles, incentives)
		return recommendation

	def decide(self, pool_articles, userID, k = 1, multiplier = []):
		# MEAN
		art_features = np.empty([len(pool_articles), len(pool_articles[0].contextFeatureVector[:self.dimension])])
		for i in range(len(pool_articles)):
			art_features[i, :] = np.multiply(pool_articles[i].contextFeatureVector[:self.dimension], multiplier[:self.dimension])
		user_features = self.users[userID].UserTheta
		mean_matrix = np.dot(art_features, user_features)

		# VARIANCE
		var_matrix = np.sqrt(np.dot(np.dot(art_features, self.users[userID].AInv), art_features.T).clip(0))
		pta_matrix = mean_matrix + self.alpha*np.diag(var_matrix)


		pool_positions = np.argsort(pta_matrix)[(k*-1):]
		articles = []
		values = []
		for i in range(k):
			articles.append(pool_articles[pool_positions[i]])
			values.append(pta_matrix[pool_positions[i]])
		return articles, values
