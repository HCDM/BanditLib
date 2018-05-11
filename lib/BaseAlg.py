from Recommendation import Recommendation
import numpy as np

class BaseAlg():
	def __init__(self, arg_dict):  # n is number of users
		self.dimension = 0
		for key in arg_dict:
			setattr(self, key, arg_dict[key])

		self.estimates = {}
		self.estimates['CanEstimateUserPreference'] = arg_dict['parameters']['Theta']
		self.estimates['CanEstimateCoUserPreference'] = arg_dict['parameters']['CoTheta']
		self.estimates['CanEstimateW'] = arg_dict['parameters']['W']
		self.estimates['CanEstimateV'] = arg_dict['parameters']['V']

	def getEstimateSettings(self):
		return self.estimates

	def decide(self, pool_articles, userID, k = 1):
		pass

	def createRecommendation(self, pool_articles, userID, k):
		articles = self.decide(pool_articles, userID, k)
		recommendation = Recommendation(k, articles)
		return recommendation

	def updateParameters(self, articlePicked, click, userID):
		pass

	def updateRecommendationParameters(self, recommendation, reward, userID):
		for i in range(recommendation.k):
			self.updateParameters(recommendation.articles[i], reward[i], userID)

	def getV(self, articleID):
		if self.dimension == 0:
			return np.zeros(self.context_dimension + self.hidden_dimension)
		else: 
			return np.zeros(self.dimension)

	def getW(self, userID):
		return np.identity(n = self.n_users)
