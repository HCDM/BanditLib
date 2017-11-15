from Recommendation import Recommendation

class BaseAlg():
	def __init__(self, arg_dict):  # n is number of users
		for key in arg_dict:
			setattr(self, key, arg_dict[key])

		self.estimates = {}
		self.estimates['CanEstimateUserPreference'] = False
		self.estimates['CanEstimateCoUserPreference'] = False 
		self.estimates['CanEstimateW'] = False
		self.estimates['CanEstimateV'] = False

	def getEstimateSettings(self):
		return self.estimates

	def decide(self, pool_articles, userID, exclude = []):
		return pool_articles[len(exclude)]

	def createRecommendation(self, pool_articles, userID, k):
		articles = []
		for x in range(k):
			articlePicked = self.decide(pool_articles, userID, articles)
			articles.append(articlePicked)
		recommendation = Recommendation(k, articles)
		return recommendation

	def updateParameters(self, articlePicked, click, userID):
		pass

	def updateRecommendationParameters(self, recommendation, reward, userID):
		for i in range(recommendation.k):
			self.updateParameters(recommendation.articles[i], reward[i], userID)