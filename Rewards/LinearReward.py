from Reward import Reward
import numpy as np

class LinearReward(Reward):
	def __init__(self, k, reward_dict={}):
		Reward.__init__(self, k)
		for key in reward_dict:
			setattr(self, key, reward_dict[key])

	def getTheta(self, user):
		return user.theta

	def getReward(self, user, pickedArticle):
		# How to conditionally change
		# return np.dot(user.CoTheta, pickedArticle.featureVector)
		###########
		# Should get
		return np.dot(user.theta, pickedArticle.featureVector)
		#return eval(self.reward_function)

	def getRecommendationReward(self, user, recommendation, noise):
		total = 0
		rewardList = []
		for i in recommendation.articles:
			articleReward = self.getReward(user, i) + noise
			total += articleReward
			rewardList.append(articleReward)
		return (total/self.k), rewardList

	def get_user_features(self, user):
		return user.theta
