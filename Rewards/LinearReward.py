from Reward import Reward
import numpy as np

class LinearReward(Reward):
	def __init__(self, arg_dict={}):
		Reward.__init__(self, arg_dict)
		# for key in reward_dict:
		# 	setattr(self, key, reward_dict[key])

	def getTheta(self, user):
		return user.theta

	def getReward(self, user, pickedArticle):
		return np.dot(user.theta, pickedArticle.featureVector)

	def get_user_features(self, user):
		return user.theta
