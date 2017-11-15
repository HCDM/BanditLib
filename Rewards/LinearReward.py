from Reward import Reward
import numpy as np

class LinearReward(Reward):
	def __init__(self, k, reward_dict):
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
		return np.dot(self.getTheta(user), pickedArticle.featureVector)
		#return eval(self.reward_function)

	def getRecommendationReward(self, user, recommendation, noise):
		print "get linear recommendation reward"
		total = 0
		rewardList = []
		for i in recommendation.articles:
			articleReward = self.getReward(user, i) + noise
			total += articleReward
			rewardList.append(articleReward)
		print "Total: " + str(total)
		return (total/self.k), rewardList

	# def getOptimalRecommendationReward(self, user, articlePool, k):
	# 	total = 0
	# 	local_pool = articlePool
	# 	for x in range(k):
	# 		articleReward, articlePicked = self.getOptimalReward(user, local_pool)
	# 		total += articleReward
	# 		local_pool.remove(articlePicked)
	# 	return total/k

	# def getOptimalReward(self, user, articlePool):		
	# 	maxReward = float('-inf')
	# 	maxx = None
	# 	for x in articlePool:	 
	# 		reward = self.getReward(user, x)
	# 		if reward > maxReward:
	# 			maxReward = reward
	# 			maxx = x
	# 	return maxReward, x
