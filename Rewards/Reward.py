import numpy as np

class Reward():
	def __init__(self, k):
		self.k = k

	def getOptimalRecommendationReward(self, user, articlePool, k):
		total = 0
		prev_selections = []
		for x in range(k):
			articleReward, articlePicked = self.getOptimalReward(user, articlePool, prev_selections)
			total += articleReward
			prev_selections.append(articlePicked)
			#local_pool.remove(articlePicked)
		return total/k

	def getOptimalReward(self, user, articlePool, exclude = []):
		art_features = np.empty([len(articlePool), len(articlePool[0].featureVector)])
		for i in range(len(articlePool)):
			art_features[i, :] = articlePool[i].featureVector
		user_features = self.get_user_features(user)
		reward_matrix = np.dot(art_features, user_features)
		pool_position = np.argmax(reward_matrix)
		return reward_matrix[pool_position], articlePool[pool_position]

 #    ### Broadcasting Here #######
	# def getOptimalReward(self, user, articlePool, exclude = []):
	# 	maxReward = float('-inf')
	# 	maxx = None
	# 	for x in articlePool:
	# 		reward = self.getReward(user, x)
	# 		if reward > maxReward and x not in exclude:
	# 		#if reward > maxReward:
	# 			maxReward = reward
	# 			maxx = x
	# 	return maxReward, x