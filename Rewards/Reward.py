import numpy as np

class Reward():
	def __init__(self, arg_dict = {}):
		for key in arg_dict:
			setattr(self, key, arg_dict[key])

	# def getOptimalRecommendationReward(self, user, articlePool, k):
	# 	total = 0
	# 	prev_selections = []
	# 	for x in range(k):
	# 		articleReward, articlePicked = self.getOptimalReward(user, articlePool, prev_selections)
	# 		total += articleReward
	# 		prev_selections.append(articlePicked)
	# 		#local_pool.remove(articlePicked)
	# 	return total/k

	def getOptimalReward(self, user, articlePool, exclude = []):
		art_features = np.empty([len(articlePool), len(articlePool[0].featureVector)])
		for i in range(len(articlePool)):
			art_features[i, :] = articlePool[i].featureVector
		user_features = self.get_user_features(user)
		reward_matrix = np.dot(art_features, user_features)
		pool_position = np.argmax(reward_matrix)
		return reward_matrix[pool_position], articlePool[pool_position]

	def getRecommendationReward(self, user, recommendation, noise):
		max_reward = float('-inf')
		max_article = None
		for i in recommendation.articles:
			articleReward = self.getReward(user, i) + noise
			if articleReward > max_reward:
				max_reward = articleReward
				max_article = i
		return max_reward, max_article

