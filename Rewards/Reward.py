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
		maxReward = float('-inf')
		maxx = None
		for x in articlePool:	 
			reward = self.getReward(user, x)
			if reward > maxReward and x not in exclude:
				maxReward = reward
				maxx = x
		return maxReward, x