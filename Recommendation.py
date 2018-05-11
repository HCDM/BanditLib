class Recommendation():
	def __init__(self, k, articles = None):
		self.k = k
		self.articles = articles

class IncentivizedRecommendation(Recommendation):
	def __init__(self, k, articles = None, incentives = None):
		Recommendation.__init__(self, k, articles)
		self.incentives = incentives