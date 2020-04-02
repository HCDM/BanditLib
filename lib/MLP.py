import numpy as np
from util_functions import vectorize
from Recommendation import Recommendation
from BaseAlg import BaseAlg
import torch

class MLP(torch.nn.Module):
	def __init__(self, input_dim, hidden_dim):
		super(MLP, self).__init__()
		self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
		self.linear2 = torch.nn.Linear(hidden_dim, 1)
		self.loss_function = torch.nn.MSELoss()
		self.optimizer = torch.optim.SGD(self.parameters(), lr = 0.1)
		self.reward_pred = 0

	def forward(self, x):
		h_relu = self.linear1(x).clamp(min=0)
                pred = self.linear2(h_relu)
		return pred

	def update_model(self, article_FeatureVector, click):
		self.train()
		self.optimizer.zero_grad()   # zero the gradient buffers
	        pred = self.forward(article_FeatureVector)	
		click = torch.tensor(np.array([click])).float()
		loss = self.loss_function(pred, click)
		loss.backward()
		self.optimizer.step() 
		self.eval()


class MLPUserStruct:
	def __init__(self, input_dim, hidden_dim):
		self.mlp = MLP(input_dim, hidden_dim)
		self.mlp.eval()

	def updateParameters(self, article_FeatureVector, click):
		feature_vector = torch.from_numpy(article_FeatureVector).float()
		self.mlp.update_model(feature_vector, click)
	
	def getProb(self, article_FeatureVector):
		y = self.mlp(torch.from_numpy(article_FeatureVector).float())
		return y 
		
#	def getProb_plot(self, alpha, article_FeatureVector):
#		mean = np.dot(self.UserTheta,  article_FeatureVector)
#		var = np.sqrt(np.dot(np.dot(article_FeatureVector, self.AInv),  article_FeatureVector))
#		pta = mean + alpha * var
#
#		return pta, mean, alpha * var
#	def getTheta(self):
#		return 0
#	
#	def getA(self):
#		return 0 
#

#---------------MLP(fixed user order) algorithm---------------
class MLPAlgorithm(BaseAlg):
	def __init__(self, arg_dict, init="zero"):  # n is number of users
		BaseAlg.__init__(self, arg_dict)
		self.users = []
		#algorithm have n users, each user has a user structure
		for i in range(arg_dict['n_users']):
			self.users.append(MLPUserStruct(arg_dict['dimension'], arg_dict['hidden_layer_dimension']))

	def decide(self, pool_articles, userID, k = 1):
		maxPTA = float('-inf')
		articlePicked = None

		for x in pool_articles:
			x_pta = self.users[userID].getProb(x.contextFeatureVector[:self.dimension])
			if maxPTA < x_pta:
				articlePicked = x
				maxPTA = x_pta

		return [articlePicked]


	def getProb(self, pool_articles, userID):
		means = []
		vars = []
		for x in pool_articles:
			x_pta, mean, var = self.users[userID].getProb_plot(self.alpha, x.contextFeatureVector[:self.dimension])
			means.append(mean)
			vars.append(var)
		return means, vars

	def updateParameters(self, articlePicked, click, userID):
		self.users[userID].updateParameters(articlePicked.contextFeatureVector[:self.dimension], click)
	

#	##### SHOULD THIS BE CALLED GET COTHETA #####
#	def getCoTheta(self, userID):
#		return self.users[userID].UserTheta
#
#	def getTheta(self, userID):
#		return self.users[userID].UserTheta
#
	# def getW(self, userID):
	# 	return np.identity(n = len(self.users))

