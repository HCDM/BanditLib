#Creates Two Layer Perceptron to predict rewards

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
		self.relu = torch.nn.ReLU(inplace=True)
		self.sigmoid = torch.nn.Sigmoid()
		self.loss_function = torch.nn.BCELoss()
		self.optimizer = torch.optim.SGD(self.parameters(), lr = 0.1)

	def forward(self, article_FeatureVector):
		output = self.linear1(article_FeatureVector)
                output = self.relu(output)
		output = self.linear2(output)
                output = self.sigmoid(output)
		return output

	def update_model(self, article_FeatureVectors, clicks):
		self.train() 
		self.optimizer.zero_grad()
	        pred = self.forward(article_FeatureVectors)	
		loss = self.loss_function(pred, clicks)
		loss.backward() # computes gradient
		self.optimizer.step() # updates weights
		self.eval()


class MLPUserStruct:
	def __init__(self, input_dim, hidden_dim):
		self.mlp = MLP(input_dim, hidden_dim)
		self.mlp.eval()
		self.article_FeatureVectors = torch.empty(0,input_dim)
		self.clicks = torch.empty(0,1)

#	def updateParameters(self, article_FeatureVector, click):
#		self.article_FeatureVectors = np.append(self.article_FeatureVectors, [article_FeatureVector], axis=0)
#		self.clicks = np.append(self.clicks, click)
#		print(self.article_FeatureVectors)
#		print(self.clicks)
#		feature_vectors = torch.from_numpy(self.article_FeatureVectors).float()
#		clicks = torch.from_numpy(self.clicks).float()
#		self.mlp.update_model(feature_vectors, click)
#	
	def updateParameters(self, article_FeatureVector, click):
		article_FeatureVector = torch.tensor([article_FeatureVector]).float()
		click = torch.tensor([[click]]).float()
		self.article_FeatureVectors = torch.cat((self.article_FeatureVectors, article_FeatureVector), 0)
		self.clicks = torch.cat((self.clicks, click), 0)
		self.mlp.update_model(self.article_FeatureVectors, self.clicks)
	
	def getProb(self, article_FeatureVector):
		prob = self.mlp(torch.from_numpy(article_FeatureVector).float())
		return prob 
		

#---------------MLP(fixed user order) algorithm---------------
class MLPAlgorithm(BaseAlg):
	def __init__(self, arg_dict):  # n is number of users
		BaseAlg.__init__(self, arg_dict)
		self.users = []
		#algorithm have n users, each user has a user structure
		for i in range(arg_dict['n_users']):
			self.users.append(MLPUserStruct(arg_dict['dimension'], arg_dict['hidden_layer_dimension']))

#		if torch.cuda.is_available():
#			device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
#			print("Running on the GPU")
#		else:
#			device = torch.device("cpu")
#			print("Running on the CPU")

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

	
# MLP with perturbed rewards
class PMLPAlgorithm(MLPAlgorithm):
	def __init__(self, arg_dict):
		MLPAlgorithm.__init__(self,  arg_dict)
	
	def updateParameters(self, articlePicked, click, userID):
		click +=  np.random.binomial(self.a, .5)
		self.users[userID].updateParameters(articlePicked.contextFeatureVector[:self.dimension], click)
		


