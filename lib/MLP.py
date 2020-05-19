#Creates Two Layer Perceptron to predict rewards

import numpy as np
import torch
import random

from util_functions import vectorize
from Recommendation import Recommendation
from BaseAlg import BaseAlg


class MLP(torch.nn.Module):
	def __init__(self, input_dim, hidden_dim):
		super(MLP, self).__init__()
		self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
		self.linear2 = torch.nn.Linear(hidden_dim, 1)
		self.relu = torch.nn.ReLU(inplace=False)
		self.sigmoid = torch.nn.Sigmoid()
		self.linear1.weight.data.fill_(1)
		self.linear2.weight.data.fill_(1)
		#print(self.linear1.weight.data)
		self.loss_function = torch.nn.MSELoss()
		self.threshold = .05
		# UPDATE: add L2 regularization through setting weight_decay
		self.optimizer = torch.optim.SGD(self.parameters(), lr = 0.1, weight_decay=1e-3)

	def forward(self, article_FeatureVector):
		output = self.linear1(article_FeatureVector)
                output = self.relu(output)
		output = self.linear2(output)
                #output = self.sigmoid(output)
		return output

        # for each datapoint take a step
	def update_model(self, article_FeatureVectors, clicks):
		self.train()
		# UPDATE: multiple updates till converge
		prev_loss = float('inf')
		while True:
			self.optimizer.zero_grad()
			pred = self.forward(article_FeatureVectors)
			loss = self.loss_function(pred, clicks)
			loss.backward() # computes gradient
			self.optimizer.step() # updates weights

			# end while
			if (loss - prev_loss).abs() < self.threshold: # please set an appropriate threshold
				break
			prev_loss = loss
		self.eval()
		return prev_loss


class MLPUserStruct:
	def __init__(self, input_dim, hidden_dim, device):
		self.device = device
		self.mlp = MLP(input_dim, hidden_dim)
		self.mlp.to(device=self.device)
		self.mlp.eval()
		self.article_FeatureVectors = torch.empty(0,input_dim).to(device=self.device)
		self.clicks = torch.empty(0,1).to(device=self.device)
		

	def updateParameters(self, article_FeatureVector, click):
		article_FeatureVector = torch.tensor([article_FeatureVector]).float().to(device=self.device)
		click = torch.tensor([[click]]).float().to(device=self.device)
		self.article_FeatureVectors = torch.cat((self.article_FeatureVectors, article_FeatureVector), 0)
		self.clicks = torch.cat((self.clicks, click), 0)
                # update many times
                # print out training loss after each update
		return self.mlp.update_model(self.article_FeatureVectors, self.clicks)

	def getProb(self, article_FeatureVector):
		prob = self.mlp(torch.from_numpy(article_FeatureVector).float().to(device=self.device))
		return prob



#---------------MLP(fixed user order) algorithm---------------
class MLPAlgorithm(BaseAlg):
	def __init__(self, arg_dict):  # n is number of users
		BaseAlg.__init__(self, arg_dict)
		#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	        device = torch.device('cpu')	

		self.users = []
		#algorithm have n users, each user has a user structure
		for i in range(arg_dict['n_users']):
			self.users.append(MLPUserStruct(arg_dict['dimension'], arg_dict['hidden_layer_dimension'], device))

	def decide(self, pool_articles, userID, k = 1):
		maxPTA = float('-inf')
		articlePicked = None

		for x in pool_articles:
			x_pta = self.users[userID].getProb(x.contextFeatureVector[:self.dimension])
			if maxPTA < x_pta:
				articlePicked = x
				maxPTA = x_pta

		return articlePicked, maxPTA


	def getProb(self, pool_articles, userID):
		means = []
		vars = []
		for x in pool_articles:
			x_pta, mean, var = self.users[userID].getProb_plot(self.alpha, x.contextFeatureVector[:self.dimension])
			means.append(mean)
			vars.append(var)
		return means, vars

	def updateParameters(self, articlePicked, click, userID):
		return self.users[userID].updateParameters(articlePicked.contextFeatureVector[:self.dimension], click)


# MLP with perturbed rewards
class PMLPAlgorithm(MLPAlgorithm):
	def __init__(self, arg_dict):
		MLPAlgorithm.__init__(self, arg_dict)

	def updateParameters(self, articlePicked, click, userID):
		click +=  .05*np.random.normal(scale=.5)
		self.users[userID].updateParameters(articlePicked.contextFeatureVector[:self.dimension], click)


# Epislon Greedy MLP Algorithm
class EGreedyMLPAlgorithm(MLPAlgorithm):
	def __init__(self, arg_dict):
		MLPAlgorithm.__init__(self, arg_dict)

	def decide(self, pool_articles, userID, k = 1):
		if random.random() < self.epsilon:
			return random.choice(pool_articles), 1

		maxPTA = float('-inf')
		articlePicked = None

		for x in pool_articles:
			x_pta = self.users[userID].getProb(x.contextFeatureVector[:self.dimension])
			if maxPTA < x_pta:
				articlePicked = x
				maxPTA = x_pta

		return articlePicked, maxPTA
