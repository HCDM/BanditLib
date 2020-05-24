#creates Two Layer Perceptron to predict rewards

import numpy as np
import torch
import random
import math

from util_functions import vectorize
from Recommendation import Recommendation
from BaseAlg import BaseAlg


class MLP(torch.nn.Module):
	def __init__(self, input_dim, hidden_dim, threshold):
		super(MLP, self).__init__()
		self.linear1 = torch.nn.Linear(input_dim, hidden_dim, bias=True)
		self.linear2 = torch.nn.Linear(hidden_dim, 1, bias=True)
		self.relu = torch.nn.ReLU(inplace=False)
		self.sigmoid = torch.nn.Sigmoid()
		self.linear2.weight.data.fill_(1)
		self.linear1.weight.data.fill_(1)
		self.linear1.bias.data.fill_(1)
		self.linear2.bias.data.fill_(1)
		self.linear1.weight.data = torch.tensor(np.random.gamma(2, size=[hidden_dim, input_dim], scale=1)).float()
		self.linear2.weight.data = torch.tensor(np.random.gamma(2, size=[1, hidden_dim], scale=1)).float()
		self.linear1.bias.data = torch.tensor(np.random.gamma(2, scale=1,size=hidden_dim)).float()
		self.linear2.bias.data = torch.tensor(np.random.gamma(2, scale=1, size=1)).float()
		self.loss_function = torch.nn.MSELoss()
		self.threshold = threshold
		# UPDATE: add L2 regularization through setting weight_decay
		self.optimizer = torch.optim.SGD(self.parameters(), lr = 0.1, weight_decay=1e-3)

	def forward(self, article_FeatureVector):
		output = self.linear1(article_FeatureVector)
                output = self.relu(output)
		output = self.linear2(output)
                #outputoutput = self.sigmoid(output)
		return output

        # for each datapoint take a step
	def update_model(self, article_FeatureVectors, clicks, perturb_scale=0):
		self.train()
		# UPDATE: multiple updates till converge
		prev_loss = float('inf') 
		while True:
			self.optimizer.zero_grad()
			pred = self.forward(article_FeatureVectors)
			loss = self.loss_function(pred, clicks)
			loss.backward() # computes gradient
			if perturb_scale != 0:
				self.perturb_gradient(self.linear1, perturb_scale)
				self.perturb_gradient(self.linear2, perturb_scale)
			self.optimizer.step() # updates weights
			# end while
			if (loss - prev_loss).abs() < self.threshold: # please set an appropriate threshold
				break
			prev_loss = loss
		self.eval()
		return prev_loss
	
	def perturb_gradient(self, layer, perturb_scale):
		size = layer.weight.grad.size()
		layer.weight.grad  = layer.weight.grad + torch.tensor(np.random.normal(scale=perturb_scale, size=size)).float()

class MLPUserStruct:
	def __init__(self, input_dim, hidden_dim, threshold, device):
		self.device = device
		self.mlp = MLP(input_dim, hidden_dim, threshold)
		self.mlp.to(device=self.device)
		self.mlp.eval()
		self.article_FeatureVectors = torch.empty(0,input_dim).to(device=self.device)
		self.clicks = torch.empty(0,1).to(device=self.device)

	def updateParameters(self, article_FeatureVector, click, perturb_scale=0):
		article_FeatureVector = torch.tensor([article_FeatureVector]).float().to(device=self.device)
		click = torch.tensor([[click]]).float().to(device=self.device)
		self.article_FeatureVectors = torch.cat((self.article_FeatureVectors, article_FeatureVector), 0)
		self.clicks = torch.cat((self.clicks, click), 0)
		return self.mlp.update_model(self.article_FeatureVectors, self.clicks, perturb_scale)

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
		for i in range(self.n_users):
			self.users.append(MLPUserStruct(self.dimension, self.hidden_layer_dimension, self.threshold, device))

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
class PerturbedRewardMLPAlgorithm(MLPAlgorithm):
	def __init__(self, arg_dict):
		MLPAlgorithm.__init__(self, arg_dict)

	def updateParameters(self, articlePicked, click, userID):
		click +=  np.random.normal(scale=self.perturb_scale)
		self.users[userID].updateParameters(articlePicked.contextFeatureVector[:self.dimension], click)


class PerturbedGradientMLPAlgorithm(MLPAlgorithm):
	def __init__(self, arg_dict):
		MLPAlgorithm.__init__(self, arg_dict)

	def updateParameters(self, articlePicked, click, userID):
		self.users[userID].updateParameters(articlePicked.contextFeatureVector[:self.dimension], click, self.perturb_scale)


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
