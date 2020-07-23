# Multi Layer Perceptron Algorithm
# They only work on the datasets (Yahoo, LastFM, Delicious) and not on current simulation

import numpy as np
import torch
import random
import math

from util_functions import vectorize
from Recommendation import Recommendation
from BaseAlg import BaseAlg


class MLP(torch.nn.Module):
	def __init__(self, input_dim, hidden_dim, threshold, learning_rate, perturb_type='normal', n=1):
		super(MLP, self).__init__()
		self.linear1 = torch.nn.Linear(input_dim, hidden_dim, bias=True)
		self.linear2 = torch.nn.Linear(hidden_dim, 1, bias=True)
		self.relu = torch.nn.ReLU(inplace=False)
		self.sigmoid = torch.nn.Sigmoid()
		self.linear2.weight.data.fill_(1)
		self.linear1.weight.data.fill_(1)
		self.linear1.bias.data.fill_(1)
		self.linear2.bias.data.fill_(1)
		self.perturb_type = perturb_type
		self.n = n
		self.loss_function = torch.nn.MSELoss()
		self.threshold = threshold
	#	# UPDATE: add L2 regularization through setting weight_decay
		self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, weight_decay=1e-3)


	def forward(self, article_FeatureVector):
		output_layer_one = self.linear1(article_FeatureVector)
                output_layer_one = self.relu(output_layer_one)
		output_layer_two = self.linear2(output_layer_one)
		pred = self.relu(output_layer_two)	
		return article_FeatureVector, output_layer_one, pred 

        # for each datapoint take a step
	# This is an gradient descent implementation
	def update_model(self, article_FeatureVectors, clicks, perturb_scale=0):
		self.train()
		# UPDATE: multiple updates till converge
		prev_loss = float('inf') 
		i = 0
		while i < 100:
			i += 1
			self.optimizer.zero_grad()
			_, _, pred = self.forward(article_FeatureVectors)
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
		if self.perturb_type == 'binomial':
			layer.weight.grad  = layer.weight.grad + torch.tensor(np.random.binomial(self.n, perturb_scale, size=size)).float()
		else:
			layer.weight.grad  = layer.weight.grad + torch.tensor(np.random.normal(scale=perturb_scale, size=size)).float()


class MLPUserStruct:
	def __init__(self, input_dim, hidden_dim, threshold, device, learning_rate, perturb_type='normal', n=1):
		self.device = device
		self.mlp = MLP(input_dim, hidden_dim, threshold, learning_rate, perturb_type=perturb_type, n=n)
		self.mlp.to(device=self.device)
		self.mlp.eval()
		self.article_FeatureVectors = torch.empty(0, input_dim).to(device=self.device)
		self.articles = []
		self.clicks_list = []
		self.history = []
		self.clicks = torch.empty(0,1).to(device=self.device)
	
	# Use single sample to update
	def updateParameters(self, article_FeatureVector, click, perturb_scale=0):
		article_FeatureVector = torch.tensor([article_FeatureVector]).float().to(device=self.device)
		click = torch.tensor([[click]]).float().to(device=self.device)
		return self.mlp.update_model(article_FeatureVector, click, perturb_scale)

	# Uses entire history to update
	#def updateParameters(self, article_FeatureVector, click, perturb_scale=0):
	#	article_FeatureVector = torch.tensor([article_FeatureVector]).float().to(device=self.device)
	#	click = torch.tensor([[click]]).float().to(device=self.device)
	#	self.article_FeatureVectors = torch.cat((self.article_FeatureVectors, article_FeatureVector), 0)
	#	self.clicks = torch.cat((self.clicks, click), 0)
	#	return self.mlp.update_model(self.article_FeatureVectors, self.clicks, perturb_scale)

	# Uses new datapoint and random subset of history to update
	# In the future should have the ability to select n different subsets of history updating the model on each subset
	#def updateParameters(self, article_FeatureVector, click, perturb_scale=0, number_sample=4):
	#	feature_subset = [] 
	#	click_subset = [] 
	#	if len(self.history) < number_sample:
	#		self.history.append((article_FeatureVector, click))
	#		feature_subset, click_subset = map(list, zip(*self.history))
	#	else:
	#		samp = random.sample(self.history, number_sample-1)
	#		feature_subset, click_subset = map(list, zip(*samp))
	#		feature_subset.append(article_FeatureVector)
	#		click_subset.append(click)
	#		self.history.append((article_FeatureVector, click))

	#	feature_subset = torch.tensor(feature_subset).float()
	#	click_subset = torch.tensor(click_subset).float()
	#	
	#	return self.mlp.update_model(feature_subset, click_subset, perturb_scale)
	
	def getProb(self, article_FeatureVector):
		_, _,  pred = self.mlp(torch.from_numpy(article_FeatureVector).float().to(device=self.device))
		return pred 
	
	# Needed for UCBMLP
	# Returns the input values of each layer and is used to calculate confidence bound of input 
	# I think this could be done with a getter in the MLP class and the two getProb functions could be avoided
	def getProbAndLayerInputs(self, article_FeatureVector):
		i_one, i_two, pred = self.mlp(torch.from_numpy(article_FeatureVector).float().to(device=self.device))
		input_one = i_one.clone().detach()
		input_two = i_two.clone().detach()
		return input_one, input_two, pred


#--------------- MLP Algorithm (Each user has its own MLP) ---------------
class MLPAlgorithm(BaseAlg):
	def __init__(self, arg_dict):  # n is number of users
		BaseAlg.__init__(self, arg_dict)
	        device = torch.device('cpu')	
		# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
			# Currently running all pytorch tensors on CPU
			# A each step we need to create new tensor and put it on GPU
			# In this case using GPU is slower 
		self.users = []
		#algorithm have n users, each user has a user structure
		for i in range(self.n_users):
			self.users.append(MLPUserStruct(self.dimension, self.hidden_layer_dimension, self.threshold, device, self.learning_rate))

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
		return self.users[userID].updateParameters(articlePicked.contextFeatureVector[:self.dimension], click)


#--------------- MLP Perturbed Reward Algorithm (Each user has its own MLP) ---------------
class PerturbedRewardMLPAlgorithm(MLPAlgorithm):
	def __init__(self, arg_dict):
		MLPAlgorithm.__init__(self, arg_dict)

	def updateParameters(self, articlePicked, click, userID):
		click +=  np.random.normal(scale=self.perturb_scale)
		self.users[userID].updateParameters(articlePicked.contextFeatureVector[:self.dimension], click)


#--------------- MLP Perturbed Gradient Algorithm (Each user has its own MLP) ---------------
class PerturbedGradientMLPAlgorithm(MLPAlgorithm):
	def __init__(self, arg_dict):
		MLPAlgorithm.__init__(self, arg_dict)

	def updateParameters(self, articlePicked, click, userID):
		self.users[userID].updateParameters(articlePicked.contextFeatureVector[:self.dimension], click, self.perturb_scale)


#--------------- Epsilon Greedy MLP Algorithm (Each user has its own MLP) ---------------
class EGreedyMLPAlgorithm(MLPAlgorithm):
	def __init__(self, arg_dict):
		MLPAlgorithm.__init__(self, arg_dict)

	def decide(self, pool_articles, userID, k = 1):
		# Random explore with probability epilon
		if random.random() < self.epsilon:
			return [random.choice(pool_articles)]

		maxPTA = float('-inf')
		articlePicked = None

		for x in pool_articles:
			x_pta = self.users[userID].getProb(x.contextFeatureVector[:self.dimension])
			if maxPTA < x_pta:
				articlePicked = x
				maxPTA = x_pta

		return [articlePicked]


#--------------- Single MLP Algorithm (All users share one MLP) ---------------
class MLPSingleAlgorithm(BaseAlg):
	def __init__(self, arg_dict):  
		BaseAlg.__init__(self, arg_dict)
		#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	        device = torch.device('cpu')	
		self.mlp = MLPUserStruct(self.dimension, self.hidden_layer_dimension, self.threshold, device, self.learning_rate)

	def decide(self, pool_articles, userID, k = 1):
		maxPTA = float('-inf')
		articlePicked = None

		for x in pool_articles:
			x_pta = self.mlp.getProb(x.contextFeatureVector[:self.dimension])
			if maxPTA < x_pta:
				articlePicked = x
				maxPTA = x_pta

		return [articlePicked]

	def updateParameters(self, articlePicked, click, userID):
		return self.mlp.updateParameters(articlePicked.contextFeatureVector[:self.dimension], click)


#--------------- Single MLP Algorithm (All users share one MLP) ---------------
class PerturbedRewardMLPSingleAlgorithm(MLPSingleAlgorithm):
	def __init__(self, arg_dict):	
		MLPSingleAlgorithm.__init__(self, arg_dict)

	def updateParameters(self, articlePicked, click, userID):
		click +=  np.random.normal(scale=self.perturb_scale)
		self.mlp.updateParameters(articlePicked.contextFeatureVector[:self.dimension], click)


class PerturbedGradientMLPSingleAlgorithm(MLPSingleAlgorithm):
	def __init__(self, arg_dict):
		MLPSingleAlgorithm.__init__(self, arg_dict)

	def updateParameters(self, articlePicked, click, userID):
		self.mlp.updateParameters(articlePicked.contextFeatureVector[:self.dimension], click, self.perturb_scale)



#--------------- Upper Confidence Bound MLP Algorithm (All users share one MLP) ---------------
class UCBMLPAlgorithm(MLPSingleAlgorithm):
	def __init__(self, arg_dict):
		MLPSingleAlgorithm.__init__(self, arg_dict)
		self.lambda_ = arg_dict['lambda_']
		self.alpha_ = arg_dict['alpha']
	        device = torch.device('cpu')	
		self.A_one = self.lambda_*np.identity(n=self.dimension) 
		self.A_oneInv = np.linalg.inv(self.A_one)
		self.A_two = self.lambda_*np.identity(n=self.hidden_layer_dimension) 
		self.A_twoInv = np.linalg.inv(self.A_two)
	

	def decide(self, pool_articles, userID, k = 1):
		maxPTA = float('-inf')
		articlePicked = None

		for x in pool_articles:
			input_one, input_two, x_pta =  \
				self.mlp.getProbAndLayerInputs(x.contextFeatureVector[:self.dimension])
			x_pta += self.add_confidence_bounds(input_one, input_two)			
			if maxPTA < x_pta:
				articlePicked = x
				maxPTA = x_pta
				self.input_two = input_two

		return [articlePicked]
	
	# This is the same idea found in LinUCB where we approxmiate uncertainty about an arm 
	# We approximate uncertaintely at each layer of the MLP
	def add_confidence_bounds(self, input_one, input_two):
		total_neurons = self.dimension + self.hidden_layer_dimension
		cb_one = np.sqrt(np.dot(np.dot(input_one, self.A_oneInv),  input_one)) * self.dimension/total_neurons
		cb_two = np.sqrt(np.dot(np.dot(input_two, self.A_twoInv),  input_two)) * self.hidden_layer_dimension/total_neurons
		var = self.alpha * (cb_one + cb_two) 
		return var 
		
	def updateParameters(self, articlePicked, click, userID):	
		feature_vector = articlePicked.contextFeatureVector[:self.dimension]
		self.A_one += np.outer(feature_vector, feature_vector)
		self.A_two += np.outer(self.input_two, self.input_two)
		self.A_oneInv = np.linalg.inv(self.A_one)
		self.A_twoInv = np.linalg.inv(self.A_two)

		self.mlp.updateParameters(feature_vector, click)

