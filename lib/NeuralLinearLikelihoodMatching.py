from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy.linalg
import torch
import torch.nn as nn
import torch.optim as optim
from backpack import backpack
from backpack.extensions import BatchGrad
from lib.BaseAlg import BaseAlg, get_device, Network,Network_NL, obs_data_all
import numpy as np
from backpack import backpack, extend
import cvxpy as cvx
import math
import numpy as np
from scipy.stats import invgamma

class NeuralLinearUserStruct:
    def __init__(self, feature, featureDimension, mlp_dims,
                 epoch, batch_size, learning_rate,buffer_s):
        # create neural model
        self.feature = feature
        self.mlp_dims = [int(x) for x in mlp_dims.split(",") if x != ""]
        self.device = get_device()
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = extend(Network_NL(feature_dim=featureDimension, mlp_dims=self.mlp_dims).to(self.device))
        self.loss_func = nn.MSELoss()
        self.data = obs_data_all(buffer_s=buffer_s) # training set
        self.latent_data = obs_data_all(buffer_s=buffer_s)
        self.time = 0

    def updateParameters(self):
        self.update_model()

    def update_model(self):
        num_data = len(self.data)
        optimizer = torch.optim.Rprop(self.model.parameters(), lr=self.learning_rate,)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9999977)
        # prepare the training data
        loss_list = []
        dataloader = torch.utils.data.DataLoader(self.data, batch_size=self.batch_size, shuffle=True, num_workers=0)
        # calculate the number of batches
        num_batch = int(num_data / self.batch_size) if num_data >= self.batch_size else 1
        for i in range(self.epoch):
            total_loss = 0
            for j, batch in enumerate(dataloader):
                if j == num_batch:
                    break
                self.model.zero_grad()
                optimizer.zero_grad()
                context_feature = batch["context"].clone().detach().float().requires_grad_(True).to(self.device)
                click = batch["click"].clone().detach().float().requires_grad_(True).to(self.device).view(-1)
                pred = self.model(context_feature).view(-1)
                loss = self.loss_func(pred, click)
                total_loss += loss
                loss.backward()
                optimizer.step()
            loss_list.append(total_loss / num_batch)

            scheduler.step()


class NeuralLinearLikelihoodMatchingAlgorithm(BaseAlg):
    def __init__(self, arg_dict,):  # n is number of users
        BaseAlg.__init__(self, arg_dict)
        self.device = get_device()
        # latent_dim
        self.mlp_dims = [int(x) for x in arg_dict["mlp"].split(",") if x != ""]
        self.latent_dim = self.mlp_dims[-1]
        self.param_dim = self.latent_dim
        # create parameters
        self.contexts = None
        self.context_dim = 0
        self.buffer_s = arg_dict["mem"]
        # Gaussian prior for each beta_i
        self._lambda_prior = float(arg_dict["lambda_"])
        self.mu = np.zeros(self.param_dim)
        self.mu_prior = self.mu
        self.f = np.zeros(self.param_dim)
        self.yy = [0]
        self.cov = (1.0 / self._lambda_prior) * np.eye(self.param_dim)
        self.precision = np.zeros_like(self.cov)
        self.precision_prior = self.precision
        # Inverse Gamma prior for each sigma2_i
        self.sigma_prior_flag = arg_dict["sigma_prior_flag"]
        self.mu_prior_flag = arg_dict["mu_prior_flag"]
        self.EPSILON = 0.00001
        self._a0 = float(arg_dict["a0"])
        self._b0 = float(arg_dict["b0"])
        self.a = self._a0
        self.b = self._b0
        self.t = 0
        self.update_freq_nn = arg_dict["training_freq_network"]
        self.current_g = None
        self.users = []
        for i in range(arg_dict["n_users"]):
            self.users.append(
                NeuralLinearUserStruct([], arg_dict["dimension"], arg_dict["mlp"], arg_dict["epoch"],
                                    arg_dict["batch_size"], arg_dict["learning_rate"],buffer_s=arg_dict["mem"]))

    def decide(self, pool_articles, userID, k=1,) -> object:
        """Samples beta's from posterior, and chooses best action accordingly."""
        # Sample sigma2, and beta conditional on sigma2
        sigma2_s = self.b * invgamma.rvs(self.a)
        try:
            beta_s = np.random.multivariate_normal(self.mu, sigma2_s*self.cov)
        except np.linalg.LinAlgError as e:
            beta_s = np.random.multivariate_normal(np.zeros(self.param_dim), np.eye((self.param_dim)))
        concat_feature = np.array([x.featureVector for x in pool_articles])
        #     #! need to double check the implementation for concatenate user and item features
        tensor = torch.tensor(concat_feature, dtype=torch.float32).to(self.users[userID].device)
        # forward
        z_context = self.users[userID].model(tensor,path="last")
        z_context_=z_context.clone().detach().numpy()
        self.contexts = z_context_
        vals = np.dot(beta_s,z_context_.T)
        pool_positions = np.argsort(vals)[(k * -1):]
        articles = []
        for i in range(k):
            articles.append(pool_articles[pool_positions[i]])
        return articles

    def calc_precision_prior(self, new_z_):
        precision_return = []
        new_z_ = np.array(new_z_).reshape(-1,10)
        n, m = new_z_.shape
        prior = (self.EPSILON) * np.eye(self.param_dim)

        if self.cov is not None:
            precision = self.cov

            # compute  confidence scores for old data
            d = []
            for c in self.contexts[:]:
                d.append(np.dot(np.dot(c, precision), c.T))
            # compute new data correlations
            phi = []
            for c in self.contexts[:]:
                phi.append(np.outer(c,c))

            X = cvx.Variable((m,m), PSD=True)

            obj = cvx.Minimize(sum([(cvx.trace(X*phi[i]) - d[i]) ** 2 for i in range(len(d))]))
            prob = cvx.Problem(obj)
            prob.solve()
            if X.value is None:
                precision_return.append(np.linalg.inv(prior))
                self.cov = prior
            else:
                precision_return.append(np.linalg.inv(X.value+prior))
                self.cov = X.value + prior
        else:
            precision_return.append(np.linalg.inv(prior))
            self.cov = prior
        return np.array(precision_return)


    def updateParameters(self, articlePicked, click, userID): # click: reward
        self.t += 1
        prior = (self.EPSILON) * np.eye(self.param_dim)
        article_id = articlePicked.id
        article_feature = articlePicked.contextFeatureVector[: self.dimension] # context_feature
        concat_feature = np.array(articlePicked.featureVector)
        #     #! need to double check the implementation for concatenate user and item features
        tensor = torch.tensor(concat_feature, dtype=torch.float32).to(self.users[userID].device)
        z_context = self.users[userID].model(tensor,path="last") # old
        # put pickedArticle data into training set to update model
        self.users[userID].data.push(article_feature, click)  # self.data_h.add(context, action, reward)
        if self.t % self.update_freq_nn == 0:
            self.users[userID].updateParameters() # update_model(Train NN for new features)
            tensor = torch.tensor(concat_feature, dtype=torch.float32).to(self.users[userID].device)
            new_z = self.users[userID].model(tensor, path="last") # new
            new_z_ = new_z.clone().detach().numpy()
            if self.sigma_prior_flag == 1:
                self.precision_prior = self.calc_precision_prior(new_z_)
            if self.mu_prior_flag == 1:
                self.mu_prior = self.users[userID].model.last_layer.weight.clone().detach().numpy().T
            self.precision = (np.dot(new_z_.T, new_z_) + self.precision_prior)
            self.f = np.dot(new_z_, click)

        else:
            z_context_ = z_context.clone().detach().numpy()
            self.precision += np.dot(z_context_.T, z_context_)
            self.f += np.dot(z_context_.T, click)
        self.yy += click**2

        if len(self.cov.shape) > 2 :
            self.cov = self.cov.squeeze(axis=0)
        if len(self.precision.shape) > 2:
            self.precision = self.precision.squeeze(axis=0)
        if len(self.precision_prior.shape) > 2:
            self.precision_prior = self.precision_prior.squeeze(axis=0)
        self.cov = np.diag(self.precision) * np.eye(self.precision.shape[0])
        temp = self.precision_prior @ self.mu_prior
        if len(temp.shape) > 1:
            temp = temp.squeeze()
        temp0 = temp + self.f
        self.mu = np.dot(self.cov,temp0.T)
        # Inverse Gamma posterior update
        self.a += 0.5
        b_ = np.array(0.5 * self.yy).reshape(1,1)
        b_ += 0.5 * np.dot(self.mu_prior.T, np.dot(self.precision_prior, self.mu_prior))
        b_ -= 0.5 * (np.dot(self.mu.T, np.dot(self.precision, self.mu)))
        self.b += b_



