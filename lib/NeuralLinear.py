from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from backpack import backpack
from backpack.extensions import BatchGrad
from lib.BaseAlg import BaseAlg, get_device, Network,Network_NL, obs_data_all
import numpy as np
from backpack import backpack, extend

import numpy as np
from scipy.stats import invgamma

class NeuralLinearUserStruct:
    def __init__(self, feature, featureDimension, mlp_dims,
                 epoch, batch_size, learning_rate,):
        # create neural model
        self.feature = feature
        self.mlp_dims = [int(x) for x in mlp_dims.split(",") if x != ""]
        self.device = get_device()
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = extend(Network_NL(feature_dim=featureDimension, mlp_dims=self.mlp_dims).to(self.device))
        self.loss_func = nn.MSELoss()
        self.data = obs_data_all() # training set
        self.latent_data = obs_data_all()
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


class NeuralLinearAlgorithm(BaseAlg):
    def __init__(self, arg_dict,):  # n is number of users
        BaseAlg.__init__(self, arg_dict)
        self.device = get_device()
        # latent_dim
        self.mlp_dims = [int(x) for x in arg_dict["mlp"].split(",") if x != ""]
        self.latent_dim = self.mlp_dims[-1]
        self.param_dim = self.latent_dim
        # create parameters
        # Gaussian prior for each beta_i
        self._lambda_prior = float(arg_dict["lambda_"])
        self.mu = np.zeros(self.param_dim)
        self.f = np.zeros(self.param_dim)
        self.yy = [0]
        self.cov = (1.0 / self._lambda_prior) * np.eye(self.param_dim)
        self.precision = self._lambda_prior * np.eye(self.param_dim)
        # Inverse Gamma prior for each sigma2_i
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
                                    arg_dict["batch_size"], arg_dict["learning_rate"],))

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
        vals = np.dot(beta_s,z_context_.T)
        pool_positions = np.argsort(vals)[(k * -1):]
        articles = []
        for i in range(k):
            articles.append(pool_articles[pool_positions[i]])
        return articles

    def updateParameters(self, articlePicked, click, userID): # click: reward
        self.t += 1
        article_id = articlePicked.id
        article_feature = articlePicked.contextFeatureVector[: self.dimension] # context_feature
        concat_feature = np.array(articlePicked.featureVector)
        #     #! need to double check the implementation for concatenate user and item features
        tensor = torch.tensor(concat_feature, dtype=torch.float32).to(self.users[userID].device)
        z_context = self.users[userID].model(tensor,path="last")
        # put pickedArticle data into training set to update model
        self.users[userID].data.push(article_feature, click)  # self.data_h.add(context, action, reward)
        if self.t % self.update_freq_nn == 0:
            self.users[userID].updateParameters() # update_model(Train NN for new features)
            tensor = torch.tensor(concat_feature, dtype=torch.float32).to(self.users[userID].device)
            new_z = self.users[userID].model(tensor, path="last")
            new_z_ = new_z.clone().detach().numpy()
            self.precision = (np.dot(new_z_.T, new_z_) + self._lambda_prior * np.eye(self.param_dim))
            self.f = np.dot(new_z_.T, click)
        else:
            z_context_ = z_context.clone().detach().numpy()
            self.precision += np.dot(z_context_.T, z_context_)
            self.f += np.dot(z_context_.T, click)
        self.yy += click**2
        self.cov = np.linalg.inv(self.precision)
        self.mu = np.dot(self.cov, self.f)
        # Inverse Gamma posterior update
        self.a += 0.5
        b_ = 0.5 * (self.yy - np.dot(self.mu.T, np.dot(self.precision, self.mu)))
        self.b += b_



