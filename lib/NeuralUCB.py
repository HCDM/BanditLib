import torch
import torch.nn as nn
import torch.optim as optim
from backpack import backpack
from backpack.extensions import BatchGrad
from lib.BaseAlg import BaseAlg, get_device, Network, obs_data_all
import numpy as np
from backpack import backpack, extend


class NeuralUCBUserStruct:
    def __init__(self, feature, featureDimension, lambda_, mlp_dims, epoch, batch_size, learning_rate, init="zero", ):

        # create neural model
        self.feature = feature
        self.lambda_ = lambda_
        self.mlp_dims = [int(x) for x in mlp_dims.split(",") if x != ""]
        self.device = get_device()
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = extend(Network(feature_dim=featureDimension, mlp_dims=self.mlp_dims).to(self.device))
        self.loss_func = nn.MSELoss()
        # create parameters
        # diagonal matrix
        self.AInv = (1 / self.lambda_) * torch.eye((self.model.total_param)).to(self.device)
        self.data = obs_data_all()
        self.time = 0

    def updateParameters(self, gradient, click):
        self.AInv -= torch.matmul(torch.matmul(torch.matmul(self.AInv, gradient.view(-1, 1)), gradient.view(1, -1)),
                                  self.AInv, ) / (1 + torch.matmul(torch.matmul(gradient.view(1, -1), self.AInv),
                                                                   gradient.view(-1, 1), ))
       
        self.update_model()

    def update_model(self):
        num_data = len(self.data)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                     weight_decay=self.lambda_ / num_data)
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
                # context_feature = torch.tensor(batch["context"], dtype=torch.float32).to(
                #     self.device
                # )
                context_feature = batch["context"].clone().detach().float().requires_grad_(True).to(self.device)
                # click = torch.tensor(batch["click"], dtype=torch.float32).to(self.device).view(-1)
                click = batch["click"].clone().detach().float().requires_grad_(True).to(self.device).view(-1)
                pred = self.model(context_feature).view(-1)
                loss = self.loss_func(pred, click)
                total_loss += loss
                loss.backward()
                optimizer.step()
            loss_list.append(total_loss.item() / num_batch)

            scheduler.step()


class NeuralUCBAlgorithm(BaseAlg):
    def __init__(self, arg_dict, init="zero"):  # n is number of users
        BaseAlg.__init__(self, arg_dict)

        self.current_g = None
        self.users = []
        for i in range(arg_dict["n_users"]):
            self.users.append(
                NeuralUCBUserStruct([], arg_dict["dimension"], arg_dict["lambda_"], arg_dict["mlp"], arg_dict["epoch"],
                                    arg_dict["batch_size"], arg_dict["learning_rate"], init, ))

    def decide(self, pool_articles, userID, k=1) -> object:
        # MEAN
        # create concatenated feature for the user and arm
        user_feat = np.array(self.users[userID].feature)
        # if user_feat == []:
        concat_feature = np.array([x.featureVector for x in pool_articles])
        # else:
        #     #! need to double check the implementation for concatenate user and item features
        #     concat_feature = pool_articles
        tensor = torch.tensor(concat_feature, dtype=torch.float32).to(self.users[userID].device)
        mean_matrix = self.users[userID].model(tensor)

        sum_mean = torch.sum(mean_matrix)
        with backpack(BatchGrad()):
            sum_mean.backward()
        gradient_list = torch.cat(
            [p.grad_batch.flatten(start_dim=1).detach() for p in self.users[userID].model.parameters()], dim=1, )
        # calculate CB
        var_matrix = torch.matmul(torch.matmul(gradient_list, self.users[userID].AInv),
                                  torch.transpose(gradient_list, 0, 1), )
        pta_matrix = mean_matrix.view(-1) + self.alpha * torch.sqrt(torch.diag(var_matrix))
        self.current_g = gradient_list

        pool_positions = torch.argsort(pta_matrix)[(k * -1):]
        articles = []
        for i in range(k):
            articles.append(pool_articles[pool_positions[i]])
        return articles

    def updateParameters(self, articlePicked, click, userID):
        article_id = articlePicked.id
        article_feature = articlePicked.contextFeatureVector[: self.dimension]
        self.users[userID].data.push(article_feature, click)
        self.users[userID].updateParameters(self.current_g[article_id], click)
