from Recommendation import Recommendation
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from backpack import extend


def get_device():
    if torch.cuda.is_available():
        device = "cuda:{}".format(torch.cuda.current_device())
    else:
        device = "cpu"
    print("use device: ", device)
    return device


class BaseAlg:
    def __init__(self, arg_dict):  # n is number of users
        self.dimension = 0
        for key in arg_dict:
            setattr(self, key, arg_dict[key])

        self.estimates = {}
        self.estimates["CanEstimateUserPreference"] = arg_dict["parameters"]["Theta"]
        self.estimates["CanEstimateCoUserPreference"] = arg_dict["parameters"]["CoTheta"]
        self.estimates["CanEstimateW"] = arg_dict["parameters"]["W"]
        self.estimates["CanEstimateV"] = arg_dict["parameters"]["V"]

    def getEstimateSettings(self):
        return self.estimates

    def decide(self, pool_articles, userID, k=1):
        pass

    def createRecommendation(self, pool_articles, userID, k):
        articles = self.decide(pool_articles, userID, k)
        recommendation = Recommendation(k, articles)
        return recommendation

    def updateParameters(self, articlePicked, click, userID):
        pass



    def updateRecommendationParameters(self, recommendation, reward, userID):
        for i in range(recommendation.k):
            self.updateParameters(recommendation.articles[i], reward[i], userID)


    def getV(self, articleID):
        if self.dimension == 0:
            return np.zeros(self.context_dimension + self.hidden_dimension)
        else:
            return np.zeros(self.dimension)

    def getW(self, userID):
        return np.identity(n=self.n_users)


class obs_data_all(torch.utils.data.Dataset):
    def __init__(self, buffer_s=-1):
        self.context_history = []
        self.click_history = []
        self.buffer_s = buffer_s

    def push(self, context, click):
        if self.buffer_s > 0:
            self.context_history.append(context)
            self.click_history.append(click)
            # fifo
            if len(self.context_history) > self.buffer_s:
                self.context_history.pop(0)
            if len(self.click_history) > self.buffer_s:
                self.click_history.pop(0)
        else:
            self.context_history.append(context)
            self.click_history.append(click)

    def __len__(self):
        return len(self.click_history)

    def __getitem__(self, idx):
        return {"context": self.context_history[idx], "click": self.click_history[idx]}


class Network(nn.Module):
    def __init__(self, feature_dim, mlp_dims):
        super(Network, self).__init__()
        layers = [torch.nn.Linear(feature_dim, mlp_dims[0]), torch.nn.ReLU()]
        for idx in range(len(mlp_dims) - 1):
            layers.append(torch.nn.Linear(mlp_dims[idx], mlp_dims[idx + 1]))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(mlp_dims[-1], 1))

        self.model = torch.nn.Sequential(*layers)
        self.total_param = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, data):
        out = self.model(data)
        return out


class Network_NL(nn.Module):
    def __init__(self,feature_dim, mlp_dims):
        super(Network_NL,self).__init__()
        layers = [torch.nn.Linear(feature_dim, mlp_dims[0]), torch.nn.ReLU()]
        for idx in range(len(mlp_dims) - 1):
            layers.append(torch.nn.Linear(mlp_dims[idx], mlp_dims[idx + 1]))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(mlp_dims[-1], 1))
        self.model = torch.nn.Sequential(*layers)
        self.model_last = torch.nn.Sequential(*layers[:-1])
        self.last_layer = layers[-1]
        self.total_param = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, data,path="all"):
        if path == "all":
            out = self.model(data)
        elif path == "last":
            out = self.model_last(data)
        return out

