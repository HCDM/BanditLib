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
    print("Use device: ", device)
    return device


class BaseAlg():
	def __init__(self, arg_dict):  # n is number of users
		self.dimension = 0
		for key in arg_dict:
			setattr(self, key, arg_dict[key])

		self.estimates = {}
		self.estimates['CanEstimateUserPreference'] = arg_dict['parameters']['Theta']
		self.estimates['CanEstimateCoUserPreference'] = arg_dict['parameters']['CoTheta']
		self.estimates['CanEstimateW'] = arg_dict['parameters']['W']
		self.estimates['CanEstimateV'] = arg_dict['parameters']['V']

	def getEstimateSettings(self):
		return self.estimates

	def decide(self, pool_articles, userID, k = 1):
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
		return np.identity(n = self.n_users)

class NeuralBase(BaseAlg):
	def __init__(self, arg_dict, init="zero"):  # n is number of users
		BaseAlg.__init__(self, arg_dict)
		self.mlp_dims = arg_dict['parameters']['mlp_dims']
		self.lr = arg_dict['parameters']['lr']
		self.lr_decay = arg_dict['paramters']['lr_decay']
		self.user_feature = np.genfromtxt(arg_dict['parameters']['user_dir'], delimiter=' ')
		self.device = get_device()

		self.model = extend(Network(feature_dim=self.n_feature, mlp_dims=self.mlp_dims).to(self.device))
        self.loss_func = nn.MSELoss()

		self.epoch = 20
        self.batch_size = 1024
        self.data = obs_data_all()
        self.len = 0
        
        self.train_round = 0

	def train_mlp(self, articlePicked, click, userID):
		# update data
		user_feat = self.user_feature[userID]
		article_feat = articlePicked.contextFeatureVector[:self.dimension]
		concate_feat = np.concatenate((user_feat.reshape(-1), article_feat.reshape(-1)), 0)
		self.data.push(concate_feat, click)

		# update model
		optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self._lambda)
        scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=self.lr_decay)
		loss_list = []
		dataloader = torch.utils.data.DataLoader(self.data, batch_size=self.batch_size, shuffle=True, num_workers=0)
  
		for i in range(self.epoch):
			total_loss = 0
			for j, batch in enumerate(dataloader)
				self.model.zero_grad()
                self.optimizer.zero_grad()
                context_feature = torch.tensor(batch["context"], dtype=torch.float32).to(self.device)
                click = torch.tensor(batch["click"], dtype=torch.float32).to(self.device).view(-1)
                pred = self.model(context_feature).view(-1)
                loss = self.loss_func(pred, click)
                total_loss += loss
                loss.backward()
                self.optimizer.step()
            loss_list.append(total_loss.item() / j)
            self.scheduler.step()
        
				
				


class obs_data_all(torch.utils.data.Dataset):
    def __init__(self):
        self.context_history = []
        self.click_history = []

    def push(self, context, click):
        self.context_history.append(context)
        self.click_history.append(click)

    def __len__(self):
        return len(self.click_history)

    def __getitem__(self, idx):
        return {
            "context": self.context_history[idx],
            "click": self.click_history[idx],
        }

class Network(nn.Module):
    def __init__(self, feature_dim, mlp_dims):
        super(Network, self).__init__()
        layers = []
        layers.append(torch.nn.Linear(feature_dim, mlp_dims[0]))
        layers.append(torch.nn.LeakyReLU())
        for idx in range(len(mlp_dims) - 1):
            layers.append(torch.nn.Linear(mlp_dims[idx], mlp_dims[idx + 1]))
            layers.append(torch.nn.LeakyReLU())
        layers.append(torch.nn.Linear(mlp_dims[-1], 1))

        self.model = torch.nn.Sequential(*layers)
        self.total_param = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input):
        out = self.model(input)
        return out

