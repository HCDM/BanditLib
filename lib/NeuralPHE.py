import torch
from BaseAlg import NeuralBase
import numpy as np
import torch.optim as optim


class NeuralUCBAlgorithm(NeuralBase):
    def __init__(self, arg_dict, init="zero"):  # n is number of users
        NeuralBase.__init__(self, arg_dict)
        self.Ainv = (1 / self.lambda_) * torch.eye((self.model.total_param)).to(self.device)
        self.current_g = None

    def decide(self, pool_articles, userID, k=1):
        # concatenate user feature and article features
        user_feat = self.user_feature[userID]
        concat_feature = np.concatenate(np.repeat(user_feat, len(pool_articles)), pool_articles)

        tensor = torch.tensor(concat_feature, dtype=torch.float32).to(self.device)
        r = self.model(tensor)

        pool_positions = torch.argsort(r)[(k * -1):]
        articles = []
        for i in range(k):
            articles.append(pool_articles[pool_positions[i]])
        return articles

    def updateParameters(self, articlePicked, click, userID):
        self.train_mlp(articlePicked, click, userID)

    def train_mlp(self, articlePicked, click, userID):
        user_feat = self.user_feature[userID]
        article_feat = articlePicked.contextFeatureVector[:self.dimension]
        concate_feat = np.concatenate((user_feat.reshape(-1), article_feat.reshape(-1)), 0)
        self.data.push(concate_feat, click)

        # update model
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self._lambda)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.lr_decay)
        loss_list = []
        dataloader = torch.utils.data.DataLoader(self.data, batch_size=self.batch_size, shuffle=True, num_workers=0)

        for i in range(self.epoch):
            total_loss = 0
            for j, batch in enumerate(dataloader):
                self.model.zero_grad()
                optimizer.zero_grad()
                context_feature = torch.tensor(batch["context"], dtype=torch.float32).to(self.device)
                # add perturbed noise to the observed reward
                click = torch.tensor(batch["click"], dtype=torch.float32).to(self.device).view(-1) \
                        + torch.normal(0, self.alpha, size=(1, len(batch["click"])), device=self.device)
                pred = self.model(context_feature).view(-1)
                loss = self.loss_func(pred, click)
                total_loss += loss
                loss.backward()
                optimizer.step()
            loss_list.append(total_loss.item() / j)
            scheduler.step()


