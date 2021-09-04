import torch
from backpack import backpack
from backpack.extensions import BatchGrad
from BaseAlg import NeuralBase
import numpy as np


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
        sum_r = torch.sum(r)
        with backpack(BatchGrad()):
            sum_r.backward()
        gradient_list = torch.cat([p.grad_batch.flatten(start_dim=1).detach() for p in self.model.parameters()], dim=1)
        var_matrix = torch.matmul(torch.matmul(gradient_list, self.Ainv), torch.transpose(gradient_list, 0, 1))
        score = torch.normal(r.view(-1), self.alpha*torch.sqrt(torch.diag(var_matrix)).view(-1))
        self.current_g = gradient_list


        pool_positions = torch.argsort(score)[(k * -1):]
        articles = []
        for i in range(k):
            articles.append(pool_articles[pool_positions[i]])
        return articles

    def updateParameters(self, articlePicked, click, userID):
        g = self.current_g[articlePicked]
        self.Ainv -= (torch.matmul(torch.matmul(torch.matmul(self.Ainv, g.view(-1, 1)), g.view(1, -1)), self.Ainv)) / (
                1 + torch.matmul(torch.matmul(g.view(1, -1), self.Ainv), g.view(-1, 1)))
        self.train_mlp(articlePicked, click, userID)


