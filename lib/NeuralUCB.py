import torch
from backpack import backpack
from backpack.extensions import BatchGrad
from BaseAlg import NeuralBase

class NeuralUCBAlgorithm(NeuralBase):
    def __init__(self, arg_dict, init="zero"):  # n is number of users
		NeuralBase.__init__(self, arg_dict)
    
    def decide(self, pool_articles, userID, k = 1):
      # concatenate user feature and article features
      user_feat = self.user_feature[userID]
      concat_feature = np.concatenate(np.repeat(user_feat, len(pool_articles)), pool_articles)
      
      tensor = torch.tensor(concat_feature, dtype=torch.float32).to(self.device)
      r = self.model(tensor)
      sum_r = torch.sum(r)
      with backpack(BatchGrad()):
        sum_r.backward()
      gradient_list = torch.cat([p.grad_batch.flatten(start_dim=1).detach() for p in self.model.parameters()], dim=1)
      
		
