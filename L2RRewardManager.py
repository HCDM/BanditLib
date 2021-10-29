from conf import *
import random
import matplotlib.pyplot as plt
from Users.Users import UserManager
import datetime
from DatasetCollection import DataSet


class randomStruct:
    def __init__(self):
        self.reward = 0
        self.regret = 0


class Article:
    def __init__(self, aid, FV=None):
        self.id = aid
        self.featureVector = FV
        self.contextFeatureVector = FV


class L2RRewardManager:
    def __init__(self, arg_dict):
        for key in arg_dict:
            setattr(self, key, arg_dict[key])

    def runAlgorithms(self, algorithms, diffLists):
        timeRun = datetime.datetime.now().strftime("_%m_%d_%H_%M")
        filenameWriteRegret = os.path.join(self.save_address, "AccRegret" + timeRun + ".csv")
        self.set_up_regret_file(filenameWriteRegret, algorithms)

        tsave = 60 * 60 * 47  # Time interval for saving model.
        tstart = datetime.datetime.now()
        save_flag = 0

        tim_ = []
        AlgReward = {}
        AlgPicked = {}  # Records what article each algorithm picks
        AlgRegret = {}
        AlgRewardRatio_vsRandom = {}
        batchCumulateRegret = {}
        RandomChoice = randomStruct()
        RandomChoiceRegret = []

        for alg_name, alg in algorithms.items():
            AlgReward[alg_name] = []
            AlgPicked[alg_name] = []
            AlgRegret[alg_name] = []
            batchCumulateRegret[alg_name] = []
            AlgRewardRatio_vsRandom[alg_name] = []

        print("Preparing the dataset...")
        data = DataSet(self.address, self.context_dimension)
        data.read_data()
        n_queries = data.n_queries
        # random shuffle the query list
        query_sequence = random.sample(range(n_queries), n_queries)

        # only one user
        UserID = 0
        cumulativeOptimalReward = 0
        print("Start simulation")
        for i, qid in enumerate(query_sequence):
            print(i, qid)
            articlePool = []
            s_i = data.DoclistRanges[qid]
            e_i = data.DoclistRanges[qid + 1]
            label_vector = data.LabelVector[s_i:e_i]
            feature = data.FeatureMatrix[s_i:e_i]
            for DocId in range(e_i - s_i):
                articlePool.append(Article(DocId, feature[DocId]))
            RandomArticlePickled = random.choice(articlePool)
            RandomChoice.reward += label_vector[RandomArticlePickled.id]

            optimalReward = max(label_vector)
            cumulativeOptimalReward += optimalReward
            for alg_name, alg in algorithms.items():
                pickedArticle = alg.createRecommendation(articlePool, UserID, self.k).articles[0]
                reward = label_vector[pickedArticle.id]
                alg.updateParameters(pickedArticle, reward, UserID)

                AlgReward[alg_name].append(reward)
                AlgPicked[alg_name].append(pickedArticle.id)
                AlgRegret[alg_name].append(optimalReward - reward)

                if i % 100 == 0:
                    batchCumulateRegret[alg_name].append(sum(AlgRegret[alg_name]))
                    if RandomChoice.reward != 0:
                        AlgRewardRatio_vsRandom[alg_name].append(
                            (cumulativeOptimalReward - batchCumulateRegret[alg_name][-1])
                            / (1.0 * RandomChoice.reward)
                        )
                    else:
                        AlgRewardRatio_vsRandom[alg_name].append(0)

    def set_up_regret_file(self, filenameWriteRegret, algorithms):
        with open(filenameWriteRegret, "w") as f:
            f.write("Time(Iteration),Random")
            f.write("," + ",".join([str(alg_name) for alg_name in algorithms.keys()]))
            f.write("\n")
