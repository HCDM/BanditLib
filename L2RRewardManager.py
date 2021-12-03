from conf import *
import random
import matplotlib.pyplot as plt
from Users.Users import UserManager
import datetime
from DatasetCollection import DataSet
from lib.NeuralLinear import NeuralLinearAlgorithm


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
        BatchCumulateRegret = {}
        RandomChoice = randomStruct()
        RandomChoiceRegret = []

        for alg_name, alg in algorithms.items(): # "NeuralLinear"
            AlgReward[alg_name] = []
            AlgPicked[alg_name] = []
            AlgRegret[alg_name] = []
            BatchCumulateRegret[alg_name] = []
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
            RandomChoice.regret = cumulativeOptimalReward - RandomChoice.reward
            for alg_name, alg in algorithms.items():
                pickedArticle = alg.createRecommendation(articlePool, UserID, self.k).articles[0]
                reward = label_vector[pickedArticle.id]
                alg.updateParameters(pickedArticle, reward, UserID)

                AlgReward[alg_name].append(reward)
                AlgPicked[alg_name].append(pickedArticle.id)
                AlgRegret[alg_name].append(optimalReward - reward)
                if i % 100 == 0:
                    BatchCumulateRegret[alg_name].append(sum(AlgRegret[alg_name]))
                    if RandomChoice.reward != 0:
                        AlgRewardRatio_vsRandom[alg_name].append(
                            (cumulativeOptimalReward - BatchCumulateRegret[alg_name][-1]) / (1.0 * RandomChoice.reward))
                    else:
                        AlgRewardRatio_vsRandom[alg_name].append(0)
            if i % 100 == 0:
                tim_.append(i)
                RandomChoiceRegret.append(RandomChoice.regret)
                if i % 1000 == 0 or i == 9999:
                    self.batchRecord(algorithms, i, tstart, RandomChoice, AlgPicked)
                    self.write_regret_to_file(filenameWriteRegret, algorithms, BatchCumulateRegret, i,
                                              RandomChoice.regret)

    def set_up_regret_file(self, filenameWriteRegret, algorithms):
        with open(filenameWriteRegret, "w") as f:
            f.write("Time(Iteration),Random")
            f.write("," + ",".join([str(alg_name) for alg_name in algorithms.keys()]))
            f.write("\n")

    def batchRecord(self, algorithms, iter_, tstart, articles_random, AlgPicked):
        print("Datapoint #{} Elapsed time".format(iter_, datetime.datetime.now() - tstart))

    def write_regret_to_file(self, filenameWriteRegret, algorithms, BatchCumulateRegret, iter_, randomRegret):
        with open(filenameWriteRegret, "a+") as f:
            f.write(str(iter_))
            f.write("," + str(randomRegret))
            f.write("," + ",".join([str(BatchCumulateRegret[alg_name][-1]) for alg_name in algorithms.keys()]))
            f.write("\n")
