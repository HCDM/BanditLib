from conf import *
from LastFM_util_functions import *
from random import shuffle 
from Users.Users import UserManager
import matplotlib.pyplot as plt

import datetime


class randomStruct():
        def __init__(self):       
                self.reward = 0
                self.regret = 0

class Article():
        def __init__(self, aid, FV=None):
                self.id = aid
                self.featureVector = FV
                self.contextFeatureVector = FV

class DatasetRewardManager():
        def __init__(self, arg_dict, dataset, clusterfile):
                for key in arg_dict:
                        setattr(self, key, arg_dict[key])
                self.dataset = dataset
                self.set_file_data()
                self.OriginaluserNum = 2100
                self.nClusters = 100
                self.userNum = self.nClusters
                self.Gepsilon = .3

#                if clusterfile:           
#                        label = read_cluster_label(args.clusterfile)
#                        userNum = nClusters = int(args.clusterfile.name.split('.')[-1]) # Get cluster number.
#                        W = initializeW_label(userNum, relationFileName, label, args.diagnol, args.showheatmap)   # Generate user relation matrix
#                        GW = initializeGW_label(Gepsilon,userNum, relationFileName, label, args.diagnol)            
#                else:
#                        normalizedNewW, newW, label = initializeW_clustering(self.OriginaluserNum, self.relationFileName, self.nClusters)
#                        GW = initializeGW_clustering(self.Gepsilon, self.relationFileName, newW)
#                        W = normalizedNewW
        
        def runAlgorithms(self, algorithms, diffLists):
            timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M')
            filenameWriteRegret = os.path.join(self.save_address, 'AccRegret' + timeRun + '.csv')
            self.set_up_regret_file(filenameWriteRegret, algorithms)

            tsave = 60*60*47 # Time interval for saving model.
            tstart = datetime.datetime.now()
            save_flag = 0

            tim_ = []
            AlgReward = {}
            AlgPicked = {} # Records what article each algorithm picks
            AlgRegret = {}      
            AlgRewardRatio_vsRandom = {} 
            BatchCumlateRegret = {}
            RandomChoice = randomStruct() 
            RandomChoiceRegret = []
             
            for alg_name, alg in algorithms.items():
                AlgReward[alg_name] = []
                AlgPicked[alg_name] = []
                AlgRegret[alg_name] = []
                BatchCumlateRegret[alg_name] = []
                AlgRewardRatio_vsRandom[alg_name] = []

            totalObservations = 0
            OptimalReward = 1
            fileName =  self.address + "/processed_events_shuffled.dat"            
            print(fileName)
            FeatureVectors = readFeatureVectorFile(self.FeatureVectorsFileName)
            
            with open(fileName, 'r') as f:
                f.readline()
                for i, line in enumerate(f, 1):
                    #if i > 10000: break 
                    articlePool = []
                    userID, tim, pool_articles = parseLine(line)
                    article_chosen = int(pool_articles[0]) 
                    for article in pool_articles:
                        article_id = int(article.strip(']'))
                        articlePool.append(Article(article_id, FeatureVectors[article_id]))
                    
                    RandomArticlePicked = choice(articlePool)
                    if RandomArticlePicked.id == article_chosen:
                        RandomChoice.reward += 1 
                    else:
                        RandomChoice.regret += 1


                    shuffle(articlePool[:self.poolArticleSize])
                        
                    for alg_name, alg in algorithms.items():
                        if alg_name in ['CoLin', 'CoLinRankOne','factorLinUCB', 'LearnWl2', 'LearnWl1', 'LearnWl1_UpdateA','LearnWl2_UpdateA', 'LearnW_WRegu']:
                            currentUserID = label[userID]
                        else:
                            currentUserID = userID
                        pickedArticle = alg.createRecommendation(articlePool, currentUserID, self.k).articles[0]
                        
                        if (pickedArticle.id == article_chosen):
                            reward = 1
                        else:
                            reward = 0
                        alg.updateParameters(pickedArticle, reward, currentUserID)

                        AlgReward[alg_name].append(reward)
                        AlgPicked[alg_name].append(pickedArticle.id)
                        AlgRegret[alg_name].append(OptimalReward - reward) 

                        if save_flag:
                            model_name = 'saved_models/'+self.dataset+'_'+str(self.nClusters)+'_shuffled_Clustering_'\
                                                +alg_name+'_Diagnol_'+args.diagnol+'_' + timeRun
                            model_dump(alg, model_name, i)

                        if i % 100==0:#self.batchSize==0:
                            BatchCumlateRegret[alg_name].append(sum(AlgRegret[alg_name]))
                            if RandomChoice.reward != 0:
                                AlgRewardRatio_vsRandom[alg_name].append((i - BatchCumlateRegret[alg_name][-1]) / (1.0 * RandomChoice.reward))
                            else:
                                AlgRewardRatio_vsRandom[alg_name].append(0)
                    if i % 100 == 0:
                        tim_.append(i) 
                        RandomChoiceRegret.append(RandomChoice.regret)
                        if i % 1000 == 0:
                            self.batchRecord(algorithms, i, tstart, RandomChoice, AlgPicked)
                            self.write_regret_to_file(filenameWriteRegret, algorithms, BatchCumlateRegret, i, RandomChoice.regret)
                self.plot_result(algorithms, BatchCumlateRegret, tim_, None, RandomChoiceRegret, AlgRewardRatio_vsRandom)


        def plot_result(self, algorithms, BatchCumlateRegret, tim_, diffLists, RandomChoiceRegret, AlgRewardRatio_vsRandom):
                # plot the results      
                f, axa = plt.subplots(1, sharex=True)
                for alg_name in algorithms.iterkeys():
                        axa.plot(tim_, BatchCumlateRegret[alg_name],label = alg_name)
                        print '%s: %.2f' % (alg_name, BatchCumlateRegret[alg_name][-1])

                print("RandomChoiceRegret: " +str(RandomChoiceRegret[-1]))
                axa.plot(tim_, RandomChoiceRegret, label='Random Choice')

                axa.legend(loc='upper left',prop={'size':9})
                axa.set_xlabel("Iteration")
                axa.set_ylabel("Regret")
                axa.set_title("Accumulated Regret")
                plt.show()

                # plot the results      
                f, axa = plt.subplots(1, sharex=True)
                for alg_name in algorithms.iterkeys():
                        axa.plot(tim_, AlgRewardRatio_vsRandom[alg_name],label = alg_name)
                        print '%s: %.2f' % (alg_name, AlgRewardRatio_vsRandom[alg_name][-1])

                axa.legend(loc='upper left',prop={'size':9})
                axa.set_xlabel("Iteration")
                axa.set_ylabel("Normalized Payoff")
                axa.set_title("Reward Ratio Algorith vs Random")
                plt.show()

 
        def batchRecord(self, algorithms, iter_, tstart, articles_random, AlgPicked):
		print "Datapoint #%d"%iter_, " Elapsed time", datetime.datetime.now() - tstart 
        
        # Creates file to record reward of each algorithm after each batch completes 
        def set_up_regret_file(self, filenameWriteRegret, algorithms):
                with open(filenameWriteRegret, 'w') as f:
                        f.write('Time(Iteration),Random')
                        f.write(',' + ','.join( [str(alg_name) for alg_name in algorithms.iterkeys()]))
                        f.write('\n')
                print(filenameWriteRegret)
        
        def write_regret_to_file(self, filenameWriteRegret, algorithms, BatchCumlateRegret, iter_, randomRegret):
                 with open(filenameWriteRegret, 'a+') as f:
                        f.write(str(iter_))
                        f.write(',' + str(randomRegret))
                        f.write(',' + ','.join([str(BatchCumlateRegret[alg_name][-1]) for alg_name in algorithms.iterkeys()]))
                        f.write('\n')

        def set_file_data(self):
                if self.dataset == 'LastFM':
                        self.relationFileName = LastFM_relationFileName
                        self.address = LastFM_address
                        self.save_address = LastFM_save_address
                        self.FeatureVectorsFileName = LastFM_FeatureVectorsFileName
                        self.itemNum = 19000
                elif self.dataset == 'Delicous':
                        self.relationFileName = Delicious_relationFileName
                        self.address = Delicious_address
                        self.save_address = Delicious_save_address
                        self.FeatureVectorsFileName = Delicious_FeatureVectorsFileName  
                        self.itemNum = 190000  
