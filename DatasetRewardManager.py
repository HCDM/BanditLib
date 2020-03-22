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
                print(self.address)
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
            self.set_up_file_write(self, algorithms)

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
                count = 0
                for i, line in enumerate(f, 1):
                        count += 1
            
            with open(fileName, 'r') as f:
                f.readline()
                for i, line in enumerate(f, 1):
                    totalObservations +=1            
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


                    #shuffle(articlePool[:self.poolArticleSize])
                    shuffle(articlePool)
                    if totalObservations % 100 == 0:
                        tim_.append(i) 
                        RandomChoiceRegret.append(RandomChoice.regret)
                        if totalObservations % 1000 == 0:
                            self.batchRecord(algorithms, i, tstart, RandomChoice, AlgPicked)
                        
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

                        if totalObservations%100==0:#self.batchSize==0:
                            BatchCumlateRegret[alg_name].append(sum(AlgRegret[alg_name]))
                            AlgRewardRatio_vsRandom[alg_name].append((i - BatchCumlateRegret[alg_name][-1]) / (1.0 * RandomChoice.reward))
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
              #  recordedStats = [articles_random.reward]
              #  for alg_name, alg in algorithms.items():
              #      recordedStats.append(AlgPicked[alg_name][-1])
              #      recordedStats.append(alg.reward)
                # write to file
              #  save_to_file(fileNameWrite, recordedStats, tim)
        
        def set_up_file_write(self, fileNameWrite, algorithms):
                fileSig = 'l2_'
                curTime = datetime.datetime.now().strftime('%m_%d_%Y %H:%M:%S')
                fileNameWrite = os.path.join(save_address, fileSig + curTime + '.csv')
                print(save_address, fileNameWrite)
                with open(fileNameWrite, 'a+') as f:
                        f.write('New Run at  ' + curTime)
                        f.write('\n, Time, RandomReward; ')
                        for alg_name, alg in algorithms.items():
                                f.write(alg_name+'Reward; ')
                        f.write('\n')

        def set_file_data(self):
                if self.dataset == 'LastFM':
                        self.relationFileName = LastFM_relationFileName
                        self.address = LastFM_address
                        self.save_address = LastFM_save_address
                        self.FeatureVectorsFileName = LastFM_FeatureVectorsFileName
                        self.itemNum = 19000
                else:
                        self.relationFileName = Delicious_relationFileName
                        self.address = Delicious_address
                        self.save_address = Delicious_save_address
                        self.FeatureVectorsFileName = Delicious_FeatureVectorsFileName  
                        self.itemNum = 190000  
