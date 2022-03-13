import time
import os
import pickle 
import datetime
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from conf import *
from Tool.utilFunc import *

from BanditAlg.BanditAlgorithms import UCB1Algorithm, eGreedyAlgorithm 
from BanditAlg.BanditAlgorithms_MF import MFAlgorithm
from BanditAlg.BanditAlgorithms_LinUCB import N_LinUCBAlgorithm
from IC.IC import runIC, runICmodel, runICmodel_n
from IC.runIAC  import weightedEp, runIAC, runIACmodel, randomEp, uniformEp

class simulateOnlineData:
    def __init__(self, G, P, oracle, seed_size, iterations, dataset):
        self.G = G
        self.TrueP = P
        self.seed_size = seed_size
        self.oracle = oracle
        self.iterations = iterations
        self.dataset = dataset
        self.startTime = datetime.datetime.now()
        self.BatchCumlateReward = {}
        self.AlgReward = {}
        self.result_oracle = []

    def runAlgorithms(self, algorithms):
        self.tim_ = []
        for alg_name, alg in list(algorithms.items()):
            self.AlgReward[alg_name] = []
            self.BatchCumlateReward[alg_name] = []

        self.resultRecord()
        optS = self.oracle(self.G, self.seed_size, self.TrueP)

        for iter_ in range(self.iterations):
            optimal_reward, live_nodes, live_edges = runICmodel_n(G, optS, self.TrueP)
            self.result_oracle.append(optimal_reward)
            print('oracle', optimal_reward)
            
            for alg_name, alg in list(algorithms.items()): 
                S = alg.decide() 
                reward, live_nodes, live_edges = runICmodel_n(G, S, self.TrueP)

                alg.updateParameters(S, live_nodes, live_edges, iter_)

                self.AlgReward[alg_name].append(reward)

            self.resultRecord(iter_)

        self.showResult()

    def resultRecord(self, iter_=None):
        # if initialize
        if iter_ is None:
            timeRun = self.startTime.strftime('_%m_%d_%H_%M_%S') 
            fileSig = '_seedsize'+str(self.seed_size) + '_iter'+str(self.iterations)+'_'+str(self.oracle.__name__)+'_'+self.dataset
            self.filenameWriteReward = os.path.join(save_address, 'AccReward' + timeRun + fileSig + '.csv')

            with open(self.filenameWriteReward, 'w') as f:
                f.write('Time(Iteration)')
                f.write(',' + ','.join( [str(alg_name) for alg_name in algorithms.keys()]))
                f.write('\n') 
        else:
            # if run in the experiment, save the results
            print("Iteration %d" % iter_, " Elapsed time", datetime.datetime.now() - self.startTime)
            self.tim_.append(iter_)
            for alg_name in algorithms.keys():
                self.BatchCumlateReward[alg_name].append(sum(self.AlgReward[alg_name][-1:]))
            with open(self.filenameWriteReward, 'a+') as f:
                f.write(str(iter_))
                f.write(',' + ','.join([str(self.BatchCumlateReward[alg_name][-1]) for alg_name in algorithms.keys()]))
                f.write('\n')

    def showResult(self):
        print('average reward for oracle:', np.mean(self.result_oracle))
        
        f, axa = plt.subplots(1, sharex=True)
        for alg_name in algorithms.keys():  
            axa.plot(self.tim_, self.BatchCumlateReward[alg_name],label = alg_name)
            print('%s: %.2f' % (alg_name, np.mean(self.BatchCumlateReward[alg_name])))
        axa.legend(loc='upper left',prop={'size':9})
        axa.set_xlabel("Iteration")
        axa.set_ylabel("Reward")
        axa.set_title("Average Reward")
        plt.savefig('./SimulationResults/AvgReward' + str(self.startTime.strftime('_%m_%d_%H_%M'))+'.pdf')
        plt.show()
        # plot accumulated reward
        f, axa = plt.subplots(1, sharex=True)
        for alg_name in algorithms.keys():  
            result = [sum(self.BatchCumlateReward[alg_name][:i]) for i in range(len(self.tim_))]
            axa.plot(self.tim_, result, label = alg_name)
        axa.legend(loc='upper left',prop={'size':9})
        axa.set_xlabel("Iteration")
        axa.set_ylabel("Reward")
        axa.set_title("Accumulated Reward")
        plt.savefig('./SimulationResults/AcuReward' + str(self.startTime.strftime('_%m_%d_%H_%M'))+'.pdf')
        plt.show()

        for alg_name in algorithms.keys():  
            try:
                loss = algorithms[alg_name].getLoss()
            except:
                continue
            '''
            f, ax1 = plt.subplots()
            color = 'tab:red'
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel('Loss of Probability', color=color)
            ax1.plot(self.tim_, loss[:, 0], color=color, label='Probability')
            ax1.tick_params(axis='y', labelcolor=color)

            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            ax2.set_ylabel('Loss of Theta and Beta', color='tab:blue')  # we already handled the x-label with ax1
            ax2.plot(self.tim_, loss[:, 1], color='tab:blue', linestyle=':', label = r'$\theta$')
            ax2.plot(self.tim_, loss[:, 2], color='tab:blue', linestyle='--', label = r'$\beta$')
            ax2.tick_params(axis='y', labelcolor='tab:blue')
            ax2.legend(loc='upper left',prop={'size':9})
            f.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.savefig('./SimulationResults/Loss' + str(self.startTime.strftime('_%m_%d_%H_%M'))+'.pdf')
            plt.show()
            '''
            np.save('./SimulationResults/Loss-{}'.format(alg_name) + str(self.startTime.strftime('_%m_%d_%H_%M'))+'.npy', loss)
        
if __name__ == '__main__':
    start = time.time()

    G = pickle.load(open(graph_address, 'rb'), encoding='latin1')
    prob = pickle.load(open(prob_address, 'rb'), encoding='latin1')
    parameter = pickle.load(open(param_address, 'rb'), encoding='latin1')
    feature_dic = pickle.load(open(edge_feature_address, 'rb'), encoding='latin1')

    P = nx.DiGraph()
    for (u,v) in G.edges():
        P.add_edge(u, v, weight=prob[(u,v)])
    print('nodes:', len(G.nodes()))
    print('edges:', len(G.edges()))
    print('Done with Loading Feature')
    print('Graph build time:', time.time() - start)
    
    simExperiment = simulateOnlineData(G, P, oracle, seed_size, iterations, dataset)

    algorithms = {}
    algorithms['UCB1'] = UCB1Algorithm(G, P, parameter, seed_size, oracle)
    algorithms['egreedy_0.1'] = eGreedyAlgorithm(G, seed_size, oracle, 0.1)
    algorithms['LinUCB'] = N_LinUCBAlgorithm(G, P, parameter, seed_size, oracle, dimension*dimension, alpha_1, lambda_, feature_dic, 1)
    algorithms['OurAlgorithm'] = MFAlgorithm(G, P, parameter, seed_size, oracle, dimension)

    simExperiment.runAlgorithms(algorithms)