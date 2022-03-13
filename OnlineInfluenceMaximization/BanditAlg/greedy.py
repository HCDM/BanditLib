from random import choice, random, sample
import numpy as np
import networkx as nx

class ArmBaseStruct(object):
    def __init__(self, armID):
        self.armID = armID
        self.totalReward = 0.0
        self.numPlayed = 0
        self.averageReward  = 0.0
        self.p_max = 1
       
    def updateParameters(self, reward):
        self.totalReward += reward
        self.numPlayed +=1
        self.averageReward = self.totalReward/float(self.numPlayed)

        
class eGreedyArmStruct(ArmBaseStruct):
    def getProb(self, epsilon):
        if random() < epsilon: # random exploration
            pta = random()
        else:
            if self.numPlayed == 0:
                pta = 0
            else:
                #print 'GreedyProb', self.totalReward/float(self.numPlayed)
                pta = self.totalReward/float(self.numPlayed)
                if pta > self.p_max:
                    pta = self.p_max
        return pta
        

class eGreedyAlgorithm:
    def __init__(self, G, seed_size, oracle, epsilon, feedback = 'edge'):
        self.G = G
        self.seed_size = seed_size
        self.oracle = oracle
        self.feedback = feedback
        self.arms = {}
        #Initialize P
        self.currentP =nx.DiGraph()
        for (u,v) in self.G.edges():
            self.arms[(u,v)] = eGreedyArmStruct((u,v))
            self.currentP.add_edge(u,v, weight=0)

        self.TotalPlayCounter = 0
        self.epsilon = epsilon

    def decide(self):
        S = self.oracle(self.G, self.seed_size, self.currentP)# self.oracle(self.G, self.seed_size, self.arms)
        return S

    def updateParameters(self, S, live_nodes, live_edges, iter_): 
        for u in live_nodes:
            for (u, v) in self.G.edges(u):
                if (u,v) in live_edges:
                    self.arms[(u, v)].updateParameters(reward=live_edges[(u,v)])
                else:
                    self.arms[(u, v)].updateParameters(reward=0)
                #update current P
                #print self.TotalPlayCounter
                self.currentP[u][v]['weight'] = self.arms[(u,v)].getProb(self.epsilon) 
    def getP(self):
        return self.currentP