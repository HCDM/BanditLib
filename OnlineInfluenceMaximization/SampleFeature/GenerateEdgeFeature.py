import random
import heapq
import datetime
import networkx as nx
import math
import argparse
import matplotlib.pyplot as plt
import time
import pickle
import numpy as np
import operator

save_dir = '../datasets/Flickr/'

dimension = 4
nodeDic = {}
edgeDic = {}
degree = []
G = pickle.load(open(save_dir+'Small_Final_SubG.G', 'rb'))
nodeDic = pickle.load(open(save_dir+'Small_nodeFeatures.dic', 'rb'))

for u in G.nodes():
	for v in G[u]:
		edgeDic[(u,v)] = np.outer(nodeDic[u][1], nodeDic[v][0]).reshape(-1)
pickle.dump(edgeDic, open(save_dir+'Small_edgeFeatures.dic', "wb" ))