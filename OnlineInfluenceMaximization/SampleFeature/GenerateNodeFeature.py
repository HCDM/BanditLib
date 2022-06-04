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
def featureUniform(dimension, scale):
	vector = np.array([random.random() for i in range(dimension)])
	l2_norm = np.linalg.norm(vector, ord =2)
	
	vector = vector/l2_norm

	gau = np.random.normal(0.5, 0.5, 1)[0]
	while gau < 0 or gau > 1:
		gau = np.random.normal(0.5, 0.5, 1)[0]

	vector = vector / scale * gau * 1.5

	return vector

dimension = 4
nodeDic = {}
edgeDic = {}
degree = []
G = pickle.load(open(save_dir+'Small_Final_SubG.G', 'rb'))
for u in G.nodes():
	s = len(G.edges(u))
	nodeDic[u] = [featureUniform(dimension, 1), featureUniform(dimension, s)]
for u in G.nodes():
	d = 0
	for v in G[u]:
		prob = np.dot(nodeDic[u][1], nodeDic[v][0]) * 4
		if prob > 1:
			prob = 1
		if prob < 0:
			prob = 0
		edgeDic[(u,v)] = prob
		d += prob
	degree.append(d)
pickle.dump(nodeDic, open(save_dir+'Small_nodeFeatures.dic', "wb" ))
pickle.dump(edgeDic, open(save_dir+'Probability.dic', "wb" ))

plt.hist(degree)
plt.show()