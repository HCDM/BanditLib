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
file_address_5 = '../datasets/Flickr/flickrEdges.txt'
save_dir = '../datasets/Flickr/'

featureDic = {}
thetaDic = {}
PDic = {}
NodeDegree = {}

with open(file_address_5) as f:
	counter = 0
	for line in f:
		if counter >=4:
			data = line.split(' ')
			u = int(data[0])
			v = int(data[1])
			if u not in NodeDegree:
				NodeDegree[u] = 1
			else:
				NodeDegree[u]  +=1
			if v not in NodeDegree:
				NodeDegree[v] = 1
			else:
				NodeDegree[v]  +=1

		counter +=1
print('Finish Processing, Start dumping')
print('Total Nodes', len(NodeDegree))
print('maxDegree', max(iter(NodeDegree.items()), key=operator.itemgetter(1))[1], min(iter(NodeDegree.items()), key=operator.itemgetter(1))[1])
print('AverageDegree', sum(NodeDegree.values())/float(len(NodeDegree)))

FinalNodeList =[]
FinalNodeDegree  = {}
max_degree = 6000
min_degree = 0

for key in NodeDegree:
	if NodeDegree[key] <= max_degree and NodeDegree[key] >= min_degree:
		FinalNodeList.append(key)
		FinalNodeDegree[key] = NodeDegree[key]

print('Total Nodes', len(FinalNodeList))
print('maxDegree', max(iter(FinalNodeDegree.items()), key=operator.itemgetter(1))[1], min(iter(FinalNodeDegree.items()), key=operator.itemgetter(1))[1])
print('AverageDegree', sum(FinalNodeDegree.values())/float(len(FinalNodeDegree)))

	

pickle.dump( FinalNodeList, open(save_dir+'NodesDegree'+str(max_degree)+'_'+str(min_degree)+'.list', "wb" ))
