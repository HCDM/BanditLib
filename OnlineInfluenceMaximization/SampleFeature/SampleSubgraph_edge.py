import matplotlib.pyplot as plt
import time
import pickle
import networkx as nx
import random

save_dir = '../datasets/Flickr/'
max_degree = 6000
min_degree = 0
NodeList = pickle.load(open(save_dir+'NodesDegree'+str(max_degree)+'_'+str(min_degree)+'.list', "rb" ))
print('Done with loading List')

NodeNum = len(NodeList)
print(NodeNum)
Small_NodeList = [NodeList[i] for i in sorted(random.sample(range(len(NodeList)), NodeNum//6))]
NodeList = Small_NodeList
print(len(NodeList))
pickle.dump(NodeList, open(save_dir+'Small_NodeList.list', "wb" ))


file_address = save_dir+'flickrEdges.txt'
start = time.time()
G = nx.DiGraph()
print('Start Reading')
with open(file_address) as f:
	#n, m = f.readline().split(',')
	for line in f:
		if line[0] != '#':
			u, v = list(map(int, line.split(' ')))
			if u in NodeList and v in NodeList:
				try:
					G[u][v]['weight'] += 1
				except:
					G.add_edge(u,v, weight=1)
				try:
					G[v][u]['weight'] += 1
				except:
					G.add_edge(v, u, weight=1)
print('Start Dumping')
print(len(G.nodes()), len(G.edges()))
pickle.dump( G, open(save_dir+'Small_Final_SubG.G', "wb" ))
#G = pickle.load(open(file_address + '.G', 'rb'))
#It may takes two minutes
print('Built Flixster graph G', time.time() - start, 's')