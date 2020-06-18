import re 			# regular expression library
from random import random, choice 	# for random strategy
from operator import itemgetter
import numpy as np
from scipy.sparse import csgraph
from scipy.spatial import distance
import pickle
#import matplotlib.pyplot as plt

def vectorize(M):
	temp = []
	for i in range(M.shape[0]*M.shape[1]):
		temp.append(M.T.item(i))
	V = np.asarray(temp)
	return V

def matrixize(V, C_dimension):
	temp = np.zeros(shape = (C_dimension, len(V)/C_dimension))
	for i in range(len(V)/C_dimension):
		temp.T[i] = V[i*C_dimension : (i+1)*C_dimension]
	W = temp
	return W

#read centroids from file
def getClusters(fileNameWriteCluster):
    with open(fileNameWriteCluster, 'r') as f:
        clusters = []
        for line in f:
            vec = []
            line = line.split(' ')
            for i in range(len(line)-1):
                vec.append(float(line[i]))
            clusters.append(np.asarray(vec))
        return np.asarray(clusters)

def getArticleDic(fileNameRead):
    with open(fileNameRead, 'r') as f:
        articleDict = {}
        l = 0
        for line in f:
            featureVec = []
            if l >=1:
                line = line.split(';')
                word = line[1].split('  ')
                
		if len(word)==5:
                    for i in range(5):
                        featureVec.append(float(word[i]))
                    if int(line[0]) not in articleDict:
                        articleDict[int(line[0])] = np.asarray(featureVec)
            l +=1
    return articleDict

# get cluster assignment of V, M is cluster centroids
def getIDAssignment(V, M):
        MinDis = float('+inf')
        assignment = None
        for i in range(M.shape[0]):
            dis = distance.euclidean(V, M[i])
            if dis < MinDis:
                assignment = i
                MinDis = dis
        return assignment

# This code simply reads one line from the source files of Yahoo!
def parseLine(line):
        line = line.split("|")
        
        tim, articleID, click = line[0].strip().split(" ")
        tim, articleID, click = int(tim), int(articleID), int(click)
        user_features = np.array([float(x.strip().split(':')[1]) for x in line[1].strip().split(' ')[1:]])
        
        pool_articles = [l.strip().split(" ") for l in line[2:]]
        pool_articles = np.array([[int(l[0])] + [float(x.split(':')[1]) for x in l[1:]] for l in pool_articles])
        return tim, articleID, click, user_features, pool_articles


# read line with userID instead of user features
def parseLine_ID(line):
        line = line.split("|")
        
        tim, articleID, click = line[0].strip().split(" ")
        tim, articleID, click = int(tim), int(articleID), int(click)
        
        userID = int(line[1].strip())
        
        pool_articles = [l.strip().split(" ") for l in line[2:]]
        pool_articles = np.array([[int(l[0])] + [float(x.split(':')[1]) for x in l[1:]] for l in pool_articles])
        return tim, articleID, click, userID, pool_articles


def save_to_file(fileNameWrite, recordedStats, tim):
    with open(fileNameWrite, 'a+') as f:
        f.write('data') # the observation line starts with data;
        f.write(',' + str(tim))
        f.write(',' + ';'.join([str(x) for x in recordedStats]))
        f.write('\n')



def initializeGW(W, epsilon):  
    n = len(W)
    G = np.zeros(shape = (n, n))
    for i in range(n):
        for j in range(n):
            if W[i][j] > 0:
                G[i][j] = 1
    L = csgraph.laplacian(G, normed = False)
    I = np.identity(n)
    GW = I + epsilon*L
    print GW
    
    return GW


def initializeW(userFeatureVectors, sparsityLevel):
    n = len(userFeatureVectors)
    W = np.zeros(shape = (n, n))

    for i in range(n):
            sSim = 0
            for j in range(n):
                sim = np.dot(userFeatureVectors[i], userFeatureVectors[j])
                W[i][j] = sim
                sSim += sim
            
            W[i] /= sSim
    SparseW = W
    
    if sparsityLevel > 0 and sparsityLevel <n:
        print 'Yesyesyes'
        for i in range(n):
            similarity = sorted(W[i], reverse = True)
            threshold = similarity[sparsityLevel]
            for j in range(n):
                if W[i][j] <= threshold:
                    SparseW[i][j] = 0
            SparseW[i] /= sum(SparseW[i])
    
    print 'SparseW', SparseW
    return SparseW.T
    


def initializeW_opt(userFeatureVectors, sparsityLevel):
    n = len(userFeatureVectors)
    W = np.zeros(shape = (n, n))
    
    for i in range(n):
            sSim = 0
            for j in range(n):
                sim = np.dot(userFeatureVectors[i], userFeatureVectors[j])
                if i == j:
                    W[i][j] = 0
                else:
                    W[i][j] = sim
                sSim += sim            
    SparseW = W
    
    if sparsityLevel > 0 and sparsityLevel <n:
        for i in range(n):
            similarity = sorted(W[i], reverse = True)
            threshold = similarity[sparsityLevel]
            for j in range(n):
                if W[i][j] <= threshold:
                    SparseW[i][j] = 0

    for i in range(n):
        SparseW[i][i] =0
        if sum(SparseW[i])!=0:
            SparseW[i][i] = np.linalg.norm(SparseW[i])**2/sum(SparseW[i])
        else:
            SparseW[i][i] = 1
        SparseW[i] /=sum(SparseW[i])
    print 'SparseW --Opt', SparseW
    return SparseW.T


def showheatmap(W):
    plt.pcolor(W)
    plt.colorbar()
    plt.show()

def model_dump(obj, filename, line, day):
    fout = open(filename +'.txt', 'w')
    fout.write("day\t"+str(day))
    fout.write("line\t"+str(linenum))
    fout.close()    
    fout = open(filename +'.model', 'w')
    pickle.dump(obj, fout)
    fout.close()


# data structure to store ctr   
class articleAccess():
    def __init__(self):
        self.accesses = 0.0 # times the article was chosen to be presented as the best articles
        self.clicks = 0.0   # of times the article was actually clicked by the user
        self.CTR = 0.0      # ctr as calculated by the updateCTR function

    def updateCTR(self):
        try:
            self.CTR = self.clicks / self.accesses
        except ZeroDivisionError: # if it has not been accessed
            self.CTR = -1
        return self.CTR

    def addrecord(self, click):
        self.clicks += click
        self.accesses += 1
