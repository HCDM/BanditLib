import pickle  # Save model

# import matplotlib.pyplot as plt
import re  # regular expression library
from random import random, choice  # for random strategy
from operator import itemgetter
import numpy as np
from scipy.sparse import csgraph
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import TruncatedSVD


def generateUserFeature(W):
    svd = TruncatedSVD(n_components=25)
    result = svd.fit(W).transform(W)
    return result


def vectorize(M):
    temp = []
    for i in range(M.shape[0] * M.shape[1]):
        temp.append(M.T.item(i))
    V = np.asarray(temp)
    return V


def matrixize(V, C_dimension):
    temp = np.zeros(shape=(C_dimension, len(V) / C_dimension))
    for i in range(len(V) / C_dimension):
        temp.T[i] = V[i * C_dimension : (i + 1) * C_dimension]
    W = temp
    return W


def readFeatureVectorFile(FeatureVectorsFileName):
    FeatureVectors = {}
    with open(FeatureVectorsFileName, "r") as f:
        f.readline()
        for line in f:
            line = line.split("\t")
            vec = line[1].strip("[]").strip("\n").split(";")
            FeatureVectors[int(line[0])] = np.array(vec).astype(np.float)
    return FeatureVectors


# This code simply reads one line from the source files of Yahoo!
def parseLine(line):
    userID, tim, pool_articles = line.split("\t")
    userID, tim = int(userID), int(tim)
    pool_articles = np.array(pool_articles.strip("[").strip("]").strip("\n").split(","))
    return userID, tim, pool_articles


def save_to_file(fileNameWrite, recordedStats, tim):
    with open(fileNameWrite, "a+") as f:
        f.write("data")  # the observation line starts with data;
        f.write("," + str(tim))
        f.write("," + ";".join([str(x) for x in recordedStats]))
        f.write("\n")


def initializeGW(Gepsilon, n, relationFileName):
    W = np.identity(n)
    with open(relationFileName) as f:
        for line in f:
            line = line.split("\t")
            if line[0] != "userID":
                if int(line[0]) <= n and (line[1]) <= n:
                    W[int(line[0])][int(line[1])] += 1
    G = W
    L = csgraph.laplacian(G, normed=False)
    I = np.identity(n)
    GW = I + Gepsilon * L  # W is a double stochastic matrix
    print(GW)
    return GW.T


# generate graph W(No clustering)
def initializeW(n, relationFileName):
    W = np.identity(n)

    with open(relationFileName) as f:
        for line in f:
            line = line.split("\t")
            if line[0] != "userID":
                if int(line[0]) <= n and int(line[1]) <= n:
                    W[int(line[0])][int(line[1])] += 1
                    # print W[int(line[0])][int(line[1])]
    row_sums = W.sum(axis=1)
    NormalizedW = W / row_sums[:, np.newaxis]
    W = NormalizedW

    print(W.T)
    print("Wtype", type(W))
    # initializeW_clustering(n,relationFileName, 5)
    return W.T


def initializeW_clustering(n, relationFileName, nClusters):
    W = np.identity(n + 1)
    with open(relationFileName) as f:
        f.readline()
        for line in f:
            line = line.split("\t")
            if int(line[0]) <= n and int(line[1]) <= n:
                W[int(line[0])][int(line[1])] += 1
    # KMeans

    # SpectralClustering
    spc = SpectralClustering(n_clusters=nClusters, affinity="precomputed")
    # spc = SpectralClustering(n_clusters=nClusters)
    spc.fit(W)  # What is the meaning
    label = spc.labels_

    with open(relationFileName + ".cluster", "w") as f:
        for i in range(n):
            f.write(str(label[i]) + "\n")

    NeighborW = np.zeros(shape=(nClusters, nClusters))
    for i in range(n):
        for j in range(n):
            if label[i] == label[j]:
                NeighborW[label[i]][label[j]] = 0
            else:
                NeighborW[label[i]][label[j]] += W[i][j]
    NormalizedNeighborW = normalizeByRow(NeighborW)

    newW = np.identity(nClusters) + NormalizedNeighborW
    print("newW", newW)

    NormalizednewW = normalizeByRow(newW)
    print("NormalizednewW", NormalizednewW.T)

    return NormalizednewW.T, newW, label


def initializeGW_clustering(Gepsilon, relationFileName, newW):
    G = newW
    n = newW.shape[0]
    L = csgraph.laplacian(G, normed=False)
    I = np.identity(n)
    GW = I + Gepsilon * L  # W is a double stochastic matrix
    print(GW)
    return GW.T


def initializeGW_label(Gepsilon, n, relationFileName, label, diagnol):
    W = np.identity(n)
    with open(relationFileName) as f:
        for line in f:
            line = line.split("\t")
            if (
                line[0] != "userID"
                and label[int(line[0])] != 10000
                and label[int(line[1])] != 10000
            ):  # 10000 means not top 100 user.
                W[label[int(line[0])]][label[int(line[1])]] += 1
    # don't need it
    """
    if diagnol=='1' or diagnol=='0':
        for i in range(n):
            W[i][i] = int(diagnol)
    """

    G = W
    L = csgraph.laplacian(G, normed=False)
    I = np.identity(n)
    GW = I + Gepsilon * L  # W is a double stochastic matrix
    print(GW)
    return GW.T


# generate graph W(No clustering)
def initializeW_label(n, relationFileName, label, diagnol, show_heatmap):
    W = np.identity(n)

    with open(relationFileName) as f:
        for line in f:
            line = line.split("\t")
            if (
                line[0] != "userID"
                and label[int(line[0])] != 10000
                and label[int(line[1])] != 10000
            ):  # 10000 means not top 100 user.
                W[label[int(line[0])]][label[int(line[1])]] += 1
    if show_heatmap:
        heatmap(W)
    # normalize
    if is_number(diagnol):
        for i in range(n):
            W[i][i] = 0
        W = normalizeByRow(W)
        if show_heatmap:
            heatmap(W)
        for i in range(n):
            W[i][i] = float(diagnol)
        if show_heatmap:
            heatmap(W)

    if diagnol == "Max":
        for i in range(n):
            W[i][i] = 0
        W = normalizeByRow(W)

        if show_heatmap:
            heatmap(W)
        for i in range(n):
            maxi = max(W[i])
            W[i][i] = maxi
        print(W)
        if show_heatmap:
            heatmap(W)
    if diagnol == "Opt":
        for i in range(n):
            W[i][i] = 0
            if sum(W[i] != 0):
                W[i][i] = np.linalg.norm(W[i]) ** 2 / sum(W[i])
            else:
                W[i][i] = 1
        print(W)
        if show_heatmap:
            heatmap(W)

    W = normalizeByRow(W)
    if show_heatmap:
        heatmap(W)
    print(W.T)
    return W.T


def read_cluster_label(labelfile):
    label = [0]
    # fin = open(labelfile,'r')
    for line in labelfile:
        label.append(int(line))
    return np.array(label)


def heatmap(X):
    plt.pcolor(X)
    plt.colorbar()
    plt.show()


def normalizeByRow(Matrix):
    row_sums = Matrix.sum(axis=1)

    for i in range(len(row_sums)):
        if row_sums[i] == 0:
            row_sums[i] = 0.00000000000001
    print(row_sums)
    NormalizednewMatrix = Matrix / row_sums[:, np.newaxis]
    return NormalizednewMatrix


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def model_dump(obj, filename, linenum):
    fout = open(filename + ".txt", "w")
    fout.write("line\t" + str(linenum))
    fout.close()
    fout = open(filename + ".model", "w")
    pickle.dump(obj, fout)
    fout.close()


def getcons(dim):
    cons = []
    cons.append({"type": "eq", "fun": lambda x: np.sum(x) - 1})

    for i in range(dim):
        cons.append({"type": "ineq", "fun": lambda x: x[i]})
        cons.append({"type": "ineq", "fun": lambda x: 1 - x[i]})

    return tuple(cons)
