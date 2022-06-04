from collections import Counter
from math import log
import numpy as np
import copy
from random import *
from custom_errors import FileExists

def sigmoid(x):
    return 1/(1+np.exp(-x))

def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

# specific : dictionary of arguments specific to the algorithm and supersedes all other parameter settings
# general : dictionary of arguments shared between all algorithms, supersedes everything except specific parameters
def createBaseAlgDict(specific, general, W, system_params):
    base_dict = {
        "dimension": system_params["context_dim"],
        "n_users": system_params["n_users"],
        "parameters": {"Theta": False, "CoTheta": False, "W": False, "V": False},
    }
    middle = update_dict(specific, general)
    return_dict = update_dict(middle, base_dict)
    return return_dict


# base_dict: dictionary of any additional default arguments required for that algorithm
def createSpecificAlgDict(specific, general, W, system_params, base_dict):
    # Define all of the required default arguments across all algorithms
    starter = createBaseAlgDict(specific, general, W, system_params)
    tmp = update_dict(specific, general)
    tmp2 = update_dict(tmp, base_dict)
    final_dict = update_dict(tmp2, starter)
    print(final_dict)
    return final_dict


def createNeuralLinearDict(specific, general, W, system_params):
    base_dict = {
        "lambda_": 0.25,
        "parameters": {
            "Theta": True,
        },
    }
    return createSpecificAlgDict(specific, general, W, system_params, base_dict)

def createNeuralLinearLikelihoodMatchingDict(specific, general, W, system_params):
    base_dict = {
        "lambda_": 0.25,
        "parameters": {
            "Theta": True,
        },
    }
    return createSpecificAlgDict(specific, general, W, system_params, base_dict)


def createNeuralUCBDict(specific, general, W, system_params):
    base_dict = {
        "alpha": 0.3,
        "lambda_": 0.1,
        "parameters": {
            "Theta": True,
        },
    }
    return createSpecificAlgDict(specific, general, W, system_params, base_dict)

def createLinUCBDict(specific, general, W, system_params):
    base_dict = {
        "alpha": 0.3,
        "lambda_": 0.1,
        "parameters": {
            "Theta": True,
        },
    }
    return createSpecificAlgDict(specific, general, W, system_params, base_dict)


def createFairUCBDict(specific, general, W, system_params):
    return createLinUCBDict(specific, general, W, system_params)


def createCoLinUCBDict(specific, general, W, system_params):
    base_dict = {
        "W": W,
        "alpha": 0.3,
        "lambda_": 0.1,
        "use_alpha_t": False,
        "n_users": system_params["n_clusters"],
        "parameters": {
            "CoTheta": True,
        },
    }
    return createSpecificAlgDict(specific, general, W, system_params, base_dict)


def createGOBLinDict(specific, general, W, system_params):
    return createCoLinUCBDict(specific, general, W, system_params)


def createHLinUCBDict(specific, general, W, system_params):
    base_dict = {
        "context_dimension": system_params["context_dim"],
        "latent_dimension": system_params["latent_dim"],
        "alpha": 0.3,
        "alpha2": 0.1,
        "lambda_": 0.1,
        "n_articles": system_params["n_articles"],
        "parameters": {"CoTheta": True, "V": True},
    }
    return createSpecificAlgDict(specific, general, W, system_params, base_dict)


def createUCBPMFDict(specific, general, W, system_params):
    base_dict = {
        "n": system_params["n_users"],
        "itemNum": system_params["n_articles"],
        "sigma": np.sqrt(0.5),
        "sigmaU": 1,
        "sigmaV": 1,
        "alpha": 0.1,
        "parameters": {
            "Theta": False,
            "CoTheta": True,
            "W": False,
            "V": True,
        },
    }
    return createSpecificAlgDict(specific, general, W, system_params, base_dict)


def createFactorUCBDict(specific, general, W, system_params):
    base_dict = {
        "W": W,
        "context_dimension": system_params["context_dim"],
        "latent_dimension": system_params["latent_dim"],
        "alpha": 0.05,
        "alpha2": 0.025,
        "lambda_": 0.1,
        "n": system_params["n_users"],
        "itemNum": system_params["n_articles"],
        "parameters": {
            "Theta": False,
            "CoTheta": True,
            "W": False,
            "V": True,
        },
    }
    return createSpecificAlgDict(specific, general, W, system_params, base_dict)


def createCLUBDict(specific, general, W, system_params):
    base_dict = {
        "alpha": 0.1,
        "lambda_": 0.1,
        "n": system_params["n_users"],
        "alpha_2": 0.5,
        "cluster_init": "Erdos-Renyi",
        "parameters": {
            "Theta": False,
            "CoTheta": False,
            "W": False,
            "V": False,
        },
    }
    return createSpecificAlgDict(specific, general, W, system_params, base_dict)


def createPTSDict(specific, general, W, system_params):
    base_dict = {
        "particle_num": 10,
        "n": system_params["n_users"],
        "itemNum": system_params["n_articles"],
        "sigma": np.sqrt(0.5),
        "sigmaU": 1,
        "sigmaV": 1,
        "parameters": {
            "Theta": False,
            "CoTheta": False,
            "W": False,
            "V": False,
        },
    }
    return createSpecificAlgDict(specific, general, W, system_params, base_dict)


def createThompsonSamplingDict(specific, general, W, system_params):
    base_dict = {
        "lambda_": 0.1,
        "R": 0.0001,
        "delata": 0.1,
        "epsilon": 0.05,
        "parameters": {
            "Theta": True,
        },
    }
    return createSpecificAlgDict(specific, general, W, system_params, base_dict)


def createLinPHEDict(specific, general, W, system_params):
    base_dict = {
        "a": 0.1,
        "lambda_": 0.1,
        "parameters": {
            "Theta": True,
        },
    }
    return createSpecificAlgDict(specific, general, W, system_params, base_dict)


def update_dict(a, b):
    c = copy.deepcopy(b)
    for i in a:
        if i == "parameters":
            for j in a[i]:
                if j in b["parameters"]:
                    c[i][j] = a[i][j]
        else:
            c[i] = a[i]
    return c


def gaussianFeature(dimension, argv):
    mean = argv["mean"] if "mean" in argv else 0
    std = argv["std"] if "std" in argv else 1

    mean_vector = np.ones(dimension) * mean
    stdev = np.identity(dimension) * std
    vector = np.random.multivariate_normal(np.zeros(dimension), stdev)

    l2_norm = np.linalg.norm(vector, ord=2)
    if "l2_limit" in argv and l2_norm > argv["l2_limit"]:
        "This makes it uniform over the circular range"
        vector = vector / l2_norm
        vector = vector * (random())
        vector = vector * argv["l2_limit"]

    if mean is not 0:
        vector = vector + mean_vector

    vectorNormalized = []
    for i in range(len(vector)):
        vectorNormalized.append(vector[i] / sum(vector))
    return vectorNormalized
    # return vector


def featureUniform(dimension, argv=None):
    vector = np.array([random() for _ in range(dimension)])

    l2_norm = np.linalg.norm(vector, ord=2)

    vector = vector / l2_norm
    return vector


def getBatchStats(arr):
    return np.concatenate((np.array([arr[0]]), np.diff(arr)))


def checkFileExists(filename):
    try:
        with open(filename, "r"):
            return 1
    except IOError:
        return 0


def fileOverWriteWarning(filename, force):
    if checkFileExists(filename):
        if force == True:
            print("Warning : fileOverWriteWarning {}".format(filename))
        else:
            raise FileExists(filename)

def vectorize(M):
    # temp = []
    # for i in range(M.shape[0]*M.shape[1]):
    # 	temp.append(M.T.item(i))
    # V = np.asarray(temp)
    # return V
    return np.reshape(M.T, M.shape[0] * M.shape[1])


def matrixize(V, C_dimension):
    # temp = np.zeros(shape = (C_dimension, len(V)/C_dimension))
    # for i in range(len(V)/C_dimension):
    # 	temp.T[i] = V[i*C_dimension : (i+1)*C_dimension]
    # W = temp
    # return W
    # To-do: use numpy built-in function reshape.
    return np.transpose(np.reshape(V, (int(len(V) / C_dimension), C_dimension)))
