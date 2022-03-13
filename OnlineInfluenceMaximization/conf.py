import os
from Oracle.generalGreedy import generalGreedy
from Oracle.degreeDiscount import degreeDiscountIC, degreeDiscountIC2, degreeDiscountIAC, degreeDiscountIAC2, degreeDiscountStar, degreeDiscountIAC3

save_address = "./SimulationResults"

graph_address = './datasets/Flickr/Small_Final_SubG.G'
prob_address = './datasets/Flickr/Probability.dic'
param_address = './datasets/Flickr/Small_nodeFeatures.dic'
edge_feature_address = './datasets/Flickr/Small_edgeFeatures.dic'
dataset = 'Flickr-Random' #Choose from 'default', 'NetHEPT', 'Flickr'

# graph_address = './datasets/Flickr/G_Union.G'
# prob_address = './datasets/Flickr/ProbUnion.dic'
# param_address = './datasets/Flickr/NodeFeaturesUnion.dic'
# edge_feature_address = './datasets/Flickr/EdgeFeaturesUnion.dic'
# dataset = 'Flickr-Cluster' #Choose from 'default', 'NetHEPT', 'Flickr'

# graph_address = './datasets/NetHEPT/Small_Final_SubG.G'
# prob_address = './datasets/NetHEPT/Probability.dic'
# param_address = './datasets/NetHEPT/Small_nodeFeatures.dic'
# edge_feature_address = './datasets/NetHEPT/Small_edgeFeatures.dic'
# dataset = 'NetHEPT' #Choose from 'default', 'NetHEPT', 'Flickr'


alpha_1 = 0.1
alpha_2 = 0.1
lambda_ = 0.4
gamma = 0.1
dimension = 4
seed_size = 300
iterations = 200

oracle = degreeDiscountIAC3
