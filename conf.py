'''
Created on May 12, 2015

@author: hongning
'''
import os

sim_files_folder = "./Simulation_MAB_files"

result_folder = "./SimulationResults"

save_address = "./SimulationResults"

LastFM_save_address = "./LastFMResults"
Delicious_save_address = "./DeliciousResults"
Yahoo_save_address = "./YahooResults"


save_addressResult = "./Results/Sparse"


datasets_address = '/if15/qw2ky/MyResearch/datasets/'  # should be modified accoring to the local address of yahoo address
#datasets_address = '/home/qingyun/Dropbox/workspace/datasets'
#yahoo_address = os.path.join(datasets_address, "R6")

Kmeansdata_address = './Dataset/Yahoo/YahooKMeansModel'
LastFM_address = './Dataset/hetrec2011-lastfm-2k'
Delicious_address = './Dataset/hetrec2011-delicious-2k'

LastFM_FeatureVectorsFileName = os.path.join(LastFM_address, 'Arm_FeatureVectors_2.dat')
LastFM_relationFileName = os.path.join(LastFM_address, 'user_friends.dat.mapped')

Delicious_FeatureVectorsFileName = os.path.join(Delicious_address, 'Arm_FeatureVectors_2.dat')
Delicious_relationFileName = os.path.join(Delicious_address, 'user_contacts.dat.mapped')

Yelp_save_address = "./YelpResults"
Yelp_address = '../../../Yelp'
Yelp_FeatureVectorsFileName = os.path.join(Yelp_address, 'Arm_FeatureVectors_2.dat')
Yelp_relationFileName = os.path.join(Yelp_address, 'user_friends.dat.mapped')
