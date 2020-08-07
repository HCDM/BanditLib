'''
Created on May 12, 2015

@author: hongning
'''
import os

sim_files_folder = "./Simulation_MAB_files"

result_folder = "./Results/SimulationResults"

save_address = "./Results/SimulationResults"

LastFM_save_address = "./Results/LastFMResults"
Delicious_save_address = "./Results/DeliciousResults"
Yahoo_save_address = "./Results/YahooResults"


save_addressResult = "./Results/Sparse"


datasets_address = '/if15/qw2ky/MyResearch/datasets/'  # should be modified accoring to the local address of yahoo address
#datasets_address = '/home/qingyun/Dropbox/workspace/datasets'

Kmeansdata_address = './Dataset/Yahoo/YahooKMeansModel'
LastFM_address = './Dataset/hetrec2011-lastfm-2k'
Delicious_address = './Dataset/hetrec2011-delicious-2k'
Yahoo_address = '/../../../zf15/hw7ww/Bandit/YahooData/R6'

LastFM_FeatureVectorsFileName = os.path.join(LastFM_address, 'Arm_FeatureVectors_2.dat')
LastFM_relationFileName = os.path.join(LastFM_address, 'user_friends.dat.mapped')

Delicious_FeatureVectorsFileName = os.path.join(Delicious_address, 'Arm_FeatureVectors_2.dat')
Delicious_relationFileName = os.path.join(Delicious_address, 'user_contacts.dat.mapped')

Yelp_save_address = "./YelpResults"
Yelp_address = '../../../Yelp'
Yelp_FeatureVectorsFileName = os.path.join(Yelp_address, 'Arm_FeatureVectors_2.dat')
Yelp_relationFileName = os.path.join(Yelp_address, 'user_friends.dat.mapped')
