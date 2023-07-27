import sys
sys.path.append('/Users/lowell/MRL/Algorithm')
sys.path.append('/Users/lowell/MRL/Maze')

from MRL_model import MRL_model
from maze_functions import value_diff, get_maze_MDP, get_maze_transition_reward, createSamples, opt_model_trajectory
from testing import plot_features
from MDPtools import SolveMDP
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

# this file takes the data trained on by adaptive models and trains mrl on them, 
# getting the optimality gaps accordingly. 

mrl_model = MRL_model()

# code to help get the OG_Cluster column 
def get_cluster(feature_0, feature_1, x, l): 
    # x: cell length
    # l: grid length
    i = -feature_1
    j = feature_0
    cluster = math.floor(i/x)*math.ceil(l/x) + math.floor(j/x)

    preserve_end_state = False
    # if end state is preserved, send the clusters in the 'end box' to the actual end 
    if preserve_end_state: 
        # if cluster in the bottom right square
        reward_square_boundary = l-1
        if i >= reward_square_boundary and j >= reward_square_boundary: 
            cluster = math.ceil(l/x) + math.floor(l - .00001/x)

    return cluster 

#df = createSamples(50, 200, 'maze-sample-5x5-v0', 0.5, reseed=True)
df = pd.read_csv('data/trials-updated.csv', index_col=0)
df = df.iloc[:20000]

df['OG_CLUSTER'] = df.apply(lambda x: get_cluster(x.FEATURE_0, x.FEATURE_1, 1, 5), axis=1)

unique_clusters = df['OG_CLUSTER'].unique()
ascending_clusters = sorted(unique_clusters)

i = 0 
# # remaps clusters since some grid cells won't have any points and we need clusters to be ascending order
for cluster in ascending_clusters: 
    df.loc[df['OG_CLUSTER'] == cluster, 'OG_CLUSTER'] = i
    i = i + 1
    
# mrl parameters defined here
max_k = 25
classification = 'DecisionTreeClassifier'
split_classifier_params = {'random_state':0, 'max_depth':2}
clustering = 'Agglomerative'
n_clusters = None
distance_threshold = 0.5
precision_thresh = 1e-14
random_state = 0
pfeatures = 2
gamma = 1
actions = [0, 1, 2, 3]
cv = 5
th = 0
eta = 25

vals = []
Ns = [1000, 2000, 3000, 4000, 5000, 7500, 10000, 15000]

for n in Ns: 
    df_new = df.iloc[:n]

    mrl_model.fit(df_new, # df: dataframe in the format ['ID', 'TIME', ...features..., 'RISK', 'ACTION']
                pfeatures, # int: number of features
                -1, # int: time horizon (# of actions we want to optimize) -1 = entire path
                gamma, # discount factor
                max_k, # int: number of iterations
                distance_threshold, # clustering diameter for Agglomerative clustering
                cv, # number for cross validation
                th, # splitting threshold
                eta, # incoherence threshold
                precision_thresh, # precision threshold
                classification, # classification method
                split_classifier_params, # classification params
                clustering,# clustering method from Agglomerative, KMeans, and Birch
                n_clusters, # number of clusters for KMeans
                random_state, # random seed
                plot=False,
                optimize=True,
                verbose=False)
        
    # get mrl optimality gap 
    mrl_model.solve_MDP(gamma=1, epsilon=1e-4, alpha=1, min_action_purity=-1, min_action_obs=0)

    print('v, pi: ', mrl_model.v, mrl_model.pi)

    maze = 'maze-sample-5x5-v0'
    T_max = 25
    K = 100

    P, R = get_maze_MDP(maze)
    f, rw = get_maze_transition_reward(maze)

    true_v, true_pi = SolveMDP(P, R, prob='max', gamma=1, epsilon=1e-4)

    print('getting mrl model opt gap')

    opt_gap = value_diff([mrl_model], [0], K, T_max, P, R, f, rw, true_v, true_pi) 

    #print(opt_gap[0]/0.04)

    vals.append(opt_gap[0]/0.04)



fig, ax = plt.subplots()
print(vals)
ax.plot(Ns, vals)
ax.set_title("Optimality gap vs episode number")
ax.set_xlabel("Number of datapoints trained on")
ax.set_ylabel("Reward")
ax.set_ylim(0, 40)
plt.show()

