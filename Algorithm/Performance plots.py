# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 02:28:40 2020
"""
#################################################################

# Set Working Directory

#import os

#os.chdir("C:/Users/omars/Desktop/Georgia/opioids/iterativeMDP/") # To change

from MDPtools import Generate_random_MDP, sample_MDP_with_features_list

import numpy as np
import matplotlib.pyplot as plt

#################################################################





#################################################################

# Load Libraries

from clustering import defaultNormal, UnifNormal, transformSamples, initializeClusters, splitter
from testing import *
#################################################################





#################################################################

# Set Parameters

n = 30

m = 5

reward_dep_action = False

deterministic = True

pfeatures = 2

sigma = [[0.03, 0], [0, 0.03]]

clustering = ''

n_clusters = 3

random_state = 0

k = n_clusters

classification = 'DecisionTreeClassifier'

n_iter = n-n_clusters


#################################################################





#################################################################

# Generate MDP

P, R = Generate_random_MDP(n,

                           m,

                           reward_dep_action=reward_dep_action,

                           deterministic=deterministic)

R0 = R

for i in range(n-3):

    R[i] = 0

R[n-2] = 1

R[n-1] = -1



# Updates the correct k automatically for initial clustering based on Risk

if all(clustering != i for i in ['KMeans', 'Agglomerative', 'Birch']):

    k = len(np.unique(np.array(R)))

#################################################################





#################################################################

# Generate Samples from Normal Distribution

#normal_distributions = defaultNormal(n,
#                                     pfeatures,
#                                     sigma)

normal_distributions = UnifNormal(n,
                                     pfeatures,
                                     sigma)



r2_1 = []
acc_1 = []
samples = []
for u in range(1,10):
    N = 100*u
    T = 5
    th = int(0.1*N*(T-1)/n)
    samples = samples + sample_MDP_with_features_list(P,
                                        R,
                                        normal_distributions,
                                        N-100*(u-1),
                                        T)
    df = transformSamples(samples,
                      pfeatures)
    df = initializeClusters(df,
                        T,
                        clustering=clustering,
                        n_clusters=n_clusters,
                        random_state=random_state)
    df_new = splitter(df,
                  T,
                  pfeatures,
                  k,
                  th,
                  classification,
                  n_iter,
                  OutputFlag = 0)
    print('Iteration ', u,'R2', R2_value(df_new,N,T))
    print('Iteration ', u,'Training accuracy ', training_accuracy(df_new)[0])
    r2_1.append(R2_value(df_new,N,T))
    acc_1.append(training_accuracy(df_new)[0])



R = R0

for i in range(n-int(n/4)):

    R[i] = 0

for i in range(int(n/4)+1,int(n/2)):
    R[i] = 1


print('SECOND IT')

r2_2 = []
acc_2 = []
samples = []
for u in range(1,10):
    N = 100*u
    T = 5
    th = int(0.1*N*(T-1)/n)
    samples = samples + sample_MDP_with_features_list(P,
                                        R,
                                        normal_distributions,
                                        N-100*(u-1),
                                        T)
    df = transformSamples(samples,
                      pfeatures)
    df = initializeClusters(df,
                        T,
                        clustering=clustering,
                        n_clusters=n_clusters,
                        random_state=random_state)
    df_new = splitter(df,
                  T,
                  pfeatures,
                  k,
                  th,
                  classification,
                  n_iter,
                  OutputFlag = 0)
    print('Iteration ', u,'R2', R2_value(df_new,N,T))
    print('Iteration ', u,'Training accuracy ', training_accuracy(df_new)[0])
    r2_2.append(R2_value(df_new,N,T))
    acc_2.append(training_accuracy(df_new)[0])


















#Tests regarding structure of rewards
#N = 130
#T = 5
#th = int(0.1*N*(T-1)/n)
#
#for u in range(n-3):
#    R0 = R
#    for i in range(u):
#        R0[i] = 0    
#
#    samples = samples + sample_MDP_with_features_list(P,
#                                        R0,
#                                        normal_distributions,
#                                        N-int(N/10),
#                                        T)
#    df = transformSamples(samples,
#                      pfeatures)
#    df = initializeClusters(df,
#                        T,
#                        clustering=clustering,
#                        n_clusters=n_clusters,
#                        random_state=random_state)
#    df_new = splitter(df,
#                  T,
#                  pfeatures,
#                  k,
#                  th,
#                  classification,
#                  n_iter,
#                  OutputFlag = 0)
#    print('Iteration ', u,'Training accuracy ', training_accuracy(df_new)[0])
#    l.append(training_accuracy(df_new)[0])
#
#plt.plot(np.arange(n-3),l)




