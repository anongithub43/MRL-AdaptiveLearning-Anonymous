#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 19:47:03 2020

Model Class that runs the MRL algorithm on any data.  
"""

#################################################################
# Load Libraries
import pandas as pd
import numpy as np

from testing import predict_value_of_cluster, testing_value_error, model_trajectory, construct_P, construct_R
from MDPtools import SolveMDP
from scipy.stats import binom
import math
#################################################################

class MDP_model:
    def __init__(self):
        self.df = None # original dataframe from data
        self.pfeatures = None # number of features
        self.CV_error = None # error at minimum point of CV
        self.CV_error_all = None # errors of different clusters after CV
        self.training_error = None # training errors after last split sequence
        self.split_scores = None # cv error from splitter (if GridSearch used)
        self.opt_k = None # number of clusters in optimal clustering
        self.eta = None # incoherence threshold
        self.df_trained = None # dataframe after optimal training
        self.m = None # model for predicting cluster number from features #CHANGE NAME
        self.clus_pred_accuracy = None # accuracy score of the cluster prediction function
        self.P_df = None # Transition function of the learnt MDP, includes sink node if end state exists
        self.R_df = None # Reward function of the learnt MDP, includes sink node of reward 0 if end state exists
        self.nc = None # dataframe similar to P_df, but also includes 'count' and 'purity' cols
        self.v = None # value after MDP solved
        self.pi = None # policy after MDP solved
        self.P = None # P_df but in matrix form of P[a, s, s'], with alterations
                        # where transitions that do not pass the action and purity thresholds
                        # now lead to a new cluster with high negative reward
        self.R = None # R_df but in matrix form of R[a, s]
        
    
    # predict() takes a list of features and a time horizon, and returns 
    # the predicted value after all actions are taken in order
    def predict(self, 
                features, # list: list OR array of features 
                actions): # list: list of actions
        
        # predict initial cluster
        s = int(self.m.predict([features]))
        
        # predict value sum given starting cluster and action path
        v = predict_value_of_cluster(self.P_df,
                                        self.R_df,
                                        s,
                                        actions)
        return v
    
    # predict_forward() takes an ID & actions, and returns the predicted value
    # for this ID after all actions are taken in order
    def predict_forward(self,
                        ID,
                        actions):
        
        # cluster of this last point
        s = self.df_trained[self.df_trained['ID']==ID].iloc[-1, -2]
        
        # predict value sum given starting cluster and action path
        v = predict_value_of_cluster(self.P_df,
                                        self.R_df,
                                        s,
                                        actions)
        return v
    
    
    # testing_error() takes a df_test, then computes and returns the testing 
    # error on this trained model 
    def testing_error(self, 
                      df_test,
                      relative=False,
                      h=-1):
        
        error = testing_value_error(df_test, 
                            self.df_trained, 
                            self.m, 
                            self.pfeatures,
                            relative=relative,
                            h=h)
        
        return error
    
    
    # solve_MDP() takes the trained model as well as parameters for gamma, 
    # epsilon, whether the problem is a minimization or maximization one, 
    # and the threshold cutoffs to not include actions that don't appear enough
    # in each state, as well as purity cutoff for next_states that do not 
    # represent enough percentage of all the potential next_states, 
    # and returns the the value and policy. When solving the MDP, creates an 
    # artificial punishment state that is reached for state/action pairs that 
    # don't meet the above cutoffs; also creates a sink node of reward 0 
    # after the goal state or punishment state is reached. 
    def solve_MDP(self,
                  alpha = 0.2, # statistical alpha threshold
                  beta = 0.6, # statistical beta threshold
                  min_action_obs = -1, # int: least number of actions that must be seen
                  min_action_purity = 0.3, # float: percentage purity above which is acceptable
                  prob='max', # str: 'max', or 'min' for maximization or minimization problem
                  gamma=0.9, # discount factor
                  epsilon=10**(-10),
                  p=False):

        self.create_PR(alpha, beta, min_action_obs, min_action_purity, prob)
        return self.solve_helper(gamma, epsilon, p, prob, threshold=self.t_max*self.r_max*3)

    
    def create_PR(self, alpha, beta, min_action_obs, min_action_purity, prob): 
        
        # if default value, then scale the min threshold with data size, ratio 0.008
        if min_action_obs == -1:
            min_action_obs = max(5, 0.008*self.df_trained.shape[0])
            
        # adding two clusters: one for sink node (reward = 0), one for punishment state
        # sink node is R[s-2], punishment state is R[s-1]

        df0 = self.df_trained[self.df_trained['NEXT_CLUSTER'] != 'None']
        P_stoch = construct_P(df0)
        R_stoch = construct_R(df0)
        
        P_df = self.P_df.copy()
        R_df = self.R_df.copy()
        P_df['count'] = self.nc['count']
        P_df['purity'] = self.nc['purity']
        P_df = P_df.reset_index()
        R_df = R_df.reset_index()

        # record parameters of transition dataframe
        a = P_df['ACTION'].nunique()

        s = max(max(P_df['CLUSTER'].unique()), max(R_df.index.unique())) + 1
        actions = P_df['ACTION'].unique()
        
        # Take out rows that don't pass statistical alpha test
        P_alph = P_df.loc[(1-binom.cdf(P_df['purity']*(P_df['count']), P_df['count'],\
                                      beta))<=alpha]
        
        
        # Take out rows where actions or purity below threshold
        P_thresh = P_alph.loc[(P_alph['count']>min_action_obs)&(P_alph['purity']>min_action_purity)]
        
        # Take note of rows where we have missing actions:
        incomplete_clusters = np.where(P_df.groupby('CLUSTER')['ACTION'].count()<a)[0]
        missing_pairs = []
        for c in incomplete_clusters:
            not_present = np.setdiff1d(actions, P_df.loc[P_df['CLUSTER']==c]['ACTION'].unique())
            for u in not_present:
                missing_pairs.append((c, u))
        
        P = np.zeros((a, s+1, s+1))
        
        # model transitions
        for row in P_thresh.itertuples():
            x, y, z = int(row[2]), row[1], row[3] #ACTION, CLUSTER, NEXT_CLUSTER
            P[x, y, z] = 1 

        # reinsert transition for cluster/action pairs taken out by alpha test
        excl_alph = P_df.loc[(1-binom.cdf(P_df['purity']*P_df['count'], P_df['count'],\
                                      beta))>alpha]

        for row in excl_alph.itertuples():
            c, u = row[1], int(row[2]) #CLUSTER, ACTION
            P[u, c, -1] = 1
            
        # reinsert transition for cluster/action pairs taken out by threshold
        excl = P_df.loc[(P_df['count']<=min_action_obs)|(P_df['purity']<=min_action_purity)]
        for row in excl.itertuples():
            c, u = row[1], int(row[2]) #CLUSTER, ACTION
            P[u, c, -1] = 1
            
        # reinsert transition for missing cluster-action pairs
        for pair in missing_pairs:
            c, u = pair
            P[int(u), c, -1] = 1

        # replacing correct sink node transitions
        nan = P_df.loc[P_df['count'].isnull()]
        for row in nan.itertuples():
            c, u, t = row[1], row[2], row[3] #CLUSTER, ACTION, NEXT_CLUSTER
            P[int(u), c, t] = 1
        
        # punishment node to 0 reward sink (if sink was created in get_MDP):
        if 'End' in self.df_trained['NEXT_CLUSTER'].unique():
            for u in range(a):
                P[int(u), -1, -2] = 1
    
        # append high negative reward for incorrect / impure transitions
        R = []
        
        T_max = self.df_trained['TIME'].max()
        r_max = abs(self.df_trained['RISK']).max()
        self.t_max = T_max
        self.r_max = r_max
        # adding sink node
        for i in range(a):
            if prob == 'max':
                # take T-max * max(abs(reward)) * 2 
                R.append(np.append(np.array(self.R_df), -self.t_max*self.r_max*2))
            else:
                R.append(np.append(np.array(self.R_df), self.t_max*self.r_max*2))
        R = np.array(R)
        self.P = P
        self.R = R

        # print('P: ', self.P[0, 5, :])
        # print('P df: ', self.P_df[self.P_df['ACTION'] == 0 and self.P_df['CLUSTER'] == 5])


    def solve_helper(self, gamma, epsilon, p, prob, threshold): 
        # solve the MDP, with an extra threshold to guarantee value iteration
        # ends if gamma=1
        v, pi = SolveMDP(self.P, self.R, gamma, epsilon, p, prob, threshold=threshold)
        
        # store values and policies and matrices
        self.v = v
        self.pi = pi
        
        return v, pi
    
    # opt_model_trajectory() takes a start state, a transition function, 
    # indices of features to be considered, a transition function, and an int
    # for number of points to be plotted. Plots and returns the transitions
    def opt_model_trajectory(self,
                             x, # start state as tuple or array
                             f, # transition function of the form f(x, u) = x'
                             f1=0, # index of feature 1 to be plotted
                             f2=1, # index of feature 2 to be plotted
                             n=30): # points to be plotted
    
        xs, ys, all_vecs = model_trajectory(self, f, x, f1, f2, n)
        return xs, ys 
    
    
    # update_predictor
    def update_predictor(self, predictor):
        self.m = predictor
        return