#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 20:57:14 2020
"""
#################################################################
# Load Libraries
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
import gym

import sys
sys.path.append('/Users/janiceyang/Dropbox (MIT)/ORC UROP/Opioids/Algorithm/')
from MDPtools import SolveMDP

# ignore warning, reference: https://stackoverflow.com/questions/14861891/runtimewarning-invalid-value-encountered-in-divide
np.seterr(divide='ignore', invalid='ignore')

# If Spyder keeps re-calling gym_maze module, run the following code
'''
# deleting the registry that spyder has already 
for env in gym.envs.registry.env_specs.copy():
    #print(env)
    if 'maze' in env:
        print('Remove {} from registry'.format(env))
        del gym.envs.registry.env_specs[env]
'''
import gym_maze

from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm
#################################################################

# list of maze options to choose from:
mazes = {1: 'maze-v0',
         2: 'maze-sample-3x3-v0',
         3: 'maze-random-3x3-v0',
         4: 'maze-sample-5x5-v0',
         5: 'maze-random-5x5-v0',
         6: 'maze-sample-10x10-v0',
         7: 'maze-random-10x10-v0',
         8: 'maze-sample-100x100-v0',
         9: 'maze-random-100x100-v0',
         10: 'maze-random-10x10-plus-v0',
         11: 'maze-random-20x20-plus-v0',
         12: 'maze-random-30x30-plus-v0'}


# createSamples() N, the number of IDs, T_max, the maximum timestep if win-
# state is not reached, and maze (str of maze-name from above dict). Generates 
# the samples based on a ratio r of randomness, and (1-r) actions taken according
# to the optimal policy of the maze. If 
# reseed = True, selects a random location in the next cell, otherwise reseed=
# False makes robot travel to the next cell with the same offset
# returns a dataframe of the form ['ID', 'TIME', features, 'ACTION', 'RISK']
def createSamples(N, T_max, maze, r, reseed=False):
    # solving for optimal policy
    P, R = get_maze_MDP(maze)
    v, pi = SolveMDP(P, R, 0.98, 1e-10, False, 'max')
    
    # initialize environment
    env = gym.make(maze)
    transitions = []
    l = env.unwrapped.maze_size[0]
    
    for i in range(N):
        
        # initialize variables 
        offset = np.array((random.random(), random.random()))
        obs = env.reset()
        
        # initialize first reward
        reward = -1/(l*l)
        x = obs + offset
        ogc = int(obs[0] + obs[1]*l)
        
        for t in range(T_max):
            # take random step or not
            if random.random() <= r:
                action = env.action_space.sample()
            else:
                action = int(pi[ogc])
            
            transitions.append([i, t, x, action, reward, ogc])
            
            new_obs, reward, done, info = env.step(action)
            ogc = int(new_obs[0] + new_obs[1]*l)
            
            # if reseed, create new offset
            if reseed:
                offset = np.array((random.random(), random.random()))
                
            # if end state reached, append one last no action no reward
            if done:
                transitions.append([i, t+1, new_obs+offset, 'None', reward, ogc])
                break
            x = new_obs + offset
    
    env.close()
    
    df = pd.DataFrame(transitions, columns=['ID', 'TIME', 'x', 'ACTION', 'RISK', 'OG_CLUSTER'])
                      
    features = df['x'].apply(pd.Series)
    features = features.rename(columns = lambda x : 'FEATURE_' + str(x))
    
    df_new = pd.concat([df.iloc[:, :2], features, df.iloc[:, 3:]], axis=1)
    df_new['FEATURE_1'] = -df_new['FEATURE_1']
    
    return df_new
             

# opt_model_trajectory() takes a trained model, the maze used to train this model, and 
# plots the path of the optimal solution through the maze. returns the path
def opt_model_trajectory(m, maze, alpha=0.2, min_action_obs=0, min_action_purity=0):
    if m.v is None:
        m.solve_MDP(alpha, min_action_obs, min_action_purity, gamma=1)
    env = gym.make(maze)
    obs = env.reset()
    l = env.unwrapped.maze_size[0]
    reward = -0.1/(l*l)
    
    xs = [obs[0]]
    ys = [-obs[1]]
    done = False

    offset = np.array((random.random(), -random.random()))
    point = np.array((obs[0], -obs[1])) + offset

    while not done:
        # find current state and action
        s = m.m.predict(point.reshape(1, -1))
        a = int(m.pi[s])
        
        obs, reward, done, info = env.step(a)
        
        offset = np.array((random.random(), -random.random()))
        point = np.array((obs[0], -obs[1])) + offset
        xs.append(obs[0])
        ys.append(-obs[1])
    
    env.close()
    
    xs = np.array(xs)
    ys = np.array(ys)
    
    u = np.diff(xs)
    v = np.diff(ys)
    pos_x = xs[:-1] + u/2
    pos_y = ys[:-1] + v/2
    norm = np.sqrt(u**2+v**2) 
    
    fig, ax = plt.subplots()
    ax.plot(xs,ys, marker="o")
    ax.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy", zorder=5, pivot="mid")
    #ax.set_xlabel('FEATURE_%i' %f1)
    #ax.set_ylabel('FEATURE_%i' %f2)
    plt.ylim(-l+0.8, 0.2)
    plt.xlim(-.2, l-0.8)
    plt.show()
    # return xs, ys


# def policy_trajectory() takes a policy, a maze, and plots the optimal
# trajectory of the policy through the maze, for a total of n steps. 
# Can use with fitted_Q policy etc.
def policy_trajectory(policy, maze, n=50, rand=True):
    env = gym.make(maze)
    obs = env.reset()
    l = env.unwrapped.maze_size[0]
    reward = -0.1/(l*l)
    
    
    offset = np.array((random.random(), -random.random()))
    point = list(np.array((obs[0], -obs[1])) + offset)

    if rand:
        xs = [point[0]]
        ys = [point[1]]
    else:
        xs = [obs[0]]
        ys = [-obs[1]]
    
    done = False
    i = 0
    
    while not done:
        # find relevant action
        #print(point)
        a = policy.get_action(point)
        #print(a)
        obs, reward, done, info = env.step(a)
        
        offset = np.array((random.random(), -random.random()))
        point = list(np.array((obs[0], -obs[1])) + offset)
        
        if rand:
            xs.append(point[0])
            ys.append(point[1])
        else:
            xs.append(obs[0])
            ys.append(-obs[1])
        
        i += 1
        if i == n:
            done = True
    
    env.close()
    
    xs = np.array(xs)
    ys = np.array(ys)
    
    u = np.diff(xs)
    v = np.diff(ys)
    pos_x = xs[:-1] + u/2
    pos_y = ys[:-1] + v/2
    norm = np.sqrt(u**2+v**2) 
    
    fig, ax = plt.subplots()
    ax.plot(xs,ys, marker="o")
    ax.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy", zorder=5, pivot="mid")
    #ax.set_xlabel('FEATURE_%i' %f1)
    #ax.set_ylabel('FEATURE_%i' %f2)
    if rand:
        plt.ylim(-l-0.2, 0.2)
        plt.xlim(-.2, l+0.2)
    else:
        plt.ylim(-l+0.8, 0.2)
        plt.xlim(-.2, l-0.8)
    plt.show()
    return xs, ys
        
    


# policy_accuracy() takes a trained model and a maze, compares every line of 
# the original dataset to the real optimal policy and the model's optimal policy, 
# then returns the percentage correct from the model
def policy_accuracy(m, maze, df):
    if m.v is None:
        m.solve_MDP(gamma=1)
        
    # finding the true optimal: 
    P, R = get_maze_MDP(maze)
    true_v, true_pi = SolveMDP(P, R, 0.98, 1e-10, True, 'max')
    
    correct = 0
    # iterating through every line and comparing 
    for index, row in df.iterrows():
        # predicted action: 
        s = m.m.predict(np.array(row[2:2+m.pfeatures]).reshape(1,-1))
        #s = m.df_trained.iloc[index]['CLUSTER']
        a = m.pi[s]
        
        # real action:
        a_true = true_pi[row['OG_CLUSTER']]
        if a == a_true:
            correct += 1
    total = df.shape[0]
    return correct/total

# fitted_Q_policy_accuracy() takes a policy given by fitted_Q, the maze, 
# and the original dataframe, compares every line of the original dataset to
# the real optimal policy and the fitted_Q's optimal policy, then returns the
# percentage correct from fitted_Q 
def fitted_Q_policy_accuracy(policy, maze, df):
        
    # finding the true optimal: 
    P, R = get_maze_MDP(maze)
    true_v, true_pi = SolveMDP(P, R, 0.98, 1e-10, True, 'max')
    
    correct = 0
    # iterating through every line and comparing 
    for index, row in df.iterrows():
        # predicted action: 
        a = policy.get_action(list(row[2:4]))
        
        # real action:
        a_true = true_pi[row['OG_CLUSTER']]
        if a == a_true:
            correct += 1
    total = df.shape[0]
    return correct/total



# opt_maze_trajectory() takes a maze name, then solves the policy and plots the
# optimal path through the maze. Returns the path. ONLY WORKS for deterministic
# mazes!
def opt_maze_trajectory(maze):
    P, R = get_maze_MDP(maze)
    v, pi = SolveMDP(P, R, 0.98, 1e-10, True, 'max')
    
    env = gym.make(maze)
    obs = env.reset()
    l = env.unwrapped.maze_size[0]
    reward = -0.1/(l*l)
    
    xs = [obs[0]]
    ys = [-obs[1]]
    done = False
    
    while not done:
        # find current state and action
        ogc = int(obs[0] + obs[1]*l)
        #print(ogc)
        a = int(pi[ogc])
        
        obs, reward, done, info = env.step(a)
        #print(done)
        xs.append(obs[0])
        ys.append(-obs[1])
        
    env.close()
    
    xs = np.array(xs)
    ys = np.array(ys)
    
    u = np.diff(xs)
    v = np.diff(ys)
    pos_x = xs[:-1] + u/2
    pos_y = ys[:-1] + v/2
    norm = np.sqrt(u**2+v**2) 
    
    fig, ax = plt.subplots()
    ax.plot(xs,ys, marker="o")
    ax.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy", zorder=5, pivot="mid")
    #ax.set_xlabel('FEATURE_%i' %f1)
    #ax.set_ylabel('FEATURE_%i' %f2)
    plt.show()
    return xs, ys

# opp_action() returns the opposite action as input action
def opp_action(a):
    if a == 0:
        return 1
    elif a == 1:
        return 0
    elif a == 2:
        return 3
    elif a == 3:
        return 2
    
# get_maze_MDP() takes a maze string name, and returns two matrices, P and R, 
# which describe the MDP of the maze (includes a sink node)
def get_maze_MDP(maze):
    # initialize maze
    env = gym.make(maze)
    obs = env.reset()
    l = env.unwrapped.maze_size[0]
    
    # initialize matrices
    a = 4
    P = np.zeros((a, l*l+1, l*l+1))
    R = np.zeros((a, l*l+1))
    #P = np.zeros((a, l*l, l*l))
    #R = np.zeros((a, l*l))
    
    # store clusters seen and cluster/action pairs seen in set
    c_seen = set()
    ca_seen = set()
    
    # initialize env, set reward of original
    obs = env.reset()
    ogc = int(obs[0] + obs[1]*l)
    reward = -1/(l*l) # TODO: change to .1 or 1
    
    while len(ca_seen) < (4*l*l-4):
        # update rewards for new cluster
        if ogc not in c_seen:
            for i in range(a):
                #R[i, ogc] = reward
                R[i, ogc] = reward #*10 # temporary change to get -0.04 reward
            c_seen.add(ogc)
        
        stop = False
        for i in range(a):
            if (ogc, i) not in ca_seen:
                ca_seen.add((ogc, i))
                #print(len(ca_seen))
                new_obs, reward, done, info = env.step(i)
                ogc_new = int(new_obs[0] + new_obs[1]*l)
                # update probabilities
                P[i, ogc, ogc_new] = 1
                #print('updated', ogc, ogc_new, done)

                if not done:
                    if ogc != ogc_new:
                        P[opp_action(i), ogc_new, ogc] = 1
                        #print('updated', ogc_new, ogc)
                        ca_seen.add((ogc_new, opp_action(i)))
                    #print(len(ca_seen))
                ogc = ogc_new
                #print('new ogc', ogc, done)
                
                if done:
                    # set next state to sink
                    for i in range(a):
                        P[i, ogc_new, l*l] = 1
                        P[i, l*l, l*l] = 1
                        R[i, l*l] = 0
                        R[i, ogc_new] = 1
                    obs = env.reset()
                    ogc = int(obs[0] + obs[1]*l)
                
                stop = True
            if stop:
                break
        
        # if all seen already, take random step 
        if not stop:
            action = env.action_space.sample()
            new_obs, reward, done, info = env.step(action)
            ogc = int(new_obs[0] + new_obs[1]*l)
            #print('trying random action', ogc)
            if done:
                obs = env.reset()
                ogc = int(obs[0] + obs[1]*l)
    env.close()
    
    return P, R


# get_maze_transition_reward() takes a maze name, and returns the transition function
# in the form of f(x, u) = x'. State, action, gives next state. Takes into account
# sink node, and stays there. Also returns reward function r(x) that takes a state
# and returns the reward
def get_maze_transition_reward(maze):
    
    P, R = get_maze_MDP(maze)
    l = int((R.size/4-1)**0.5)
    
    def f(x, u, plot=True):
        #print('x', x, 'u', u)
        
        # if sink, return None again
        if plot:
            if x[0] == None:
                return (None, None)
        
        # if no action, or an action '4' to simulate no action: 
        if u == 'None' or u == 4:
            if plot:
                return (None, None)
            else:
                return (0, -l)
        
        # first define the cluster of the maze based on position
        x_orig = (int(x[0]), int(-x[1]))
        offset = np.array((random.random(), -random.random()))
    
        
        c = int(x_orig[0] + x_orig[1]*l)
        c_new = P[u, c].argmax()
        
        
        # if maze at sink, return None
        if c_new == R.size/4-1:
            if plot:
                return (None, None)
            else:
                return (0, -l)
        else:
            x_new = (c_new%l, c_new//l)
            x_new = (x_new[0], -x_new[1])
        return x_new + offset
        
        
    def r(x):
        # if sink, return 0 reward
        if x[0] == None:
            return R[0][-1]
        else:
            x_orig = (int(x[0]), int(-x[1]))
            c = int(x_orig[0] + x_orig[1]*l)
            #print(c)
            return R[0][c]
    
    return f, r
    


# plot_paths() takes a dataframe with 'FEATURE_1' and 'FEATURE_2', and plots
# the first n paths (by ID). Returns nothing
def plot_paths(df, n): 
    fig, ax = plt.subplots()
    
    for i in range(n):
        x = df.loc[df['ID']==i]['FEATURE_0']
        y = df.loc[df['ID']==i]['FEATURE_1']
        xs = np.array(x)
        ys = np.array(y)
        
        u = np.diff(xs)
        v = np.diff(ys)
        pos_x = xs[:-1] + u/2
        pos_y = ys[:-1] + v/2
        norm = np.sqrt(u**2+v**2) 
        
        ax.plot(xs,ys, marker="o")
        ax.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy", zorder=5, pivot="mid")
    
    plt.show()
    return



# fitted_Q() trains K functions Q1 to QK that determine the optimal strategy
# x_df is a dataframe of the form ['ID', 'TIME', features, 'ACTION', 'RISK']
# for each one-step transition returns the last iteration QK, and a function \
# policy that takes a state and outputs the optimal action
def fitted_Q(K, # number of iterations
             x_df, # dataframe 
             gamma, # decay factor
             pfeatures, # number of features in dataframe
             actions, # list of action possibilities
             reward_c, # reward function
             take_max = True, # True if max_cost is good, otherwise false
             regression = 'LinearRegression', # str: type of regression
             one_hot = False, # one-hot encode the given actions
             ):
    
    x_df = x_df.copy(deep=True)
    #x_df = x_df.loc[x_df['']]
    # initialize storage and actions
    #Qs = []
    values = []
    num_actions = len(actions)
    
    # create the first Q1 function
    class Q:
        def predict(self, array): 
            x = array[0][:pfeatures]
            #print('x', x)
            return reward_c(x)
    Q_new = Q()
    #Qs.append(Q_new)

    
    # calculate the x_t2 next step for each x
    x_df['x_t2'] = x_df.iloc[:, 2:2+pfeatures].values.tolist()
    x_df['x_t2'] = x_df['x_t2'].shift(-1)
    
    
    # delete the ones that don't finish
    x_df.loc[(x_df['ID']!=x_df['ID'].shift(-1))&(x_df['ACTION']!='None'), 'ACTION'] = 'replace'
    x_df = x_df[x_df['ACTION']!='replace']
    
    # deal with last x_t2 if it happens to finish, with empty list
    x_df.loc[x_df['x_t2'].isnull(), ['x_t2']] = \
        x_df.loc[x_df['x_t2'].isnull(), 'x_t2'].apply(lambda x: [])
    
    if one_hot:
        actions_new = [[int(x==i) for x in range(num_actions)] for i in range(num_actions)]
        
    else:
        actions_new = [[a] for a in actions]
        
    
    for i in range(num_actions):
        # regular action names
        x_df['a%s'%i] = x_df.apply(lambda x: x.x_t2+actions_new[i], axis=1)
        
    action_names = ['a%s'%i for i in range(num_actions)]
    
    print(x_df, flush=True)
    # create X using x_t and u
    # select (x_t, u) pair as training
    # setting 'None' action as action -1
    
    if one_hot:
        X = x_df.iloc[:, 2:2+pfeatures]
        one_hot_actions = pd.get_dummies(x_df['ACTION'])[actions]
        # represent 'None' as action 0 in the training data so it gets predicted
        try:
            one_hot_actions.loc[one_hot_actions['None']==1, 0] = 1
        except:
            pass
        one_hot_actions = one_hot_actions[actions]
        X = pd.concat([X, one_hot_actions], axis=1)
    else:
        X = x_df.iloc[:, 2:3+pfeatures]
        X.loc[X['ACTION']=='None', 'ACTION'] = 0
    
    
    # maybe trying to put action as a tuple of (0, 1) or 1-hot to help it learn better...?
    #X = x_df[['FEATURE_0', 'FEATURE_1']].merge(x_df['ACTION'].apply(pd.Series), \
                #left_index = True, right_index = True) 
    
    print('New training features', flush=True)
    print(X, flush=True)
    
    # update function that returns the correct updated Q for the given row
    # and the Q_new prediction function. Takes just the risk (with no extra update)
    # for ending state, which is when ACTION = 'None'
    def update(row, Q_new, take_max):
        #print('action:', row.ACTION)
        if row.ACTION == 'None':
            #print('returning this')
            return np.array([row.RISK])
        if take_max:
            return row.RISK + gamma*max([Q_new.predict([g]) \
                                for g in [row[a] for a in action_names]])
        else:
            return row.RISK + gamma*min([Q_new.predict([g]) \
                                for g in [row[a] for a in action_names]])
            
    
    
    bar = tqdm(range(K))
    #bar = range(K)
    #creating new Qk functions
    for i in bar:
        # create y using Qk-1 and x_t2
        # non-DP
        '''
        if take_max: 
            y = x_df.apply(lambda x: x.RISK + gamma*max([Q_new.predict([g]) \
                                for g in [x[a] for a in action_names]]), axis=1)
        else:
            y = x_df.apply(lambda x: x.RISK + gamma*min([Q_new.predict([g]) \
                                for g in [x[a] for a in action_names]]), axis=1)
        '''
        y = x_df.apply(lambda x: update(x, Q_new, take_max), axis=1)
        
        '''                               
        # initialize dp
        memo = {}
        mu = 0
        y = []                        
        for index, row in x_df.iterrows():
            qs = []
            for f in [row['a0'], row['a1'], row['a2'], row['a3']]:
                if f in memo:
                    qs.append(memo[f])
                    #print('memo used')
                    mu += 1
                else:
                    q = Q_new.predict([f])
                    memo[f] = q
                    qs.append(memo[f])
            y.append(row['c'] + gamma*min(qs))
        '''
        
        y = np.array(y)
        #print(y, flush=True)
        #print(np.unique(y), flush=True)
        #print(y)
        
        # train the actual Regression function as Qk
        #regr = MLPRegressor(random_state=1, max_iter=500).fit(X, y)
        if regression ==  'LinearRegression':
            regr = LinearRegression().fit(X, y)
        if regression == 'RandomForest':
            regr = RandomForestRegressor(random_state=0, oob_score=True).fit(X, y)
        if regression == 'ExtraTrees':
            regr = ExtraTreesRegressor(n_estimators=50).fit(X,y.ravel())
        if regression == 'SGDRegressor':
            regr = SGDRegressor().fit(X, y.ravel())
        if regression == 'DecisionTrees':
            regr = DecisionTreeRegressor(random_state=5).fit(X, y.ravel())
        #Qs.append(regr)
        Q_new = regr
        #print('memo size', len(memo), 'used', mu, flush=True)
        
        # calculate and store value of new policy
        p = policy(actions_new, take_max)
        p.fit(Q_new)
        v = p.get_value([0.5, -0.5]) # get value of starting state
        a = p.get_action([0.5, -0.5])
        values.append(v)
        
        print('Iteration:', i, 'Start Value', v, 'action', a)
        
    
    #QK = Qs[-1]
    QK = Q_new
    
    p = policy(actions_new, take_max)
    p.fit(QK)
        
    return QK, p, x_df, values


class policy:
    def __init__(self, actions, take_max):
        self.QK = None
        self.actions = actions
        self.take_max = take_max
        
    def fit(self, 
            QK): # model, the latest fitted_Q 
        self.QK = QK
    
    # get_action() takes a state x and predicts the optimal action
    def get_action(self,
             x): 
        if self.take_max:
            i = np.argmax([self.QK.predict([x + u]) \
                                        for u in self.actions])
            #print([self.QK.predict([x + [u]]) \
                                        #for u in self.actions])
        else:
            i = np.argmin([self.QK.predict([x + u]) \
                                            for u in self.actions])
        
        if len(self.actions[i]) == 1:
            return self.actions[i]
        else: return np.argmax(self.actions[i])
    
    # TODO! Update to be compatible with np.arrays (use concat)
    # get_value() takes a state x and returns the max value from fitted Q
    def get_value(self, x):
        if self.take_max:
            val = max([self.QK.predict([x + u]) \
                                        for u in self.actions])
        else:
            val = min([self.QK.predict([x + u]) \
                                        for u in self.actions])
        return val

    
# value_diff() takes a list of models or policies, a list
# of data sizes N from which models/policies were trained (same length as list 
# of models/policies), parameter K number of simulations from starting cell, 
# the real P, R transition matrices, and f, r model transition function and rewards. 
# calculates optimality gap difference between values: |v_policy/algo - v_opt*|. 
# v_policy/algo is found by randomly generating K points in the starting cell, 
# simulating over t_max steps through the actual maze, and taking the avg
# over these K trials. v_opt is the value taking the optimal policy
# If toggle policy = True, calculates the optimality gap of a list of policies
# fed in as the first parameter in the 'models' input. 
def value_diff(models, Ns, K, t_max, P, R, f, r, true_v = None, true_pi = None, policy=False, adaptive=False): 
    # first calculate v_opt for this particular maze and t_max steps
    v_opt = 0
    s = 0
    if true_v is None or true_pi is None:
        true_v, true_pi = SolveMDP(P, R, prob='max', gamma=1, epsilon=1e-8)
    for t in range(t_max):
        v_opt += R[0, s]
        #print(R[0, s], v_opt)
        a = true_pi[s]
        s_new = P[a, s].argmax()
        s = s_new
        #print(s)
    
    # then for each model or policy, run through the K trials and return
    # an array of differences corresponding to each N 
    n = len(Ns)
    
    v_alg = []
    
    # for each n of this set:
    for i in range(n):
        m = models[i]
        #p = policies[i]
        
        # calculating average value for this model and policy
        if not policy: 
            if not adaptive: 
                if m.pi is None:
                    m.solve_MDP(gamma=1, epsilon=1e-4)
        
        model_vs = []
        

        for k in range(K):
            vm_estim = 0 # initializing model value estimate

            x_model = np.array((random.random(), -random.random()))
            
            # estimate value for model or policy
            vm_estim += r(x_model)
            
            for t in range(t_max):
                # predict action and upate value for model
                if x_model[0] == None:
                    a = 0
                else:
                    if policy: 
                        a = m.get_action(list(x_model))
                    elif adaptive: 
                        a = m.pick_action(np.array([x_model[0]/5, -x_model[1]/5]), t)
                        # translating model output to action
                        action = a[1]
                        real_action = 3
                        if action < 0.25: 
                            real_action = 0
                        elif action < 0.5: 
                            real_action = 1
                        elif action < 0.75: 
                            real_action = 2
                        a = real_action

                    else:
                        s = int(m.m.predict([x_model]))
                        a = m.pi[s]
                x_model_new = f(x_model, a)
                # print('v estim: ', vm_estim)
                # print('x: ', x_model_new)
                # print('----')
                vm_estim += r(x_model_new)
                #print('new model state', x_model_new, 'reward', r(x_model_new))
                x_model = x_model_new
                
            
            # append the total value from this trial
            # print(vm_estim, v_opt)
            vm_diff = abs(vm_estim - v_opt)
            model_vs.append(vm_diff)

        
        # average values of all trials for this model/policy
        model_v = np.mean(model_vs)
        v_alg.append(model_v)
    
    
    return v_alg




# value_est() takes a list of models or policies, integer K number of simulations, 
# real transition matricies P, R and transition/reward functions f and r
# and compares the v_opt (correctly solved MDP value from cell 0) and v_alg 
# (value determined by model/policy) for K number of random points from the same 
# start cell. Result truncates everything greater than 1 to 1. 

def value_est(models, Ns, K, P, R, f, r, true_v=None, true_pi=None, policy=False):
    # first calculate v_opt for this particular maze and t_max steps
    v_opt = 0
    s = 0
    if true_v is None or true_pi is None:
        true_v, true_pi = SolveMDP(P, R, prob='max', gamma=1, epsilon=1e-8)
    #print(true_v)
    v_opt = true_v[0]
    
    # then for each model and policy, run through the K trials and return
    # an array of differences corresponding to each N 
    n = len(Ns)
    
    v_alg = []
    # for each n of this set:
    for i in range(n):
        #print('Round N=', Ns[i])
        m = models[i]
        
        # calculating average value for this model and policy
        if not policy:
            #print('resolved model')
            if m.pi is None: # new edit
                m.solve_MDP(gamma=1, epsilon=1e-4)
        #m.solve_MDP(gamma=1)
        
        model_vs = []
        for k in range(K):
            start = np.array((random.random(), -random.random()))
            if policy:
                val = m.get_value(list(start))
            else:
                s = m.m.predict(start.reshape(1, -1))
                val = m.v[s]
            #print('val', val, 'v_opt', v_opt, 's', s)
            v_diff = abs(val - v_opt)
            val_trunc = min(1, v_diff) # truncate and take 1 as highest

            if isinstance(val_trunc, np.ndarray):
                val_trunc = val_trunc[0]
            #print(val_trunc)
            model_vs.append(val_trunc)

        model_v = np.mean(model_vs)
        #print('model avg val', model_v)
        v_alg.append(model_v)
    
    # calculate differences between values and optimal
    #v_alg_diff = abs(v_alg - v_opt)
    return v_alg



# opt_path_value_diff() compares the v_opt from the optimal sequence of actions
# to the value derived from the MDP also taking this exact sequence of actions
# averaged over K trials. Result truncates anything greater than 1 to 1. 
# Takes a list of models/policies, list of data sizes used to train models Ns, 
# int K number of simulations, real transition matrices P, R and real transition
# and reward functions f, r from get_maze_transition_reward
# Returns list of differences for each model
def opt_path_value_diff(models, Ns, K, t_max, P, R, f, r, true_v = None, true_pi = None): 
    # first calculate v_opt for this particular maze and t_max steps
    v_opt = 0
    s = 0
    actions = []
    if true_v is None or true_pi is None:
        true_v, true_pi = SolveMDP(P, R, prob='max', gamma=1, epsilon=1e-8)
    for t in range(t_max):
        v_opt += R[0, s]
        #print(R[0, s], v_opt)
        a = true_pi[s]
        actions.append(a)
        s_new = P[a, s].argmax()
        s = s_new
        #print(s)
    
    # then for each model or policy, run through the K trials and return
    # an array of differences corresponding to each N 
    n = len(Ns)
    
    v_alg = []

    # for each n of this set:
    for i in range(n):
        #print('Round N=', Ns[i])
        m = models[i]
        
        # calculating average value for this model
        if m.pi is None:
            #print('resolved model')
            m.solve_MDP(gamma=1, epsilon=1e-4, alpha = 1, min_action_obs = 0, min_action_purity=-1)
        #m.solve_MDP(gamma=0.999)
        
        model_vs = []
      
        
        for k in range(K):
            vm_estim = 0 # initializing model value estimate
            
            start = np.array((random.random(), -random.random()))
            
            # estimate value for model / policy
            
            s = m.m.predict(start.reshape(1, -1))
            #s_prime = s
            vm_estim += m.R[0, s]
            
            # take the list of actions through MDP and update value
            for a in actions:
                
                s = m.P[a,s].argmax()
                vm_estim += m.R[0, s]
                
            
            # append the total value from this trial
            vm_diff = abs(vm_estim - v_opt)
            val_trunc = min([1], vm_diff)
            model_vs.append(val_trunc)
            
        
        # average values of all trials for this model/policy
        model_v = np.mean(model_vs)
        v_alg.append(model_v)
        
        
    return v_alg
    

        
# Initialize the "maze" environment
# =============================================================================
# =============================================================================
# env = gym.make(mazes[4])
# obs = []
# 
# # first point for ID
# observation = env.reset()
# offset = np.array((random.random(), random.random()))
# # #env.state = offset
# # #print(env.state)
# obs.append(observation+offset)
# 
# for _ in range(1000):
#     
#     env.render()
#     action = env.action_space.sample() # your agent here (this takes random actions)
#     observation, reward, done, info = env.step(action)
#     print(observation, reward, done, info)
#     obs.append(observation+offset)
#       
#     if done:
#       observation = env.reset()
# env.close()
# =============================================================================

# =============================================================================
# Load models
# s = 5
# Ns = [10, 20, 30, 40, 50, 70, 90, 110, 130, 150, 170, 200]
# models = []
# for n in Ns:
#     m = pickle.load(open(f'round_{s}_model_N={n}.sav', 'rb'))
#     models.append(m)
