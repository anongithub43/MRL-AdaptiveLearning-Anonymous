import or_suite
import numpy as np

import copy

import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import pandas as pd

import gym

# Getting out configuration parameter for the environment
CONFIG =  or_suite.envs.env_configs.maze_default_config

path = '/Users/lowell/MRL/ORSuite/'


# Specifying training iteration, epLen, number of episodes, and number of iterations
epLen = CONFIG['epLen']
nEps = 500
numIters = 1
alpha=CONFIG['alpha']

# can add more scalings (exploration parameters) here
scaling_list = [1]


# Configuration parameters for running the experiment
DEFAULT_SETTINGS = {'seed': 1, 
                    'recFreq': 1, 
                    'dirPath': '../data/ambulance/', 
                    'deBug': True, 
                    'nEps': nEps, 
                    'numIters': numIters, 
                    'saveTrajectory': True, # save trajectory for calculating additional metrics
                    'epLen' : CONFIG['epLen'],
                    'render': False,
                    'pickle': False # indicator for pickling final information
                    }

maze_env = gym.make('Maze-v0', config=CONFIG)
mon_env = Monitor(maze_env)

state_space = maze_env.observation_space
action_space = maze_env.action_space

agents = {
    
'AdaMB': or_suite.agents.rl.ada_mb.AdaptiveDiscretizationMB(epLen, 0, 0, 2, True, False, 2, 1),
'AdaQL' : or_suite.agents.rl.ada_ql.AdaptiveDiscretizationQL(epLen, scaling_list[0], True, 2)

}

# code here repurposed from other experiments, the main portion is just 
# the 'run_single_algo' functions that we need to run. 
path_list_line = []
algo_list_line = []
path_list_radar = []
algo_list_radar= []
for agent in agents:
    print(agent)
    DEFAULT_SETTINGS['dirPath'] = path + '/data/maze_'+str(agent)+'_'+str(alpha)+'/'
    if agent == 'SB PPO':
        or_suite.utils.run_single_sb_algo(mon_env, agents[agent], DEFAULT_SETTINGS)
    elif agent == 'AdaMB':
        or_suite.utils.run_single_algo_tune(maze_env, agents[agent], scaling_list, DEFAULT_SETTINGS)
    elif agent == 'AdaQL':
        or_suite.utils.run_single_algo_tune(maze_env, agents[agent], scaling_list, DEFAULT_SETTINGS)
    elif agent == 'DiscreteQL' or agent == 'Unif QL' or agent == 'DiscreteMB' or agent == 'Unif MB'or agent == 'DiscreteMB':
        or_suite.utils.run_single_algo_tune(maze_env, agents[agent], scaling_list, DEFAULT_SETTINGS)
    else:
        or_suite.utils.run_single_algo(maze_env, agents[agent], DEFAULT_SETTINGS)

    path_list_line.append(path + '/data/maze_2_'+str(agent)+'_'+str(alpha))
    algo_list_line.append(str(agent))
    if agent != 'SB PPO':
        path_list_radar.append(path + '/data/maze_2_'+str(agent)+'_'+str(alpha))
        algo_list_radar.append(str(agent))

