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
CONFIG =  or_suite.envs.env_configs.ambulance_graph_default_config


# Specifying training iteration, epLen, number of episodes, and number of iterations
epLen = CONFIG['epLen']
nEps = 10
numIters = 20


scaling_list = [0.01, .1, 1, 10]


# Configuration parameters for running the experiment
DEFAULT_SETTINGS = {'seed': 1, 
                    'recFreq': 1, 
                    'dirPath': '../data/ambulance/', 
                    'deBug': False, 
                    'nEps': nEps, 
                    'numIters': numIters, 
                    'saveTrajectory': True, # save trajectory for calculating additional metrics
                    'epLen' : 5,
                    'render': False,
                    'pickle': False # indicator for pickling final information
                    }


alpha = CONFIG['alpha']
num_ambulance = CONFIG['num_ambulance']

ambulance_env = gym.make('Ambulance-v1', config=CONFIG)
mon_env = Monitor(ambulance_env)

state_space = ambulance_env.observation_space
action_space = ambulance_env.action_space
# #print(state_space)
# #print(action_space)
#print(state_space.nvec)
# test = [2]
# print(np.append(test, (state_space.nvec,action_space.nvec)))

agents = { #action_space, state_space, epLen, scaling, alpha, flag
'Stable': or_suite.agents.ambulance.stable.stableAgent(CONFIG['epLen']),
'DiscreteMB': or_suite.agents.rl.discrete_mb.DiscreteMB(action_space, state_space, epLen, scaling_list[0], 0, False),
'DiscreteQL' : or_suite.agents.rl.discrete_ql.DiscreteQl(action_space, state_space, epLen, scaling_list[0])}

path_list_line = []
algo_list_line = []
path_list_radar = []
algo_list_radar= []
for agent in agents:
    print(agent)
    DEFAULT_SETTINGS['dirPath'] = '../data/ambulance_metric_'+str(agent)+'_'+str(num_ambulance)+'_'+str(alpha)+'/'
    if agent == 'SB PPO':
        or_suite.utils.run_single_sb_algo(mon_env, agents[agent], DEFAULT_SETTINGS)
    elif agent == 'DiscreteQL':
        or_suite.utils.run_single_algo_tune(ambulance_env,agents[agent], scaling_list, DEFAULT_SETTINGS)
    elif agent == 'DiscreteQL':
        or_suite.utils.run_single_algo_tune(ambulance_env,agents[agent], scaling_list, DEFAULT_SETTINGS)
    elif agent == 'AdaQL' or agent == 'Unif QL' or agent == 'AdaMB' or agent == 'Unif MB'or agent == 'DiscreteMB':
        or_suite.utils.run_single_algo_tune(ambulance_env, agents[agent], scaling_list, DEFAULT_SETTINGS)
    else:
        or_suite.utils.run_single_algo(ambulance_env, agents[agent], DEFAULT_SETTINGS)

    path_list_line.append('../data/ambulance_metric_'+str(agent)+'_'+str(num_ambulance)+'_'+str(alpha))
    algo_list_line.append(str(agent))
    if agent != 'SB PPO':
        path_list_radar.append('../data/ambulance_metric_'+str(agent)+'_'+str(num_ambulance)+'_'+str(alpha))
        algo_list_radar.append(str(agent))
fig_path = '../figures/'
fig_name = 'ambulance_metric'+'_'+str(num_ambulance)+'_'+str(alpha)+'_line_plot'+'.pdf'
or_suite.plots.plot_line_plots(path_list_line, algo_list_line, fig_path, fig_name, int(nEps / 40)+1)

additional_metric = {}
fig_name = 'ambulance_metric'+'_'+str(num_ambulance)+'_'+str(alpha)+'_radar_plot'+'.pdf'
or_suite.plots.plot_radar_plots(path_list_radar, algo_list_radar,
fig_path, fig_name,
additional_metric
)

