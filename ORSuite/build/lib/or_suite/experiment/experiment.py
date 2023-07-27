import time
from shutil import copyfile
import pandas as pd
import tracemalloc
import numpy as np
import pickle
import os

import sys
sys.path.append('/Users/lowell/MRL/Algorithm')
sys.path.append('/Users/lowell/MRL/Maze')
from MRL_model import MRL_model
from maze_functions import value_diff, get_maze_MDP, get_maze_transition_reward
from MDPtools import SolveMDP

class Experiment(object):
    """Optional instrumentation for running an experiment.

    Runs a simulation between an arbitrary openAI Gym environment and an algorithm, saving a dataset of (reward, time, space) complexity across each episode,
    and optionally saves trajectory information.

    Attributes:
        seed: random seed set to allow reproducibility
        dirPath: (string) location to store the data files
        nEps: (int) number of episodes for the simulation
        deBug: (bool) boolean, when set to true causes the algorithm to print information to the command line
        env: (openAI env) the environment to run the simulations on
        epLen: (int) the length of each episode
        numIters: (int) the number of iterations of (nEps, epLen) pairs to iterate over with the environment
        save_trajectory: (bool) boolean, when set to true saves the entire trajectory information
        render_flag: (bool) boolean, when set to true renders the simulations
        agent: (or_suite.agent.Agent) an algorithm to run the experiments with
        data: (np.array) an array saving the metrics along the sample paths (rewards, time, space)
        trajectory_data: (list) a list saving the trajectory information
    """

    def __init__(self, env, agent, dict):
        '''
        Args:
            env: (openAI env) the environment to run the simulations on
            agent: (or_suite.agent.Agent) an algorithm to run the experiments with
            dict: a dictionary containing the arguments to send for the experiment, including:
                dirPath: (string) location to store the data files
                nEps: (int) number of episodes for the simulation
                deBug: (bool) boolean, when set to true causes the algorithm to print information to the command line
                env: (openAI env) the environment to run the simulations on
                epLen: (int) the length of each episode
                numIters: (int) the number of iterations of (nEps, epLen) pairs to iterate over with the environment
                save_trajectory: (bool) boolean, when set to true saves the entire trajectory information
                render: (bool) boolean, when set to true renders the simulations
                pickle: (bool) when set to true saves data to a pickle file
        '''

        self.seed = dict['seed']
        self.dirPath = dict['dirPath']
        self.deBug = dict['deBug']
        self.nEps = dict['nEps']
        self.env = env
        self.epLen = dict['epLen']
        self.num_iters = dict['numIters']
        self.save_trajectory = dict['saveTrajectory']
        self.render_flag = dict['render']
        self.agent = agent
        # initializes the dataset to save the information
        self.data = np.zeros([dict['nEps']*self.num_iters, 5])
        self.pickle = dict['pickle']

        if self.save_trajectory:  # initializes the list to save the trajectory
            self.trajectory = []

        np.random.seed(self.seed)  # sets seed for experiment

    # Runs the experiment
    def run(self):
        '''
            Runs the simulations between an environment and an algorithm
        '''

        # print('**************************************************')
        # print('Running experiment')
        # print('**************************************************')

        index = 0

        for i in range(self.num_iters):  # loops over the numer of iterations
            self.agent.reset()  # resets algorithm, updates based on environment's configuration
            # updates agent configuration based on environment
            self.agent.update_config(self.env, self.env.get_config())

            mrl_flag = False
            mrl_model = MRL_model()

            # MRL parameters
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
            # MRL dataframe

            df = pd.DataFrame({}, columns=['ID', 'TIME', 'FEATURE_0', 'FEATURE_1', 'ACTION', 'RISK'])

            for ep in range(0, self.nEps):  # loops over the episodes
                if self.deBug:
                    print('Episode : %s' % (ep))

                # Reset the environment
                self.env.reset()

                if self.render_flag:  # optionally renders the environments
                    self.env.render()

                oldState = self.env.state  # obtains old state
                epReward = 0
                epGap = 0
                t = 0 # keeping track of time for mrl 
                reward = -0.004

                # updates agent policy based on episode
                self.agent.update_policy(ep)

                done = False
                h = 0

                start_time = time.time()  # starts time and memory tracker
                tracemalloc.start()

                # repeats until episode is finished
                while (not done) and h < self.epLen:
                    # Step through the episode
                    if self.deBug:
                        print('state : %s' % (oldState))
                    action = self.agent.pick_action(
                        oldState, h)  # algorithm picks a state
                    if self.deBug:
                        print('action : %s' % (action))

                    real_action = 3

                    if action[0] < 0.25: 
                        real_action = 0
                    elif action[0] < 0.5: 
                        real_action = 1
                    elif action[0] < 0.75: 
                        real_action = 2

                    df = df.append({'ID': ep, 
                                    'TIME': t, 
                                    'FEATURE_0': oldState[0] * 5, # because the maze is normally [0,4]
                                    'FEATURE_1': oldState[1] * 5, 
                                    'ACTION': int(real_action), 
                                    'RISK': reward * 10}, ignore_index=True) # reward * 10 to force 0.04

                    # steps based on the action chosen by the algorithm
                    newState, reward, done, info = self.env.step(action)
                    epReward += reward

                    if reward == 1: 
                        mrl_flag = 1

                    if self.deBug:
                        print('new state: %s' % (newState))
                        print('reward: %s' % (reward))
                        print('epReward so far: %s' % (epReward))
                        print(f'Info: {info}')

                    self.agent.update_obs(
                        oldState, action, reward, newState, h, info)

                    if self.save_trajectory:  # saves trajectory step if desired
                        record = {'iter': i,
                                  'episode': ep,
                                  'step': h,
                                  'oldState': oldState,
                                  'action': action,
                                  'reward': reward,
                                  'newState': newState,
                                  'info': info}

                        self.trajectory.append(record)
                    
                    oldState = newState
                    if self.render_flag:  # optionally renders the environment
                        self.env.render()
                    
                    h = h + 1
                    t = t + 1

                    if done:
                        df = df.append({'ID': ep,
                                        'TIME': t + 1,
                                        'FEATURE_0': oldState[0] * 5, 
                                        'FEATURE_1': oldState[1] * 5, 
                                        'ACTION': 'None', 
                                        'RISK': reward}, ignore_index=True)
                        
                if mrl_flag:
                    mrl_model.fit(df, # df: dataframe in the format ['ID', 'TIME', ...features..., 'RISK', 'ACTION']
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
                    
                    def augment_reward(R_array): 
                        '''
                        forces -0.04 reward instead of -0.004 reward for 5x5 maze example
                        Loophole to repro paper results that bypasses having to make deep repo changes
                        '''
                        new_R = np.zeros(R_array.shape[1:])
                        for i, reward in enumerate(R_array[0]): 
                            if reward < 0 and reward > -1: # don't touch end, sink, and punishment rewards
                                new_R[i] = 10*reward
                            else: 
                                new_R[i] = reward
                        new_R = np.repeat(np.array([new_R]), R_array.shape[0], axis=0)
                        return new_R
                    
                    # get mrl optimality gap 
                    mrl_model.solve_MDP(gamma=1, epsilon=1e-4)

                    maze = 'maze-sample-5x5-v0'
                    T_max = 25
                    K = 100
    
                    P, R = get_maze_MDP(maze)
                    f, rw = get_maze_transition_reward(maze)
                    true_v, true_pi = SolveMDP(P, augment_reward(R), prob='max', gamma=1, epsilon=1e-8)

                    opt_gap = value_diff([mrl_model], [0], K, T_max, P, R, f, rw, true_v, true_pi) 

                    epGap = epGap + opt_gap[0] / 0.04 # normalize by reward of each cell

                current, _ = tracemalloc.get_traced_memory()  # collects memory / time usage
                tracemalloc.stop()
                end_time = time.time()

                if self.deBug:
                    print('final state: %s' % (newState))

                # Logging to dataframe
                self.data[index, 0] = ep
                self.data[index, 1] = i
                self.data[index, 2] = epReward
                self.data[index, 3] = current
                self.data[index, 4] = epGap #np.log(((end_time) - (start_time)))

                index += 1


            self.env.close()

        # print('**************************************************')
        # print('Experiment complete')
        # print('**************************************************')

    # Saves the data to the file location provided to the algorithm
    def save_data(self):
        '''
            Saves the acquired dataset to the noted location

            Returns:
                dataframe corresponding to the saved data
        '''

        # print('**************************************************')
        # print('Saving data')
        # print('**************************************************')

        if self.deBug:
            print(self.data)

        dir_path = self.dirPath

        data_loc = 'data.csv'
        traj_loc = 'trajectory.obj'
        agent_loc = 'agent.obj'

        data_filename = os.path.join(dir_path, data_loc)
        traj_filename = os.path.join(dir_path, traj_loc)
        agent_filename = os.path.join(dir_path, agent_loc)

        dt = pd.DataFrame(self.data, columns=[
                          'episode', 'iteration', 'epReward', 'memory', 'time'])
        dt = dt[(dt.T != 0).any()]

        print('Writing to file ' + data_loc)

        if os.path.exists(dir_path):
            # saves the collected dataset
            dt.to_csv(data_filename, index=False,
                      float_format='%.5f', mode='w')
            if self.save_trajectory:  # saves trajectory to filename
                outfile = open(traj_filename, 'wb')
                pickle.dump(self.trajectory, outfile)
                outfile.close()
        else:  # same as before, but first makes the directory
            os.makedirs(dir_path)
            dt.to_csv(data_filename, index=False,
                      float_format='%.5f', mode='w')
            if self.save_trajectory:  # saves trajectory to filename
                outfile = open(traj_filename, 'wb')
                pickle.dump(self.trajectory, outfile)
                outfile.close()

        if self.pickle:

            if hasattr(self.agent, 'tree_list'):
                outfile = open(agent_filename, 'wb')
                pickle.dump(self.agent.tree_list, outfile)
                outfile.close()
            elif hasattr(self.agent, 'qVals'):
                outfile = open(agent_filename, 'wb')
                pickle.dump(self.agent.qVals, outfile)
                outfile.close()

        # print('**************************************************')
        # print('Data save complete')
        # print('**************************************************')

        return dt
