import numpy as np
import gym
from gym import spaces
import gym_maze
import math
import random

from or_suite.envs import env_configs

'''Implementation of the Maze environment for MRL'''
class MazeEnvironment(gym.Env): 
    def __init__(self, config=env_configs.maze_default_config):
        '''
            env - AI Gym Environment
            epLen - Number of steps per episode
        '''
        self.env = gym.make('maze-sample-5x5-v0')
        self.epLen = config['epLen']
        self.timestep = 0
        self.state = np.array((random.random(), random.random())) / 5
        self.observation_space = self.env.observation_space
        # actions are continuous but become discretized in the advance function
        self.action_space = spaces.Box(low=0, high=1,
                                       shape=(1,), dtype=np.float32) 
        self.config = config
        self.last_action = None

    def get_epLen(self):
        return self.epLen

    def reset(self):
        '''Reset the environment'''
        self.timestep = 0
        self.state = self.env.reset() + np.array((random.random(), random.random())) / 5
    
    def get_config(self):
        return self.config

    def step(self, action):
        '''
        Move one step in the environment

        Args:
        action - (tuple(float))
        Returns:
            reward - double - reward
            newState - int - new state
            pContinue - 0/1 - flag for end of the episode
        '''

        # we need to convert the action (tuple of action 1, action 2) into a single
        # action N, S, W, E = 0,1,2,3 so take the biggest direction of action. 
        # feature 0 = x coordinate, feature 1 = y coordinate

        action = action[0]
        real_action = 3

        if action < 0.25: 
            real_action = 0
        elif action < 0.5: 
            real_action = 1
        elif action < 0.75: 
            real_action = 2

        self.last_action = real_action

        offset = np.array((random.random(), random.random()))

        newState, reward, terminal, info = self.env.step(real_action)

        print('------------')
        print('state: ', self.state)
        print('action: ', action)
        print('taken action: ', real_action)
        print('next state int: ', newState)


        newState = (newState + offset) / 5

        
        print('new state: ', newState)
        print('reward: ', reward)

        self.state = newState

        print('next state: ', self.state)

        if self.timestep == self.epLen or terminal:
            done = True
            self.reset()
        else:
            done = False
        
        self.timestep += 1

        return newState, reward, done, info