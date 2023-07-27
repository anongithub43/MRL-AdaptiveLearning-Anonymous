import gym
import numpy as np
import sys
sys.path.append('/Users/lowell/MRL/ORSuite/or_suite/envs/maze/')
from scipy.stats import poisson

from or_suite.envs import env_configs
import pytest
from stable_baselines3.common.env_checker import check_env

CONFIG = env_configs.maze_default_config

env = gym.make('Maze-v0', config=CONFIG)


def test_initial_state():
    # Testing that state has all 0s as starting values.
    assert env.state[0] <= 1, 'does not start in box'
    assert env.state[1] <= 1, 'does not start in box'
    assert env.state[0] >= 1, 'does not start in box'
    assert env.state[1] >= 1, 'does not start in box'


def test_step():
    #TODO: finish this test
    env.reset()
    newState, reward, done, info = env.step([1, 1])

    # Test if new state is part of observation space
    assert env.observation_space.contains(
        newState), "Returned state is not part of given observation space after step"

    # Test to see if returned reward is a float
    assert type(reward) == float, "Reward is not a float"

    assert reward == 0.

    expected_state = [20/3, 4., 4.]
    for i in range(len(env.state)):
        assert env.state[i] == expected_state[i], "New state does not match expected state at index {}".format(
            i)

    # Do step again
    newState, reward, done, info = env.step([1, 1])

    # Test if new state is part of observation space
    assert env.observation_space.contains(
        newState), "Returned state is not part of given observation space after step"

    assert reward == 1.

    expected_state = [20/3 - 2, 1., 2.]
    for i in range(len(env.state)):
        assert env.state[i] == expected_state[i], "New state does not match expected state at index {}".format(
            i)
    check_env(env, skip_render_check=True)


def test_bad_action():
    # Testing to see if action not in action space raises an exception
    with pytest.raises(AssertionError):
        env.step(
            [])


def test_reset():
    env.reset()
    assert env.timestep == 0, "Timestep not set to 0 on reset"
    for i in range(len(env.state)):
        assert env.state[i] - env.starting_state[i] < 1, "State not set back to starting state on reset at index {}".format(
            i)
