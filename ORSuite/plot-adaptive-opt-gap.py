import sys
sys.path.append('/Users/lowell/MRL/Algorithm')
sys.path.append('/Users/lowell/MRL/Maze')

import pandas as pd 
from maze_functions import plot_paths
import matplotlib.pyplot as plt

# choose maze file to use 
df = pd.read_csv('data/maze_ADAMB_0.25/data.csv')

fig, ax = plt.subplots()
# in order to keep things working in the rest of the repo,
# we just monkey patch the memory column to be the opt gap for
# the maze experiment here. 
opt_gap = df['memory'][df['memory'] > 0]

ax.plot(opt_gap)
ax.set_title("Optimality gap vs episode number")
ax.set_xlabel("Episode")
ax.set_ylabel("Reward")
ax.set_ylim(0, max(opt_gap))
plt.show()
