# AdaptiveLearning-MRL

This repo contains our experiments for testing MRL against ORSuite on the maze environment. 

After installing the requirments, you must also run

`cd gym-maze` 
`python setup.py install`

in order to install the gym-maze environment

## New or Changed Files

`run-adaptive-maze.py` runs adaptive learning models on the maze and records the data, rewards, and optimality gap for the adaptive models. 

`retrain-mrl-on-maze.py` trains mrl on the saved transition data and records the optimality gap. 

`plot-adaptive-opt-gap.py` plots the optimality gaps from saved data. 

`experiment.py` has been repurposed for the running of this experiment to also train mrl and record the transition data for the format of training MRL. 

`envs/maze.py` now includes the maze environment, which we run the experiments on. 
