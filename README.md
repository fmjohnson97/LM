# LM
Code for the LM project collaboration 

### Initial Stock Experiments 
* stockSynthesis.py   - simple LSTM close price prediction given the corresponding open price 
* predictOpen.py      - predict the next open price given the previous one
* LSTM_Sequential.py  - predict an open price sequence given a sequence of previous open prices
* stockDataset.py     - manipulate kaggle stock data into sequences of open prices
* stockAgent.py       - reinforcement learning agent to learn a multiplier to produce the next open price
* stockRL             - runner code for the Stock Agent to predict open prices
* pytorchSequentialPredict.py - LSTM_Sequential.py but implemented in pytorch instead of keras

### Q Learning and Feudal RL Maze Experiments
* maze_2d_q_learning.py - solve a maze using Q learning
* maze_env.py - creates the base maze environment for these experiments
* maze_view_2D.py - helper for maze_env.py
* maze_2d_dqn.py - solve a maze using Deep Q learning
* maze_multiscale_qlearning.py - solve a maze using Feudal Q learning
* multi_maze.py - creates the environment for the feudal learning experiments 
* maze2d_002.npy - maze file for the manager (2x2 maze)
* maze2d_003.npy - maze file for the worker (4x4 maze)
* maze_qmodels.py - contains code for the worker and manager networks for maze_multiscale_qlearning.py
* maze_multiscale_dqn.py - solve a maze using Feudal Deep Q Networks 
* maze_dqn_models.py - contains code for the worker and manager networks for maze_multiscale_dqn.py

### Stock Agent Experiments
