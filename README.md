# LM
Code for the LM project collaboration 

### Initial Stock Experiments 
* __stockSynthesis.py__   - simple LSTM close price prediction given the corresponding open price 
* __predictOpen.py__      - predict the next open price given the previous one
* __LSTM_Sequential.py__  - predict an open price sequence given a sequence of previous open prices
* __stockDataset.py__     - manipulate kaggle stock data into sequences of open prices
* __stockAgent.py__       - reinforcement learning agent to learn a multiplier to produce the next open price
* __stockRL.py__             - runner code for the Stock Agent to predict open prices
* __pytorchSequentialPredict.py__ - LSTM_Sequential.py but implemented in pytorch instead of keras

### Q Learning and Feudal RL Maze Experiments
* __maze_2d_q_learning.py__ - solve a maze using Q learning
* __maze_env.py__ - creates the base maze environment for these experiments
* __maze_view_2D.py__ - helper for maze_env.py
* __maze_2d_dqn.py__ - solve a maze using Deep Q learning
* __maze_multiscale_qlearning.py__ - solve a maze using Feudal Q learning
* __multi_maze.py__ - creates the environment for the feudal learning experiments 
* __maze2d_002.npy__ - maze file for the manager (2x2 maze)
* __maze2d_003.npy__ - maze file for the worker (4x4 maze)
* __maze_qmodels.py__ - contains code for the worker and manager networks for maze_multiscale_qlearning.py
* __maze_multiscale_dqn.py__ - solve a maze using Feudal Deep Q Networks 
* __maze_dqn_models.py__ - contains code for the worker and manager networks for maze_multiscale_dqn.py

### Stock Agent Portfolio Experiments
* __stock_qlearning.py__ - portfolio experiment with a Q learning agent
* __stock_env.py__ - stock market environment for stock_qlearning.py
* __stock_dqn.py__ - portfolio experiment with a DQN agent
* __stock_env_dqn.py__ - stock market environment for stock_dqn.py
* __multiscale_q_stock.py__ - portfolio experiment with feudal Q learning 
* __multi_stock_env.py__ - stock market environment for multiscale_q_stock.py
* __stock_qmodles.py__ - contains the worker and manager code for multiscale_q_stock.py
* __stock_multi_dqn.py__ - portfolio experiment with feudal DQNs
* __multi_stock_env_dqn.py__ - stock market environment for stock_multi_dqn.py
* __stock_dqn_models.py__ - contains the worker, manager, and other helper classes for multi_stock_env_dqn.py
* __multi_transaction_feudal.py__ - feudal portfolio experiment where the manager chooses from several workers with different levels of temporal abstraction
* __multi_trans_stock_env.py__ - stock market environment for multi_transaction_feudal.py
* __hard_coded_qLearning.py__ - portfolio experiment with a hard coded agent as a baseline
* __dynamicM_hardcodeW.py__ - feudal portfolio experiment where the manager chooses from several hard coded workers
* __hard_coded_agents.py__ - contains the hard coded agents for dynamicM_hardcodeW.py

### Steering Angle and TSNE Prediction
* __mutAngTSNE.py__ - use the TSNE embedding as the subroutine ID when predicting steering angles; also capable of predicting multiple steering angles out
* __predZKomanda.py__ - use a modified version of the winning udacity steering angle challenge network to predict steering angles and subroutine IDs
* __tsnePredict.py__ - predict the TSNE centroids using images (poor results)
* __udacityData.py__ - dataset class to process the udacity driving dataset
* __komanda.py__ - predict steering angles using the winning udacity steering angle challenge network
* __tsnePrevDataPred.py__ - predict the next TSNE centroid given the input data for the previous centroid (poor results)
