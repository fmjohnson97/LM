import sys
from multi_maze import MultiMaze
from maze_dqn_models import Worker, Manager, Replay

# Define Constants/ other variables
NUM_EPISODES = 50000
STREAK_TO_END = 50
RENDER_MAZE = True
LOC_MAP = {(0, 0): [(0, 0), (1, 0), (0, 1), (1, 1)],
           (1, 0): [(2, 0), (3, 0), (2, 1), (3, 1)],
           (0, 1): [(0, 2), (1, 2), (0, 3), (1, 3)],
           (1, 1): [(2, 2), (3, 2), (2, 3), (3, 3)]}

# this function came with the original code
# it maps environment observations to human readible, cartesian map coordinates
def state_to_bucket(state, maze_size):
    STATE_BOUNDS = [(0, maze_size[0] - 1), (0, maze_size[1] - 1)]
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = maze_size[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (maze_size[i] - 1) * STATE_BOUNDS[i][0] / bound_width
            scaling = (maze_size[i] - 1) / bound_width
            bucket_index = int(round(scaling * state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)

# computes the new reward for the worker
def compute_reward(goal, pos, reward):
    ### this is the version of this function used for the manager giving the worker a quadrant to move to###
    if reward == 1:
        # if the target is found, return the regular reward
        return reward
    elif tuple(pos) in goal:
        # if the worker is in the manager specified quandrant, return reward/2
        # the reward is negative if you haven't found the target so reward/2 is better than just reward
        return reward / 2
    else:
        return reward

### Begin the Simulation ###

# declare the environment
man_maze_path='/home/faith/PycharmProjects/FeudalTests/gym-maze-master/gym_maze/envs/maze_samples/maze2d_002.npy'#None #ToDo: replace this with the path to a maze file
work_maze_path='/home/faith/PycharmProjects/FeudalTests/gym-maze-master/gym_maze/envs/maze_samples/maze2d_003.npy'#None #ToDo: replace this with the path to a maze file
gx=None #ToDo: None makes the goal in a random place. Change this to the x coordinate of your desired goal in the maze to have it fixed
gy=None #ToDo: None makes the goal in a random place. Change this to the y coordinate of your desired goal in the maze to have it fixed
env = MultiMaze(man_maze_path, work_maze_path, gx, gy)
# render the worker's maze (can only render one at a time)
env.render(-1)

# declare the manager and worker using their respective maze sizes and the number of actions they can take
manager = Manager(env.mazes[0].maze_size)
worker = Worker(env.mazes[1].maze_size)
m_replay=Replay()
w_replay=Replay()
num_streaks = 0

# start the training loop
for episode in range(NUM_EPISODES):

    # reset the manager/worker environments
    obv_m = env.reset(manager.maze_ind)
    obv_w = env.reset(worker.maze_ind)

    # setting the initial states
    s0m = state_to_bucket(obv_m, manager.maze_size)
    s0w = state_to_bucket(obv_w, worker.maze_size)
    # setting the initial learning rates
    lrm = manager.get_learning_rate(episode)
    lrw = worker.get_learning_rate(episode)
    # setting the initial exploration rates
    erm = manager.get_explore_rate(episode)
    erw = worker.get_explore_rate(episode)

    # initialzing the reward variables
    m_reward = 0
    w_reward = 0

    # initializing the loss variables
    m_loss=0
    w_loss=0

    # initializing the variables that keep track of whether or not the manager/worker have found the target
    done_m = False
    done_w = False

    # Loop to execute the Q learning
    for t in range(worker.max_t):

        # execute manager action if the manager has not already found it's goal
        if not done_m:
            # select an action
            action_m = manager.select_action(s0m, erm)
            # execute the action and collect the state and reward feedback
            obv_m, r_m, done_m, _ = env.step(action_m, manager.maze_ind)
            sm = state_to_bucket(obv_m, manager.maze_size)
            # increment the rewards and update the Q function values
            m_reward += r_m

            m_replay.store(s0m, sm, action_m, r_m)
            m_loss+=manager.updateDQN(m_replay)
            s0m = sm

            # pass goal (quadrant) to the worker
            worker.goal = LOC_MAP[sm]  # action_m

            # loop to let the worker move until it is in the quadrant specified by the manager
            while (s0w not in worker.goal):
                # select the worker's action
                action_w = worker.select_action(s0w, erw)
                # execute the action and collect the new state and reward
                obv_w, r_w, done_w, _ = env.step(action_w, worker.maze_ind)

                # alter the reward to take obeying the manager into account
                r_w = compute_reward(worker.goal, env.mazes[worker.maze_ind].maze_view.robot, r_w)  # action_w

                # compute new states and cumulative reward
                sw = state_to_bucket(obv_w, worker.maze_size)
                w_reward += r_w
                w_replay.store(s0w,sw, action_w, r_w)

                # update the Q tables
                w_loss+=worker.updateDQN(w_replay)
                s0w = sw

                # Render both mazes (doesn't actually work but it kind of superimposes them so you can make out what
                # the manager is doing
                if RENDER_MAZE:
                    env.render(0)  # <-- comment out this line if you don't want to see what the manager does
                    env.render(-1)

                # check if worker at target without following the manager's directions
                if done_w:
                    # update the manager with a negative reward
                    r_m = -0.1 / (manager.maze_size[0] * manager.maze_size[1])
                    m_reward += r_m
                    m_loss+=manager.updateDQN(None,s0m, sm, r_m)
                    break
        else:

            # execute the worker action loop if the manager has already found its target
            action_w = worker.select_action(s0w, erw)
            obv_w, r_w, done_w, _ = env.step(action_w, worker.maze_ind)

            # alter the reward to take obeying the manager into account
            r_w = compute_reward(worker.goal, env.mazes[worker.maze_ind].maze_view.robot, r_w)  # action_w

            # compute new states and cumulative reward
            sw = state_to_bucket(obv_w, worker.maze_size)
            w_reward += r_w
            w_replay.store(s0w,sw, action_w, r_w)

            # update the Q tables
            w_loss+=worker.updateDQN(w_replay)
            s0w = sw

            # Render tha maze
            if RENDER_MAZE:
                env.render(0)
                env.render(-1)

        # exit the system if the goal has been reached
        if env.is_game_over(-1):
            sys.exit()

        # print statistics on the episode and increment the number of streaks
        if done_w:
            print("Episode %d finished after %f time steps with total reward = (%f, %f),loss = (%f, %f) (streak %d)."
                  % (episode, t, m_reward, w_reward,m_loss,w_loss, num_streaks))

            if t <= worker.max_t / 100.0:
                num_streaks += 1
            else:
                num_streaks = 0
            break

    # It's considered done when it's solved over 50 times consecutively
    if num_streaks > STREAK_TO_END:
        break