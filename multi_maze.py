from maze_env import MazeEnv
import random


# matches the goal in the lower resolution map to the proper coordinates in the higher resolution map
def goal_match(gx, gy):
    if gx == 1:
        newx = random.randint(1, 2)
    else:
        newx = random.randint(3, 4)

    if gy == 1:
        newy = random.randint(1, 2)
    else:
        newy = random.randint(3, 4)

    return (newx, newy)


class MultiMaze(object):
    def __init__(self,man_maze_path, work_maze_path, gx=None, gy=None):
        # creates the 2x2 maze from a saved file.
        twoXtwo = MazeEnv(maze_file=man_maze_path, gx=gx, gy=gy)
        # creates the goal mapping
        [gx, gy] = twoXtwo.maze_view.goal
        gx, gy = goal_match(gx, gy)
        # creates the 4x4 maze from a saved file. YOU WILL NEED TO CHANGE THIS LINK!!!
        fourXfour = MazeEnv(maze_file=work_maze_path, gx=gx, gy=gy)
        # set class variables
        self.mazes = [twoXtwo, fourXfour]
        self.action_space = 4

    # the rest of these call functions from the regular maze on a specific maze from the array
    def render(self, ind):
        self.mazes[ind].render()

    def reset(self, ind):
        return self.mazes[ind].reset()

    def step(self, action, ind):
        return self.mazes[ind].step(action)

    def is_game_over(self, ind):
        return self.mazes[ind].maze_view.game_over

