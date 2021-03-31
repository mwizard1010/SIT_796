import gym
import numpy as np
import base64
import io
import IPython

from PIL import Image as Image
import matplotlib.pyplot as plt
import copy
MAX_REWARD = 5000000

class World(gym.Env):
    def __init__(self, position=[0, 0], max_moves=30, grid_size=[2, 2], dirt=[[1, 1], [1, 1]]):
        """
            Defaults grid: 2 x 2.
            Map: [[1, 1],
                  [1, 1]].
        """
        metadata = {'render.mode': ['human']}
        super(World, self).__init__()

        self.initial_position = position
        self.max_moves = max_moves
        self.grid_size = grid_size
        self.initial_dirt = dirt
        self.reward_range = (0, MAX_REWARD)

    def reset(self):
        self.score = 0
        self.move = 0
        self.position = self.initial_position
        self.current_scat = None
        self.dirt = copy.deepcopy(self.initial_dirt)

        observation = self._next_observation()
        self.visited = set()
        self.visited.add((self.position[0], self.position[1]))

        return observation

    def step(self, action):
        self.perform_action(action)
        reward = self.score 
        observation = self._next_observation()
        done = self.move == self.max_moves
        info = {}

        return observation, reward, done, info

    def _next_observation(self):
        obs = {'C': 0, 'R' : 0, 'L' : 0, 'U' : 0, 'D' : 0}
        obs['C'] = self.dirt[self.position[0]][self.position[1]]
        if self._crosses_boundary('R') == False:
            obs['R'] = self.dirt[self.position[0]][self.position[1] + 1]
        if self._crosses_boundary('L') == False:
            obs['L'] = self.dirt[self.position[0]][self.position[1] - 1]
        if self._crosses_boundary('U') == False:
            obs['U'] = self.dirt[self.position[0] - 1][self.position[1]]
        if self._crosses_boundary('D') == False:
            obs['D'] = self.dirt[self.position[0] + 1][self.position[1]]
        next_action = self._action_space()

        return [next_action, self.position, obs]

    def perform_action(self, action):
        self.move += 1
        if self._crosses_boundary(action):
            self.score = -0.1
            return

        if action == 'R':
            self.position[1] += 1
        if action == 'L':
            self.position[1] -= 1
        if action == 'U':
            self.position[0] -= 1
        if action == 'D':
            self.position[0] += 1
        if(self.dirt[self.position[0]][self.position[1]]) > 0:
            self.score = self.dirt[self.position[0]][self.position[1]]
            self.dirt[self.position[0]][self.position[1]] = 0
        else:
            self.score = 0
        self.visited.add((self.position[0], self.position[1]))

    def _crosses_boundary(self, action):
        """This function checks if action taken by the agent will cross boundary.
        Returns:
            boolean: True if boundary will be crossed
        """

        if action == 'R':
            if self.position[1]+1 > self.grid_size[1]-1:
                return True
        if action == 'L':
            if self.position[1]-1 < 0:
                return True
        if action == 'U':
            if self.position[0]-1 < 0:
                return True
        if action == 'D':
            if self.position[0]+1 > self.grid_size[0]-1:
                return True
        return False

    def _action_space(self):
        return ['R', 'L', 'U', 'D']
    def _action_prob(self):
        return ['0.1', '0.1', '0.4', '0.4']

    def render(self):
        # if self.move % 5 == 0:
        #     self.print_dirt()

        fig = plt.figure(1)
        ax = fig.add_subplot(111)

        # Major ticks
        ax.set_xticks(np.arange(0, self.grid_size[1], 1))
        ax.set_yticks(np.arange(0, self.grid_size[0], 1))

        if(self.current_scat != None):
            self.current_scat.remove() # clear
        plt.imshow(self.dirt)
        self.current_scat =  ax.scatter(self.position[1], self.position[0], color='red') # draw current position
        plt.pause(0.01)
        plt.draw()


    def positio_function(self):
        anim = animation.FuncAnimation(fig, drawframe, frames=100, interval=20, blit=True)

    def print_dirt(self):
        """This function prints the current world representation with dirt in
        each tile
        """
        part = x[:self.position[0]]
        print()
        for row in part:
            print(*row, sep=", ")

        current = self.dirt[self.position[0]]
        print(*current[:self.position[1]], sep=", ", end=" ")
        print("["+str(self.dirt[self.position[0]][self.position[1]])
              + "]", end=" ")
        print(*current[self.position[1]+1:], sep=", ")

        part = self.dirt[self.position[0]+1:]
        for row in part:
            print(*row, sep=", ")
        print()
