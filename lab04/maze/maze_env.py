import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MazeEnv(gym.Env):
    def __init__(self, size=20):
        super().__init__()
        self.size = size
        self.observation_space = spaces.Discrete(self.size * self.size)
        self.action_space = spaces.Discrete(4)

        self.maze = np.zeros((self.size, self.size), dtype=np.uint8)


        self.walls = [
            [0, 1], [0, 2], [0, 3], [0, 5], [0, 7], [0, 8], [0, 9], [0, 10], [0, 12], [0, 13], [0, 15], [0, 16], [0, 18],
            [1, 3], [1, 5], [1, 10], [1, 12], [1, 15], [1, 18],
            [2, 1], [2, 2], [2, 3], [2, 5], [2, 7], [2, 9], [2, 10], [2, 12], [2, 15], [2, 17], [2, 18],
            [3, 7], [3, 9], [3, 12], [3, 17],
            [4, 1], [4, 2], [4, 3], [4, 4], [4, 7], [4, 9], [4, 12], [4, 14], [4, 15], [4, 16], [4, 17],
            [5, 4], [5, 9], [5, 12],
            [6, 1], [6, 2], [6, 3], [6, 4], [6, 6], [6, 7], [6, 8], [6, 9], [6, 11], [6, 12], [6, 14], [6, 15],
            [7, 6], [7, 11], [7, 15],
            [8, 1], [8, 2], [8, 3], [8, 4], [8, 6], [8, 8], [8, 9], [8, 11], [8, 13], [8, 14], [8, 15], [8, 16],
            [9, 8], [9, 13], [9, 16],
            [10, 1], [10, 3], [10, 4], [10, 6], [10, 8], [10, 9], [10, 11], [10, 13], [10, 16], [10, 18],
            [11, 3], [11, 6], [11, 9], [11, 13], [11, 18],
            [12, 1], [12, 2], [12, 3], [12, 6], [12, 8], [12, 9], [12, 10], [12, 13], [12, 14], [12, 15], [12, 16],
            [13, 6], [13, 10], [13, 13], [13, 16],
            [14, 1], [14, 2], [14, 3], [14, 4], [14, 6], [14, 8], [14, 10], [14, 13], [14, 16], [14, 17], [14, 18],
            [15, 8], [15, 10], [15, 13], [15, 18],
            [16, 1], [16, 2], [16, 4], [16, 5], [16, 6], [16, 8], [16, 10], [16, 12], [16, 13], [16, 15], [16, 16],
            [17, 6], [17, 10], [17, 15],
            [18, 1], [18, 2], [18, 4], [18, 6], [18, 8], [18, 10], [18, 12], [18, 13], [18, 15], [18, 17],
            [19, 1], [19, 4], [19, 8], [19, 12], [19, 15], [19, 17]
        ]

        for y, x in self.walls:
            self.maze[y, x] = 1

        # 👾 Zombie – przeszkody karne
        self.penalty_points = [[2, 6], [4, 13], [6, 13], [8, 10], [10, 12], [12, 6], [18, 16]]

        self.goal_pos = [self.size - 1, self.size - 1]
        self.reset()

    def reset(self, seed=None, options=None):
        self.agent_pos = [0, 0]
        return self._get_state(), {}

    def _get_state(self):
        return self.agent_pos[0] * self.size + self.agent_pos[1]

    def step(self, action):
        y, x = self.agent_pos
        new_pos = [y, x]

        if action == 0 and y > 0: new_pos = [y - 1, x]       # up
        elif action == 1 and x < self.size - 1: new_pos = [y, x + 1]  # right
        elif action == 2 and y < self.size - 1: new_pos = [y + 1, x]  # down
        elif action == 3 and x > 0: new_pos = [y, x - 1]     # left

        if self.maze[new_pos[0], new_pos[1]] == 0:
            self.agent_pos = new_pos

        reward = -0.01
        done = False

        if self.agent_pos in self.penalty_points:
            reward += -0.5
            print(f"⚠️ Zombie! {self.agent_pos}")

        if self.agent_pos == self.goal_pos:
            reward += 3.0
            print("🎯 Agent dotarł do celu!")
            done = True

        return self._get_state(), reward, done
