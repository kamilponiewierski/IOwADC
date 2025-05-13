import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MazeEnv(gym.Env):
    def __init__(self, size=20, seed=42):
        super(MazeEnv, self).__init__()
        self.size = size
        self.seed = seed
        self.rng = np.random.default_rng(seed)  # deterministyczny generator losowy
        self.observation_space = spaces.Discrete(size * size)
        self.action_space = spaces.Discrete(4)  # 0=up, 1=right, 2=down, 3=left
        self.maze = self._generate_maze()
        self.reset()

    def _generate_maze(self):
        maze = np.zeros((self.size, self.size), dtype=np.uint8)
        for _ in range(int(self.size * self.size * 0.2)):  # 20% ścian
            y, x = self.rng.integers(0, self.size), self.rng.integers(0, self.size)
            if [y, x] not in [[0, 0], [self.size - 1, self.size - 1]]:
                maze[y, x] = 1  # ściana
        return maze

    def reset(self):
        self.agent_pos = [0, 0]
        self.goal_pos = [self.size - 1, self.size - 1]

        # Stałe lokalizacje bonusów i kar
        self.bonus_points = [[5, 5], [10, 10]]  # bonusy
        self.penalty_points = [[15, 15], [7, 12], [13, 3]]  # zombie
        self.visited_bonus = set()

        return self._get_state(), {}

    def _get_state(self):
        return self.agent_pos[0] * self.size + self.agent_pos[1]

    def step(self, action):
        y, x = self.agent_pos
        new_pos = [y, x]

        if action == 0 and y > 0:
            new_pos = [y - 1, x]
        elif action == 1 and x < self.size - 1:
            new_pos = [y, x + 1]
        elif action == 2 and y < self.size - 1:
            new_pos = [y + 1, x]
        elif action == 3 and x > 0:
            new_pos = [y, x - 1]

        # Ruch tylko jeśli nie ma ściany
        if self.maze[new_pos[0], new_pos[1]] == 0:
            self.agent_pos = new_pos

        reward = -0.01  # domyślna lekka kara za krok
        done = False

        # Trafienie na zombie (kara)
        if self.agent_pos in self.penalty_points:
            reward = -1.0
            done = True  # opcjonalnie kończy epizod

        # Zbieranie bonusu
        elif self.agent_pos in self.bonus_points and tuple(self.agent_pos) not in self.visited_bonus:
            reward = 1.0
            self.visited_bonus.add(tuple(self.agent_pos))

        # Dotarcie do celu
        elif self.agent_pos == self.goal_pos:
            if len(self.visited_bonus) == len(self.bonus_points):
                reward = 2.0  # za wszystkie bonusy + cel
            else:
                reward = -0.5  # za ignorowanie bonusów
            done = True

        return self._get_state(), reward, done

    def render(self):
        grid = np.copy(self.maze).astype(str)
        grid[grid == '0'] = '.'
        grid[grid == '1'] = '#'
        for y, x in self.bonus_points:
            grid[y, x] = '+'
        for y, x in self.penalty_points:
            grid[y, x] = '-'
        y, x = self.agent_pos
        grid[y, x] = 'A'
        gy, gx = self.goal_pos
        grid[gy, gx] = 'G'
        print("\n".join(" ".join(row) for row in grid))
        print()
