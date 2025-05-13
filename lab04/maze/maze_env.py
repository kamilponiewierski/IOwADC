import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MazeEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.size = 10
        self.observation_space = spaces.Discrete(self.size * self.size)
        self.action_space = spaces.Discrete(4)

        self.maze = np.zeros((self.size, self.size), dtype=np.uint8)

        # Ściany (czarne pola na obrazku)
        walls = [
            [0,1], [0,2], [0,3], [0,4], [0,6], [0,8],
            [1,0], [1,5], [1,6], [1,8],
            [2,0], [2,1], [2,4], [2,8],
            [3,0], [3,1], [3,4], [3,5], [3,8],
            [4,2], [4,3], [4,4],
            [5,1], [5,2], [5,3], [5,4],
            [6,0], [6,2], [6,4], [6,5],
            [7,0], [7,1], [7,2], [7,5], [7,7],
            [8,0], [8,4], [8,5], [8,6], [8,8],
            [9,2], [9,4], [9,6], [9,7], [9,8],
        ]
        for y, x in walls:
            self.maze[y, x] = 1

        # BONUSY (zielone)
        self.bonus_points = [
            [1,2], [2,5], [4,0], [4,1], [5,0]
        ]

        # PUŁAPKI (czerwone)
        self.penalty_points = [
            [2,4], [5,5], [6,3], [7,3], [8,1], [8,3], [9,3], [9,5]
        ]

        self.reset()

    def reset(self, seed=None, options=None):
        self.agent_pos = [0, 0]
        self.goal_pos = [9, 9]
        self.visited_bonus = set()
        return self._get_state(), {}

    def _get_state(self):
        return self.agent_pos[0] * self.size + self.agent_pos[1]

    def step(self, action):
        y, x = self.agent_pos
        new_pos = [y, x]

        # Ruch agenta
        if action == 0 and y > 0:
            new_pos = [y - 1, x]
        elif action == 1 and x < self.size - 1:
            new_pos = [y, x + 1]
        elif action == 2 and y < self.size - 1:
            new_pos = [y + 1, x]
        elif action == 3 and x > 0:
            new_pos = [y, x - 1]

        # Sprawdź czy pole jest przejściowe
        if self.maze[new_pos[0], new_pos[1]] == 0:
            self.agent_pos = new_pos

        reward = -0.1  # kara za każdy krok
        done = False

        if self.agent_pos in self.bonus_points and tuple(self.agent_pos) not in self.visited_bonus:
            reward += 2.0
            self.visited_bonus.add(tuple(self.agent_pos))
            print(f"✅ Bonus zebrany w {self.agent_pos}!")

        elif self.agent_pos in self.penalty_points:
            reward += -2.0
            print(f"⚠️ Agent wszedł w pułapkę w {self.agent_pos}!")

        if self.agent_pos == self.goal_pos:
            # Kara za brak zebranych bonusów
            missing_bonus = len([pt for pt in self.bonus_points if tuple(pt) not in self.visited_bonus])
            penalty = missing_bonus * 2.0
            reward += 3.0 - penalty
            done = True

            if missing_bonus > 0:
                print(f"⚠️ Agent nie zebrał {missing_bonus} bonusów – kara: -{penalty}")
            else:
                print("✅ Agent zebrał wszystkie bonusy i dotarł do celu!")

        return self._get_state(), reward, done

