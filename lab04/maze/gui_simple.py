import pygame
import sys
import numpy as np
import time

from maze_env import MazeEnv
from agent import QLearningAgent

CELL_SIZE = 30
MARGIN = 2
FPS = 3

# Kolory
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
AGENT_COLOR = (255, 165, 0)
START_COLOR = (0, 255, 255)
GOAL_COLOR = (255, 215, 0)
BONUS_COLOR = (0, 200, 0)
PENALTY_COLOR = (200, 0, 0)
BG_COLOR = (200, 200, 200)

def draw_grid(screen, maze, agent_pos, start_pos, goal_pos, bonus_points, penalty_points, collected_bonus):
    for y in range(maze.shape[0]):
        for x in range(maze.shape[1]):
            rect = pygame.Rect(
                x * (CELL_SIZE + MARGIN),
                y * (CELL_SIZE + MARGIN),
                CELL_SIZE,
                CELL_SIZE
            )

            pos = [y, x]

            if pos == agent_pos:
                color = AGENT_COLOR
            elif pos == start_pos:
                color = START_COLOR
            elif pos == goal_pos:
                color = GOAL_COLOR
            elif pos in bonus_points and tuple(pos) not in collected_bonus:
                color = BONUS_COLOR
            elif pos in penalty_points:
                color = PENALTY_COLOR
            elif maze[y, x] == 1:
                color = BLACK
            else:
                color = WHITE

            pygame.draw.rect(screen, color, rect)

def main():
    env = MazeEnv()
    agent = QLearningAgent(env)

    # Wczytaj Q-tabelę z treningu
    try:
        agent.q_table = np.load("q_table.npy")
    except FileNotFoundError:
        print("Błąd: brak pliku q_table.npy. Uruchom najpierw train.py.")
        sys.exit()

    best_path = agent.find_best_path()
    collected_bonus = set()
    visited_penalties = set()

    pygame.init()
    screen_size = env.size * (CELL_SIZE + MARGIN)
    screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption("Agent 10x10 - Bonusy i Pułapki")
    clock = pygame.time.Clock()

    step = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if step < len(best_path):
            agent_pos = best_path[step]
            step += 1

            if agent_pos in env.bonus_points and tuple(agent_pos) not in collected_bonus:
                collected_bonus.add(tuple(agent_pos))
                print(f"✅ Bonus zebrany w {agent_pos}!")

            if agent_pos in env.penalty_points and tuple(agent_pos) not in visited_penalties:
                visited_penalties.add(tuple(agent_pos))
                print(f"⚠️  Agent wszedł w pułapkę w {agent_pos}!")
        else:
            time.sleep(2)
            pygame.quit()
            sys.exit()

        screen.fill(BG_COLOR)
        draw_grid(
            screen,
            env.maze,
            agent_pos,
            [0, 0],
            env.goal_pos,
            env.bonus_points,
            env.penalty_points,
            collected_bonus
        )
        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()
