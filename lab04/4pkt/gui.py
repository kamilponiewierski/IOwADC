import pygame
import sys
import numpy as np
from maze_env import MazeEnv
from agent import QLearningAgent

CELL_SIZE = 25
MARGIN = 2
FPS = 5  # Spowolnij trochę, żeby ścieżka była widoczna

# Kolory
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (220, 220, 220)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 128, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
START_COLOR = (0, 255, 255)

def draw_grid(screen, maze, agent_pos, goal_pos, bonus_points, penalty_points, best_path=None):
    for y in range(maze.shape[0]):
        for x in range(maze.shape[1]):
            rect = pygame.Rect(
                x * (CELL_SIZE + MARGIN),
                y * (CELL_SIZE + MARGIN),
                CELL_SIZE,
                CELL_SIZE
            )

            color = WHITE
            if maze[y, x] == 1:
                color = BLACK
            elif [y, x] == agent_pos:
                color = BLUE
            elif [y, x] == goal_pos:
                color = GREEN
            elif [y, x] in bonus_points:
                color = YELLOW
            elif [y, x] in penalty_points:
                color = RED
            elif [y, x] == [0, 0]:
                color = START_COLOR

            pygame.draw.rect(screen, color, rect)

            # Podświetl najlepszą ścieżkę
            if best_path and (y, x) in best_path:
                pygame.draw.rect(screen, ORANGE, rect, 3)

def main():
    pygame.init()
    env = MazeEnv(size=20, seed=42)
    agent = QLearningAgent(env)
    agent.load_q_table("q_table.pkl")  # Wczytaj wytrenowaną Q-tabelę

    best_state_path = agent.find_best_path()
    best_path = [divmod(s, env.size) for s in best_state_path]

    screen_size = env.size * (CELL_SIZE + MARGIN)
    screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption("Maze Agent – Najlepsza ścieżka")
    clock = pygame.time.Clock()

    state, _ = env.reset()
    step_index = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if step_index < len(best_state_path):
            next_state = best_state_path[step_index]
            env.agent_pos = list(divmod(next_state, env.size))
            step_index += 1
        else:
            pygame.time.wait(2000)  # Zatrzymaj na końcu
            state, _ = env.reset()
            step_index = 0

        screen.fill(GRAY)
        draw_grid(screen, env.maze, env.agent_pos, env.goal_pos, env.bonus_points, env.penalty_points, best_path)
        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()
