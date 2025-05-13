import pygame
import sys
import numpy as np
from maze_env import MazeEnv
from agent import QLearningAgent

CELL_SIZE = 30
MARGIN = 2
FPS = 5

# Inicjalizacja Pygame
pygame.init()
try:
    AGENT_IMG = pygame.transform.scale(pygame.image.load("images/agent.png"), (CELL_SIZE, CELL_SIZE))
    GOAL_IMG = pygame.transform.scale(pygame.image.load("images/goal.webp"), (CELL_SIZE, CELL_SIZE))
    WALL_IMG = pygame.transform.scale(pygame.image.load("images/brick.jpg"), (CELL_SIZE, CELL_SIZE))
    BONUS_IMG = pygame.transform.scale(pygame.image.load("images/bonus.webp"), (CELL_SIZE, CELL_SIZE))
    PENALTY_IMG = pygame.transform.scale(pygame.image.load("images/zombie.webp"), (CELL_SIZE, CELL_SIZE))
    GRASS_IMG = pygame.transform.scale(pygame.image.load("images/grass.jpg"), (CELL_SIZE, CELL_SIZE))
    START_IMG = pygame.transform.scale(pygame.image.load("images/start.jpg"), (CELL_SIZE, CELL_SIZE))
    BONUS_COLLECTED_IMG = pygame.transform.scale(pygame.image.load("images/bonus_collected.png"), (CELL_SIZE, CELL_SIZE))
except Exception as e:
    print("Błąd wczytywania obrazków:", e)
    sys.exit()

def draw_grid(screen, maze, agent_pos, goal_pos, bonus_points, penalty_points, collected_bonus_points):
    for y in range(maze.shape[0]):
        for x in range(maze.shape[1]):
            pos = [y, x]
            rect = pygame.Rect(
                x * (CELL_SIZE + MARGIN),
                y * (CELL_SIZE + MARGIN),
                CELL_SIZE,
                CELL_SIZE
            )
            screen.blit(GRASS_IMG, rect)
            if pos == [0, 0]:
                screen.blit(START_IMG, rect)
            elif pos == agent_pos:
                screen.blit(AGENT_IMG, rect)
            elif pos == goal_pos:
                screen.blit(GOAL_IMG, rect)
            elif pos in bonus_points and pos not in collected_bonus_points:
                screen.blit(BONUS_IMG, rect)
            elif pos in collected_bonus_points:
                screen.blit(BONUS_COLLECTED_IMG, rect)
            elif pos in penalty_points:
                screen.blit(PENALTY_IMG, rect)
            elif maze[y, x] == 1:
                screen.blit(WALL_IMG, rect)

def main():
    pygame.init()
    env = MazeEnv()
    agent = QLearningAgent(env)

    # Wczytaj wytrenowaną Q-tabelę
    try:
        agent.q_table = np.load("q_table.npy")
    except FileNotFoundError:
        print("Nie znaleziono pliku q_table.npy. Uruchom najpierw train.py.")
        sys.exit()

    best_path = agent.find_best_path()
    collected_bonus_points = set()

    screen_size = env.size * (CELL_SIZE + MARGIN)
    screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption("Najlepsza ścieżka agenta")
    clock = pygame.time.Clock()

    step_index = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if step_index < len(best_path):
            state = best_path[step_index]
            y, x = divmod(state, env.size)
            env.agent_pos = [y, x]

            if env.agent_pos in env.bonus_points:
                collected_bonus_points.add(tuple(env.agent_pos))

            step_index += 1
        else:
            pygame.time.wait(2000)
            pygame.quit()
            sys.exit()

        screen.fill((0, 0, 0))
        draw_grid(screen, env.maze, env.agent_pos, env.goal_pos, env.bonus_points, env.penalty_points, collected_bonus_points)
        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()
