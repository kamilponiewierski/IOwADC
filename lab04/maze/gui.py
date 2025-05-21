import pygame
import sys
import numpy as np
import time

from maze_env import MazeEnv
from agent import QLearningAgent

CELL_SIZE = 35
MARGIN = 2
FPS = 6  # tempo animacji

pygame.init()

# Wczytaj obrazy
def load_img(name):
    return pygame.transform.scale(pygame.image.load(f"images/{name}"), (CELL_SIZE, CELL_SIZE))

MARIO_FRAMES = [
    load_img("mario_1.png"),
    load_img("mario_2.png"),
    load_img("mario_3.png"),
]

START_IMG = load_img("start1.jpg")
GOAL_IMG = load_img("goal1.gif")
WALL_IMG = load_img("brick1.jpg")
GRASS_IMG = load_img("floor.avif")
ZOMBIE_IMG = load_img("mob.jpg")

def draw_grid(screen, env, agent_pos, frame_index):
    for y in range(env.size):
        for x in range(env.size):
            rect = pygame.Rect(
                x * (CELL_SIZE + MARGIN),
                y * (CELL_SIZE + MARGIN),
                CELL_SIZE,
                CELL_SIZE
            )

            screen.blit(GRASS_IMG, rect)

            if [y, x] == [0, 0]:
                screen.blit(START_IMG, rect)
            elif [y, x] == env.goal_pos:
                screen.blit(GOAL_IMG, rect)
            elif [y, x] == agent_pos:
                screen.blit(MARIO_FRAMES[frame_index], rect)
            elif [y, x] in env.penalty_points:
                screen.blit(ZOMBIE_IMG, rect)
            elif env.maze[y, x] == 1:
                screen.blit(WALL_IMG, rect)

def main():
    env = MazeEnv(size=20)
    agent = QLearningAgent(env)

    try:
        agent.q_table = np.load("q_table.npy")
    except FileNotFoundError:
        print("❌ Brak pliku q_table.npy – uruchom train.py.")
        sys.exit()

    path = agent.find_best_path()

    screen_size = env.size * (CELL_SIZE + MARGIN)
    screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption("🍄 Mario Maze Agent")
    clock = pygame.time.Clock()

    step = 0
    frame_index = 0
    frame_timer = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if step < len(path):
            agent_pos = path[step]
            step += 1
        else:
            time.sleep(2)
            pygame.quit()
            sys.exit()

        frame_timer += 1
        if frame_timer >= 5:
            frame_index = (frame_index + 1) % len(MARIO_FRAMES)
            frame_timer = 0

        screen.fill((0, 0, 0))
        draw_grid(screen, env, agent_pos, frame_index)
        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()
