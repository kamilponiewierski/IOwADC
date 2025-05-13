import pygame

pygame.init()
screen = pygame.display.set_mode((400, 400))
pygame.display.set_caption("Test")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    screen.fill((255, 255, 255))  # biały ekran
    pygame.display.flip()

pygame.quit()
