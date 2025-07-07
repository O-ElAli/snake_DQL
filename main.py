# importing libraries
import pygame
import time

snake_speed = 15

# Window size
window_x = 720
window_y = 480

# defining colors
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)

# Initialising pygame
pygame.init()

# Initialise game window
pygame.display.set_caption('QDL Snakes')
game_window = pygame.display.set_mode((window_x, window_y))

# FPS controller
fps = pygame.time.Clock()

# Snake starting position and body
snake_position = [100, 50]
snake_body = [[100, 50], [90, 50], [80, 50], [70, 50]]

# Fixed fruit positions
fruit_positions = [
    [100, 100], [300, 150], [500, 100],
    [200, 250], [400, 200], [600, 300],
    [150, 350], [350, 300], [550, 250],
    [100, 400], [250, 150], [450, 350],
    [300, 50],  [600, 150], [200, 50]
]
fruit_index = 0  # Track which fruit to spawn
fruit_spawn = True

# Initial direction
direction = 'RIGHT'
change_to = direction

# Initial score
score = 0

# Show score function
def show_score(color, font, size):
    score_font = pygame.font.SysFont(font, size)
    score_surface = score_font.render('Score : ' + str(score), True, color)
    score_rect = score_surface.get_rect()
    game_window.blit(score_surface, score_rect)

# Game over function
def game_over():
    my_font = pygame.font.SysFont('times new roman', 50)
    game_over_surface = my_font.render('Your Score is : ' + str(score), True, red)
    game_over_rect = game_over_surface.get_rect()
    game_over_rect.midtop = (window_x / 2, window_y / 4)
    game_window.blit(game_over_surface, game_over_rect)
    pygame.display.flip()
    time.sleep(2)
    pygame.quit()
    quit()

# Main loop
while True:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                change_to = 'UP'
            if event.key == pygame.K_DOWN:
                change_to = 'DOWN'
            if event.key == pygame.K_LEFT:
                change_to = 'LEFT'
            if event.key == pygame.K_RIGHT:
                change_to = 'RIGHT'

    # Prevent reverse movement
    if change_to == 'UP' and direction != 'DOWN':
        direction = 'UP'
    if change_to == 'DOWN' and direction != 'UP':
        direction = 'DOWN'
    if change_to == 'LEFT' and direction != 'RIGHT':
        direction = 'LEFT'
    if change_to == 'RIGHT' and direction != 'LEFT':
        direction = 'RIGHT'

    # Move the snake
    if direction == 'UP':
        snake_position[1] -= 10
    if direction == 'DOWN':
        snake_position[1] += 10
    if direction == 'LEFT':
        snake_position[0] -= 10
    if direction == 'RIGHT':
        snake_position[0] += 10

    # Snake body growth and fruit check
    snake_body.insert(0, list(snake_position))

    if fruit_index < len(fruit_positions) and snake_position == fruit_positions[fruit_index]:
        score += 10
        fruit_index += 1
    else:
        snake_body.pop()

    # Fill screen
    game_window.fill(black)

    # Draw snake
    for pos in snake_body:
        pygame.draw.rect(game_window, green, pygame.Rect(pos[0], pos[1], 10, 10))

    # Draw fruit if available
    if fruit_index < len(fruit_positions):
        fruit = fruit_positions[fruit_index]
        pygame.draw.rect(game_window, white, pygame.Rect(fruit[0], fruit[1], 10, 10))
    else:
        # All fruits collected
        game_over()

    # Check for boundary collision
    if snake_position[0] < 0 or snake_position[0] > window_x - 10 or \
       snake_position[1] < 0 or snake_position[1] > window_y - 10:
        game_over()

    # Check for self collision
    for block in snake_body[1:]:
        if snake_position == block:
            game_over()

    # Show score
    show_score(white, 'times new roman', 20)

    # Update screen
    pygame.display.update()

    # Tick
    fps.tick(snake_speed)
