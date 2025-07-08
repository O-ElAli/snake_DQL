import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
import cv2 

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, render_mode=None):
        super(SnakeEnv, self).__init__()

        self.window_x = 720
        self.window_y = 480
        self.block_size = 10
        self.snake_speed = 15

        self.action_space = spaces.Discrete(4)  # 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        self.observation_space = spaces.Box(low=0, high=255, shape=(3, 84, 84), dtype=np.uint8)

        self.render_mode = render_mode
        self.window = None
        self.clock = None

        self._init_game()

    def _init_game(self):
        self.snake_position = [100, 50]
        self.snake_body = [[100, 50], [90, 50], [80, 50]]
        self.fruit_index = 0
        self.fruits = [
            [100, 100], [300, 150], [500, 100],
            [200, 250], [400, 200], [600, 300],
            [150, 350], [350, 300], [550, 250],
            [100, 400], [250, 150], [450, 350],
            [300, 50],  [600, 150], [200, 50]
        ]
        self.fruit_position = self.fruits[self.fruit_index]
        self.direction = 'RIGHT'
        self.score = 0
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.snake = [(5, 5)]
        self.direction = random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])
        self.score = 0
        self.done = False
        self._init_game()
        self.frame_count = 0

        obs = self._get_obs()
        return obs, {}


    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        prev_distance = np.linalg.norm(np.array(self.snake_position) - np.array(self.fruit_position))

        # Update direction
        if action == 0 and self.direction != 'DOWN':
            self.direction = 'UP'
        elif action == 1 and self.direction != 'UP':
            self.direction = 'DOWN'
        elif action == 2 and self.direction != 'RIGHT':
            self.direction = 'LEFT'
        elif action == 3 and self.direction != 'LEFT':
            self.direction = 'RIGHT'

        # Move snake
        x, y = self.snake_position
        if self.direction == 'UP':
            y -= self.block_size
        elif self.direction == 'DOWN':
            y += self.block_size
        elif self.direction == 'LEFT':
            x -= self.block_size
        elif self.direction == 'RIGHT':
            x += self.block_size
        self.snake_position = [x, y]
        self.snake_body.insert(0, list(self.snake_position))

        reward = 0.1  # baseline reward per step (encourages survival)

        new_distance = np.linalg.norm(np.array([x, y]) - np.array(self.fruit_position))
        reward += (prev_distance - new_distance) * 0.2  # reward for getting closer to fruit

        # === New Tail-Safety Penalty ===
        # Encourage not getting too close to self
        for body_segment in self.snake_body[1:]:
            dist = np.linalg.norm(np.array(self.snake_position) - np.array(body_segment))
            if dist < self.block_size * 1.5:
                reward -= 0.3  # small penalty for dangerous proximity
                break

        # === Fruit Collection ===
        if self.snake_position == self.fruit_position:
            reward += 10
            self.score += 10
            self.fruit_index += 1
            if self.fruit_index < len(self.fruits):
                self.fruit_position = self.fruits[self.fruit_index]
            else:
                self.done = True
        else:
            self.snake_body.pop()

        # === Collision Detection ===
        if (x < 0 or x >= self.window_x or
            y < 0 or y >= self.window_y or
            self.snake_position in self.snake_body[1:]):
            reward = -100
            self.done = True

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, self.done, False, {}


    def _get_obs(self):
        if self.window is None:
            # Create a surface to render the game if window doesn't exist yet
            surface = pygame.Surface((self.window_x, self.window_y))
            surface.fill((0, 0, 0))
            for pos in self.snake_body:
                pygame.draw.rect(surface, (0, 255, 0), pygame.Rect(pos[0], pos[1], self.block_size, self.block_size))
            pygame.draw.rect(surface, (255, 255, 255), pygame.Rect(self.fruit_position[0], self.fruit_position[1], self.block_size, self.block_size))
        else:
            surface = self.window  # if you have a pygame window, grab from it directly

        # Get pixel data (shape: (width, height, channels))
        frame = pygame.surfarray.array3d(surface)  # returns shape (width, height, 3)
        frame = np.transpose(frame, (1, 0, 2))    # transpose to (height, width, 3) = (window_y, window_x, 3)

        # Resize to 84x84 as expected by your model
        small_frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)

        # Convert to channel-first (3, 84, 84) format for stable-baselines3
        obs = np.transpose(small_frame, (2, 0, 1))

        return obs

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            pygame.display.set_caption("Snake RL")
            self.window = pygame.display.set_mode((self.window_x, self.window_y))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.window.fill((0, 0, 0))
        for pos in self.snake_body:
            pygame.draw.rect(self.window, (0, 255, 0), pygame.Rect(pos[0], pos[1], self.block_size, self.block_size))
        pygame.draw.rect(self.window, (255, 255, 255), pygame.Rect(self.fruit_position[0], self.fruit_position[1], self.block_size, self.block_size))
        pygame.display.flip()
        self.clock.tick(self.snake_speed)

    def render(self):
        self._render_frame()

    def close(self):
        if self.window is not None:
            pygame.quit()
