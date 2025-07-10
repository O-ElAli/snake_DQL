import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, render_mode=None):
        super(SnakeEnv, self).__init__()

        self.window_x = 720
        self.window_y = 480
        self.block_size = 10
        self.snake_speed = 15

        self.action_space = spaces.Discrete(4)  # 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(11,), dtype=np.float32)

        self.render_mode = render_mode
        self.window = None
        self.clock = None

        self._init_game()

    def _init_game(self):
        self.snake_position = [100, 50]
        self.snake_body = [[100, 50], [90, 50], [80, 50]]
        self.fruit_position = self._generate_fruit_position()
        self.direction = random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])
        self.score = 0
        self.done = False
        self.steps_in_episode = 0  # Track episode length

    def _generate_fruit_position(self):
        """Generate random fruit position not occupied by snake"""
        while True:
            x = random.randrange(0, self.window_x, self.block_size)
            y = random.randrange(0, self.window_y, self.block_size)
            fruit_pos = [x, y]
            if fruit_pos not in self.snake_body:
                return fruit_pos

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_game()
        self.frame_count = 0
        return self._get_obs(), {}

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        self.steps_in_episode += 1

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

        reward = 0.1  # base reward

        # Distance improvement reward
        new_distance = np.linalg.norm(np.array([x, y]) - np.array(self.fruit_position))
        reward += (prev_distance - new_distance) * 0.2

        # Tail proximity penalty
        for body_segment in self.snake_body[1:]:
            dist = np.linalg.norm(np.array(self.snake_position) - np.array(body_segment))
            if dist < self.block_size * 1.5:
                reward -= 0.5
                if dist < self.block_size * 0.9:
                    reward -= 1.0
                break

        # Eat fruit
        if self.snake_position == self.fruit_position:
            reward += 10
            self.score += 10
            self.fruit_position = self._generate_fruit_position()
        else:
            if len(self.snake_body) > 1:
                self.snake_body.pop()

        # Collision detection
        if (x < 0 or x >= self.window_x or
            y < 0 or y >= self.window_y or
            self.snake_position in self.snake_body[1:]):
            reward = -100
            self.done = True

        # Max step limit to avoid freezing
        if self.steps_in_episode > 1000:
            self.done = True
            print("⚠️ Episode terminated due to step limit.")

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, self.done, False, {}

    def _get_obs(self):
        x, y = self.snake_position
        fx, fy = self.fruit_position

        def norm(v, max_v):
            return v / max_v

        delta_x = norm(fx - x, self.window_x)
        delta_y = norm(fy - y, self.window_y)

        dir_up = int(self.direction == 'UP')
        dir_down = int(self.direction == 'DOWN')
        dir_left = int(self.direction == 'LEFT')
        dir_right = int(self.direction == 'RIGHT')

        ahead = self._get_next_position()
        danger = 0
        if (ahead[0] < 0 or ahead[0] >= self.window_x or
            ahead[1] < 0 or ahead[1] >= self.window_y or
            ahead in self.snake_body):
            danger = 1

        return np.array([
            norm(x, self.window_x),
            norm(y, self.window_y),
            norm(fx, self.window_x),
            norm(fy, self.window_y),
            delta_x,
            delta_y,
            dir_up,
            dir_down,
            dir_left,
            dir_right,
            danger
        ], dtype=np.float32)

    def _get_next_position(self):
        x, y = self.snake_position
        if self.direction == 'UP':
            y -= self.block_size
        elif self.direction == 'DOWN':
            y += self.block_size
        elif self.direction == 'LEFT':
            x -= self.block_size
        elif self.direction == 'RIGHT':
            x += self.block_size
        return [x, y]

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            pygame.display.set_caption("Snake RL")
            self.window = pygame.display.set_mode((self.window_x, self.window_y))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.window.fill((0, 0, 0))

        for pos in self.snake_body:
            pygame.draw.rect(self.window, (0, 255, 0),
                             pygame.Rect(pos[0], pos[1], self.block_size, self.block_size))

        pygame.draw.rect(self.window, (255, 255, 255),
                         pygame.Rect(self.fruit_position[0], self.fruit_position[1],
                                     self.block_size, self.block_size))

        font = pygame.font.SysFont('Arial', 20)
        apple_text = font.render(f'Apples: {self.score // 10}', True, (255, 255, 255))
        self.window.blit(apple_text, (10, 10))

        pygame.display.flip()
        self.clock.tick(self.snake_speed)

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None
